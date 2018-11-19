import functools

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from .abstract_model import AbstractModel


class BaseModel(AbstractModel):

    def input_fn(self, params, mode=tf.estimator.ModeKeys.TRAIN):
        def generator_fn(src, tag):
            if tag is None:
                with open(src, mode="rt", encoding="utf8") as fsrc:
                    for src_line in fsrc:
                        yield parse_fn(src_line=src_line, tag_line=None)
            else:
                with open(src, mode="rt", encoding="utf8") as fsrc, open(tag, mode="rt", encoding="utf8") as ftag:
                    for src_line, tag_line in zip(fsrc, ftag):
                        yield parse_fn(src_line, tag_line)

        def parse_fn(src_line, tag_line):
            words = [w.encode() for w in src_line.strip("\n").strip().split()]
            if tag_line is None:
                tags = None
            else:
                tags = [t.encode() for t in tag_line.strip("\n").strip().split()]
                assert len(words) == len(tags)
            return ((words, len(words)), tags)

        src_file, tag_file = None, None
        if mode == tf.estimator.ModeKeys.TRAIN:
            src_file = params['train_src_file']
            tag_file = params['train_tag_file']
        elif mode == tf.estimator.ModeKeys.EVAL:
            src_file = params['eval_src_file']
            tag_file = params['eval_tag_file']
        elif mode == tf.estimator.ModeKeys.PREDICT:
            src_file = params['predict_src_file']
            tag_file = None
        if mode != tf.estimator.ModeKeys.PREDICT:
            if not src_file or not tag_file:
                raise ValueError("src file and tag file must be provided.")

        if mode == tf.estimator.ModeKeys.PREDICT:
            padded_shapes = (([None], ()), [None])
            padding_values = ((params['pad'], 0), [None])
            data_types = ((tf.string, tf.int32), None)

        else:
            padded_shapes = (([None], ()), [None])
            padding_values = ((params['pad'], 0), params['oov_tag'])
            data_types = ((tf.string, tf.int32), tf.string)

        dataset = tf.data.Dataset.from_generator(
            functools.partial(generator_fn, src_file, tag_file),
            output_shapes=padded_shapes,
            output_types=data_types)
        if params['shuffle']:
            dataset = dataset.shuffle(buffer_size=params['buff_size'],
                                      reshuffle_each_iteration=params['reshuffle_each_iteration'])
        if params['repeat']:
            dataset = dataset.repeat(params['repeat'])

        dataset = dataset.padded_batch(batch_size=params.get('batch_size', 32),
                                       padded_shapes=padded_shapes,
                                       padding_values=padding_values).prefetch(params['buff_size'])
        return dataset

    def model_fn(self, features, labels, mode, params, config):
        words, nwords = features
        # a UNK token should placed in the first row in vocab file
        words_str2idx = lookup_ops.index_table_from_file(params['src_vocab'], default_value=0)
        words_ids = words_str2idx.lookup(words)

        training = mode == tf.estimator.ModeKeys.TRAIN

        # embedding
        with tf.variable_scope("embedding", reuse=tf.AUTO_REUSE):
            variable = tf.get_variable("words_embedding",
                                       shape=(params['vocab_size'], params['embedding_size']),
                                       dtype=tf.float32)
            embedding = tf.nn.embedding_lookup(variable, words_ids)
            embedding = tf.layers.dropout(embedding, rate=params['dropout'], training=training)

        # BiLSTM
        with tf.variable_scope("bilstm", reuse=tf.AUTO_REUSE):
            inputs = tf.transpose(embedding, perm=[1, 0, 2])
            lstm_fw = tf.nn.rnn_cell.LSTMCell(params['lstm_size'])
            lstm_bw = tf.nn.rnn_cell.LSTMCell(params['lstm_size'])
            (output_fw, output_bw), (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=lstm_fw,
                cell_bw=lstm_bw,
                inputs=inputs,
                sequence_length=nwords,
                dtype=tf.float32,
                swap_memory=True,
                time_major=True)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.transpose(output, perm=[1, 0, 2])
            output = tf.layers.dropout(output, rate=params['dropout'], training=training)

        logits, predict_ids = self.decode(output, nwords, params)

        # TODO(luozhouyang) Add hooks

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = self.build_predictions(predict_ids, params)
            prediction_hooks = []
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=predictions,
                                              prediction_hooks=prediction_hooks)

        loss = self.compute_loss(logits, labels, nwords, params)

        if mode == tf.estimator.ModeKeys.EVAL:
            metrics = self.build_eval_metrics(predict_ids, labels, nwords, params)
            eval_hooks = []
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              eval_metric_ops=metrics,
                                              evaluation_hooks=eval_hooks)

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = self.build_train_op(loss, params)
            train_hooks = []
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op,
                                              training_hooks=train_hooks)

    def decode(self, output, nwords, params):
        raise NotImplementedError()

    def compute_loss(self, logits, labels, nwords, params):
        raise NotImplementedError()

    def build_predictions(self, predict_ids, params):
        tags_idx2str = lookup_ops.index_to_string_table_from_file(params['tag_vocab'], default_value=params['oov_tag'])
        predict_tags = tags_idx2str.lookup(predict_ids)
        predictions = {
            "predict_ids": predict_ids,
            "predict_tags": predict_tags
        }
        return predictions

    def build_eval_metrics(self, predict_ids, labels, nwords, params):
        tags_str2idx = lookup_ops.index_table_from_file(params['tag_vocab'], default_value=0)
        actual_ids = tags_str2idx.lookup(labels)
        weights = tf.sequence_mask(nwords)
        metrics = {
            "accuracy": tf.metrics.accuracy(actual_ids, predict_ids, weights)
        }
        return metrics

    def build_train_op(self, loss, params):
        if params['optimizer'].lower() == 'adam':
            return tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_or_create_global_step())
        if params['optimizer'].lower() == 'momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=params['learning_rate'], momentum=params['momentum'])
            return opt.minimize(loss, global_step=tf.train.get_or_create_global_step())
        if params['optimizer'].lower() == 'adadelta':
            opt = tf.train.AdadeltaOptimizer()
            return opt.minimize(loss, global_step=tf.train.get_or_create_global_step())
        if params['optimizer'].lower() == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=params['learning_rate'])
            return opt.minimize(loss, global_step=tf.train.get_or_create_global_step())

        # TODO(luozhouyang) decay lr
        sgd = tf.train.GradientDescentOptimizer(learning_rate=params.get('learning_rate', 1.0))
        return sgd.minimize(loss, global_step=tf.train.get_or_create_global_step())
