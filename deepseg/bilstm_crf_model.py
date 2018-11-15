import functools

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from .abstract_model import AbstractModel


class BiLSTMCRFModel(AbstractModel):

    def input_fn(self, params):

        def generator_fn(src, tag):
            with open(src, mode="r", encoding="utf8") as fsrc, open(tag, mode="r", encoding="utf8") as ftag:
                for src_line, tag_line in zip(fsrc, ftag):
                    yield parse_fn(src_line, tag_line)

        def parse_fn(src_line, tag_line):
            words = [w.encode() for w in src_line.strip("\n").strip().split()]
            tags = [t.encode() for t in tag_line.strip("\n").strip().split()]
            assert len(words) == len(tags)
            return ((words, len(words)), tags)

        src_file = params['src_file']
        tag_file = params['tag_file']
        if not src_file or not tag_file:
            raise ValueError("src file and tag file must be provided.")

        padded_shapes = (([None], ()), [None])
        padding_values = ((params['pad'], 0), params['oov_tag'])
        data_types = ((tf.string, tf.int32), tf.string)

        dataset = tf.data.Dataset.from_generator(
            functools.partial(generator_fn, src_file, tag_file),
            output_shapes=padded_shapes,
            output_types=data_types, )
        if params['shuffle']:
            dataset = dataset.shuffle(buffer_size=params['buff_size'],
                                      reshuffle_each_iteration=params['reshuffle_each_iteration'])
        if params['repeat']:
            dataset = dataset.repeat(params['repeat'])

        dataset = dataset.padded_batch(batch_size=params.get('batch_size', 32),
                                       padded_shapes=padded_shapes,
                                       padding_values=padding_values).prefetch(params['buff_size'])
        return dataset

    def model_fn(self, train_hooks=None, eval_hooks=None):

        def _model_fn(features, labels, mode, params, config=None):
            words, nwords = features
            # a UNK token should placed in the first row in vocab file
            words_str2idx = lookup_ops.index_table_from_file(params['src_vocab'], num_oov_buckets=0)
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
                (output_fw, output_bw), (state_fw, state_bw) = tf.contrib.rnn.bidirectional_dynamic_rnn(
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

            if mode == tf.estimator.ModeKeys.PREDICT:
                predictions = self.build_predictions(predict_ids, params)
                return tf.estimator.EstimatorSpec(mode, predictions=predictions)

            loss = self.compute_loss(logits, labels, nwords, params)

            if mode == tf.estimator.ModeKeys.EVAL:
                metrics = self.build_eval_metrics(predict_ids, labels, nwords, params)
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = self.build_train_op(loss, params)
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

        return _model_fn

    def decode(self, output, nwords, params):
        with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(output, params['num_tags'])
            variable = tf.get_variable("crf", shape=[params['num_tags'], params['num_tags']])
            predict_ids, _ = tf.contrib.crf.crf_decode(logits, variable, nwords)
        return logits, predict_ids

    def compute_loss(self, logits, labels, nwords, params):
        tags_str2idx = lookup_ops.index_table_from_file(params['tags_vocab'])
        actual_ids = tags_str2idx.lookup(labels)
        trans_val = tf.get_variable("crf/crf", shape=[params['num_tags'], params['num_tags']])
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=actual_ids,
            sequence_length=nwords,
            transition_params=trans_val)
        loss = tf.reduce_mean(-log_likelihood)
        return loss

    def build_eval_metrics(self, predict_ids, labels, nwords, params):
        tags_str2idx = lookup_ops.index_table_from_file(params['tags_vocab'])
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

    def build_predictions(self, predict_ids, params):
        tags_idx2str = lookup_ops.index_to_string_table_from_file(params['tags_vocab'])
        predict_tags = tags_idx2str.lookup(predict_ids)
        predictions = {
            "predict_ids": predict_ids,
            "predict_tags": predict_tags
        }
        return predictions
