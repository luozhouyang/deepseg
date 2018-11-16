import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from .base_model import BaseModel


class BiLSTMCRFModel(BaseModel):

    def decode(self, output, nwords, params):
        with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(output, params['num_tags'])
            variable = tf.get_variable("transition", shape=[params['num_tags'], params['num_tags']])
            predict_ids, _ = tf.contrib.crf.crf_decode(logits, variable, nwords)
        return logits, predict_ids

    def compute_loss(self, logits, labels, nwords, params):
        tags_str2idx = lookup_ops.index_table_from_file(params['tags_vocab'])
        actual_ids = tags_str2idx.lookup(labels)
        with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
            trans_val = tf.get_variable("transition", shape=[params['num_tags'], params['num_tags']])
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=actual_ids,
            sequence_length=nwords,
            transition_params=trans_val)
        loss = tf.reduce_mean(-log_likelihood)
        return loss
