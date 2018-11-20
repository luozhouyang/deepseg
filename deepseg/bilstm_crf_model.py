import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from .base_model import BaseModel


class BiLSTMCRFModel(BaseModel):

    def decode(self, output, nwords, params):
        logits = tf.layers.dense(output, params['num_tags'])
        with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
            variable = tf.get_variable(
                "transition",
                shape=[params['num_tags'], params['num_tags']],
                dtype=tf.float32)
            predict_ids, _ = tf.contrib.crf.crf_decode(logits, variable, nwords)
        return logits, predict_ids

    def compute_loss(self, logits, labels, nwords, params):
        tags_str2idx = lookup_ops.index_table_from_file(
            params['tag_vocab'], default_value=0)
        actual_ids = tags_str2idx.lookup(labels)
        with tf.variable_scope("crf", reuse=True):
            trans_val = tf.get_variable(
                "transition",
                shape=[params['num_tags'], params['num_tags']],
                dtype=tf.float32)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=actual_ids,
            sequence_lengths=nwords,
            transition_params=trans_val)
        loss = tf.reduce_mean(-log_likelihood)
        return loss
