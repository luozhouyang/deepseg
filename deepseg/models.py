import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.text.crf import crf_log_likelihood

from .crf import CRF


def unpack_data(data):
    if len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data
    else:
        raise TypeError("Expected data to be a tuple of size 2 or 3.")


class CRFEndpoint(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        raise NotImplementedError()

    def _compute_loss(self, x, y, sample_weight, training=False):
        y_pred, potentials, sequence_length, chain_kernel = self(x, training=True)
        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]
        if sample_weight:
            crf_loss = crf_loss * sample_weight
        crf_loss = tf.reduce_mean(crf_loss)
        # loss = self.compiled_loss(
        #     tf.cast(y, dtype=self.dtype),
        #     tf.cast(y_pred, dtype=self.dtype),
        #     sample_weight=sample_weight,
        #     regularization_losses=self.losses)
        # return crf_loss, loss, y_pred
        return crf_loss, sum(self.losses), y_pred

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)
        with tf.GradientTape() as tape:
            crf_loss, loss, y_pred = self._compute_loss(x, y, sample_weight, training=True)
            total_loss = crf_loss + loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)

        res = {"crf_loss": crf_loss, 'interal_loss': loss}
        for m in self.metrics:
            res[m.name] = m.result()
        return res

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        crf_loss, internal_loss, y_pred = self._compute_loss(x, y, sample_weight)
        self.compiled_metrics.update_state(y, y_pred)
        res = {"crf_loss": crf_loss, 'interal_loss': internal_loss}
        for m in self.metrics:
            res[m.name] = m.result()
        return res


class BiLSTMCRFModel(CRFEndpoint):

    def __init__(self, vocab_size, embedding_size, hidden_size, num_tags, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size, name='embedding')
        self.embedding_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.bilstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_size, return_sequences=True), name='bilstm')
        self.bilstm_dropout = tf.keras.layers.Dropout(dropout_rate)
        self.crf = CRF(num_tags)

    def call(self, inputs, training=None):
        seq = inputs
        outputs = self.embedding_dropout(self.embedding(seq), training=training)
        outputs = self.bilstm_dropout(self.bilstm(outputs), training=training)
        outputs = self.crf(outputs)
        return outputs
