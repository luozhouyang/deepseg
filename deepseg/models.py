import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.text.crf import crf_log_likelihood


def unpack_data(data):
    if len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data
    else:
        raise TypeError("Expected data to be a tuple of size 2 or 3.")


class CRFEndpoint(tf.keras.Model):

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def call(self, inputs):
        return self.model(inputs)

    def compute_loss(self, x, y, sample_weight, training=False):
        y_pred = self(x, training=training)
        _, potentials, sequence_length, chain_kernel = y_pred

        crf_loss = -crf_log_likelihood(potentials, y, sequence_length, chain_kernel)[0]

        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight

        return tf.reduce_mean(crf_loss), sum(self.losses)

    def train_step(self, data):
        x, y, sample_weight = unpack_data(data)
        with tf.GradientTape() as tape:
            crf_loss, internal_losses = self.compute_loss(
                x, y, sample_weight, training=True
            )
            total_loss = crf_loss + internal_losses

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"crf_loss": crf_loss, "internal_losses": internal_losses}

    def test_step(self, data):
        x, y, sample_weight = unpack_data(data)
        crf_loss, internal_losses = self.compute_loss(x, y, sample_weight)
        return {"crf_loss": crf_loss, "internal_losses": internal_losses}
