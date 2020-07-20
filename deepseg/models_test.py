import tensorflow as tf

from .models import BiLSTMCRFModel


class ModelsTest(tf.test.TestCase):

    def testBiLSTMCRFModel(self):
        model = BiLSTMCRFModel(100, 100, 64, 4)
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.CategoricalAccuracy(name='acc'),
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.AUC(num_thresholds=10, name='auc')])

        x = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], shape=(2, 5), dtype=tf.int32)
        y = tf.constant([0, 0, 1, 2, 0, 0, 0, 1, 2, 0], shape=(2, 5), dtype=tf.int32)

        model.fit(x=x, y=y, epochs=10)

        model.evaluate(x=x, y=y)

        y_, _, _, _ = model.predict(x=x)

        print(y_)


if __name__ == "__main__":
    tf.test.main()
