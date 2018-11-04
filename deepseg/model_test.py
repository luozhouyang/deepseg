import tensorflow as tf
from deepseg import model
from keras.utils import plot_model


class ModelTest(tf.test.TestCase):

    def testBuildModel(self):
        m = model.build_model()
        m.summary()
        plot_model(m, "model.png")


if __name__ == "__main__":
    tf.test.main()
