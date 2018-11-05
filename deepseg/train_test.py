import tensorflow as tf
from deepseg import train
import os
from deepseg import test_utils


class TrainTest(tf.test.TestCase):

    def _getHparamsFile(self):
        cur = os.path.dirname(__file__)
        return os.path.join(cur, "../deepseg", "hparams.json")

    def testTrain(self):
        hparams = test_utils._create_test_hparams()
        train.train(hparams)


if __name__ == "__main__":
    tf.test.main()
