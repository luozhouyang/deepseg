import tensorflow as tf

from deepseg import data
from deepseg import vocab
import os


class DataTest(tf.test.TestCase):

    def _getTestFile(self, name):
        test_dir = os.path.join(os.path.dirname(__file__), "../testdata")
        return os.path.join(test_dir, name)

    def _createTestHParams(self):
        hparams = tf.contrib.training.HParams()
        hparams.add_hparam("features_file", self._getTestFile("feature.txt"))
        hparams.add_hparam("labels_file", self._getTestFile("label.txt"))
        hparams.add_hparam("random_seed", 3)
        hparams.add_hparam("max_len", 40)
        hparams.add_hparam("batch_size", 3)
        hparams.add_hparam("buff_size", 1)
        hparams.add_hparam("skip_count", 0)
        return hparams

    def testGenerateTrainingFiles(self):
        seg_file = "testdata/seg_file.txt"
        feat_output = "testdata/feature.txt"
        label_output = "testdata/label.txt"
        data.generate_training_files(seg_file, feat_output, label_output)

    def testBuildTrainingDataset(self):
        feat_vocab = self._getTestFile("vocab.feature.txt")
        label_vocab = self._getTestFile("vocab.label.txt")
        print(feat_vocab, label_vocab)
        feat_vocab_table, _ = vocab.create_vocab_table(feat_vocab)
        label_vocab_table, _ = vocab.create_vocab_table(label_vocab)
        hparams = self._createTestHParams()

        iterator = data.build_train_dataset(feat_vocab_table, label_vocab_table, hparams)

        with self.test_session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)
            for _ in range(3):
                x, y = iterator.get_next()
                x = sess.run(x)
                y = sess.run(y)
                print(x)
                print(y)


if __name__ == "__main__":
    tf.test.main()
