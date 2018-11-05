import tensorflow as tf

from deepseg import data
from deepseg import vocab
import os
from deepseg.data import Dataset


def get_test_file(name):
    test_dir = os.path.join(os.path.dirname(__file__), "../testdata")
    return os.path.join(test_dir, name)


def _create_test_hparams():
    hparams = tf.contrib.training.HParams()
    hparams.add_hparam("features_file", get_test_file("feature.txt"))
    hparams.add_hparam("labels_file", get_test_file("label.txt"))
    hparams.add_hparam("features_vocab_file", get_test_file("vocab.feature.txt"))
    hparams.add_hparam("labels_vocab_file", get_test_file("vocab.label.txt"))
    hparams.add_hparam("random_seed", 3)
    hparams.add_hparam("max_len", 20)
    hparams.add_hparam("batch_size", 32)
    hparams.add_hparam("buff_size", 1)
    hparams.add_hparam("skip_count", 0)
    return hparams


class DataTest(tf.test.TestCase):

    def testGenerateTrainingFiles(self):
        seg_file = "testdata/seg_file.txt"
        feat_output = "testdata/feature.txt"
        label_output = "testdata/label.txt"
        data.generate_training_files(seg_file, feat_output, label_output)

    def testBuildTrainingDataset(self):
        feat_vocab = self._getTestFile("vocab.feature.txt")
        label_vocab = self._getTestFile("vocab.label.txt")
        print(feat_vocab, label_vocab)
        feat_vocab_table, _ = vocab.create_features_vocab_table(feat_vocab)
        label_vocab_table, _ = vocab.create_features_vocab_table(label_vocab)
        hparams = _create_test_hparams()

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


class DatasetTest(tf.test.TestCase):

    def testTrainingGenerator(self):
        hparams = _create_test_hparams()
        ds = Dataset(hparams)
        iter = ds.training_generator()
        for i in range(10):
            features, labels = iter.__next__()
            print(features, labels)


if __name__ == "__main__":
    tf.test.main()
