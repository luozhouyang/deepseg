import tensorflow as tf

from . import data


class DataTest(tf.test.TestCase):

    def testGenerateTrainingFiles(self):
        seg_file = "testdata/seg_file.txt"
        feat_output = "testdata/feature.txt"
        label_output = "testdata/label.txt"
        data.generate_training_files(seg_file, feat_output, label_output)


if __name__ == "__main__":
    tf.test.main()
