import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from . import dataset_util
import os


class DatasetUtilTest(tf.test.TestCase):

    @staticmethod
    def getParams():
        testdata_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../testdata"))
        params = {
            "model_dir": os.path.join(testdata_dir, "model"),
            "train_src_file": os.path.join(testdata_dir, "feature.txt"),
            "train_tag_file": os.path.join(testdata_dir, "label.txt"),
            "eval_src_file": os.path.join(testdata_dir, "feature_eval.txt"),
            "eval_tag_file": os.path.join(testdata_dir, "label_eval.txt"),
            "predict_src_file": os.path.join(testdata_dir, "feature.txt"),
            "src_vocab": os.path.join(testdata_dir, "vocab.feature.txt"),
            "tag_vocab": os.path.join(testdata_dir, "vocab.label.txt"),
            "pad": "<PAD>",
            "oov_tag": "O",
            "shuffle": True,
            "buff_size": 1000,
            "reshuffle_each_iteration": True,
            "repeat": 5,
            "batch_size": 4,
            "vocab_size": 17,
            "embedding_size": 256,
            "dropout": 0.5,
            "lstm_size": 256,
            "optimizer": "adam",
            "save_ckpt_steps": 100,
            "keep_ckpt_max": 5,
            "log_step_count_steps": 10,
            "num_tags": 5,
            "num_parallel_call": 4,
            "skip_count": 0,
            "max_src_len": 40,
            "random_seed": 1000
        }
        return params

    def testBuildTrainDataset(self):
        dataset = dataset_util.build_train_dataset(self.getParams())
        iterator = dataset.make_initializable_iterator()

        with self.test_session() as sess:
            sess.run(iterator.initializer)
            try:
                while True:
                    (src, src_len) = iterator.get_next()
                    src = sess.run(src)
                    src_len = sess.run(src_len)
                    print(src, src_len)
            except OutOfRangeError as e:
                return

    def testBuildEvalDataset(self):
        dataset = dataset_util.build_eval_dataset(self.getParams())
        iterator = dataset.make_initializable_iterator()

        with self.test_session() as sess:
            sess.run(iterator.initializer)
            try:
                while True:
                    (src, src_len) = iterator.get_next()
                    src = sess.run(src)
                    src_len = sess.run(src_len)
                    print(src, src_len)
            except OutOfRangeError as e:
                return

    def testBuildPredictDataset(self):
        dataset = dataset_util.build_predict_dataset(self.getParams())
        iterator = dataset.make_initializable_iterator()

        with self.test_session() as sess:
            sess.run(iterator.initializer)
            try:
                while True:
                    (src, src_len) = iterator.get_next()
                    src = sess.run(src)
                    src_len = sess.run(src_len)
                    print(src, src_len)
            except OutOfRangeError as e:
                return


if __name__ == "__main__":
    tf.test.main()
