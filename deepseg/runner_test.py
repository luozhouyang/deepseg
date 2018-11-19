import os

import tensorflow as tf

from .runner import Runner


class RunnerTest(tf.test.TestCase):

    def _buildParams(self):
        testdata_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../testdata"))
        params = {
            "model_dir": os.path.join(testdata_dir, "model"),
            "train_src_file": os.path.join(testdata_dir, "feature.txt"),
            "train_tag_file": os.path.join(testdata_dir, "label.txt"),
            "eval_src_file": os.path.join(testdata_dir, "feature.txt"),
            "eval_tag_file": os.path.join(testdata_dir, "label.txt"),
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
            "num_tags": 5
        }
        return params

    def testTrain(self):
        r = Runner(self._buildParams())
        r.train()


if __name__ == "__main__":
    tf.test.main()
