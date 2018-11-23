# Copyright 2018 luozhouyang.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os

import tensorflow as tf
from tensorflow.python.framework.errors_impl import OutOfRangeError

from . import dataset_util


class DatasetUtilTest(tf.test.TestCase):

    @staticmethod
    def getParams():
        testdata_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../testdata"))
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
            "num_tags": 5,
            "num_parallel_call": 4,
            "skip_count": 0,
            "max_src_len": 40,
            "random_seed": 1000
        }
        return params

    def testBuildTrainDataset(self):
        results = dataset_util.build_train_dataset(self.getParams())
        with self.test_session() as sess:
            for _ in range(100):
                try:
                    features, tag = sess.run(results)
                    print(features['inputs'], features['inputs_length'], tag)
                except OutOfRangeError as e:
                    break

    def testBuildEvalDataset(self):
        results = dataset_util.build_eval_dataset(self.getParams())
        with self.test_session() as sess:
            for _ in range(100):
                try:
                    features, tag = sess.run(results)
                    print(features['inputs'], features['inputs_length'], tag)
                except OutOfRangeError as e:
                    break

    def testBuildPredictDataset(self):
        results = dataset_util.build_eval_dataset(self.getParams())
        with self.test_session() as sess:
            for _ in range(100):
                try:
                    features, _ = sess.run(results)
                    print(features['inputs'], features['inputs_length'])
                except OutOfRangeError as e:
                    break


if __name__ == "__main__":
    tf.test.main()
