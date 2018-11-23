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

from .runner import Runner


class RunnerTest(tf.test.TestCase):

    @staticmethod
    def _buildParams():
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
            "max_src_len": 40,
            "random_seed": 1000,
            "skip_count": 0
        }
        return params

    def testTrain(self):
        r = Runner(self._buildParams())
        r.train()

    def testEval(self):
        r = Runner(self._buildParams())
        r.eval()

    def testTrainAndEval(self):
        r = Runner(self._buildParams())
        r.train_and_eval()

    def testPredict(self):
        r = Runner(self._buildParams())
        predictions = r.predict()
        for i, p in enumerate(predictions):
            print(i, p['predict_ids'])
            print(i, p['predict_tags'])

    def testExport(self):
        r = Runner(self._buildParams())
        r.export()


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.WARN)
    tf.test.main()
