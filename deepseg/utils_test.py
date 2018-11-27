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

from . import utils


class UtilsTest(tf.test.TestCase):

    def getParams(self):
        testdata_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "../testdata"))
        return {
            "src_vocab": os.path.join(testdata_dir, "vocab.feature.txt"),
            "tag_vocab": os.path.join(testdata_dir, "vocab.label.txt"),
            "unk": "<UNK>",
            "pad": "<PAD>",
            "oov_tag": "O"
        }

    def testCheckSrcVocab(self):
        params = self.getParams()
        utils.check_src_vocab_file(params)


if __name__ == "__main__":
    tf.test.main()
