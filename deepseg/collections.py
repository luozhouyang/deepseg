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

import six
import tensorflow as tf

GRADS = "grads"
TRAIN_LOSS = "train_loss"
EVAL_LOSS = "eval_loss"


def get_dict_from_collection(name):
    key = tf.get_collection(name + "_key")
    value = tf.get_collection(name + "_value")
    return dict(zip(key, value))


def add_dict_to_collection(name, dict_):
    for k, v in six.iteritems(dict_):
        tf.add_to_collection(name + "_key", k)
        tf.add_to_collection(name + "_value", v)
