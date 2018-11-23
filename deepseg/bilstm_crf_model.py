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

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from .base_model import BaseModel


class BiLSTMCRFModel(BaseModel):
    """BiLSTM + CRF model."""

    def decode(self, output, nwords, params):
        """Decode outputs of BiLSTM.

        Args:
            output: A tensor, output of BiLSTM
            nwords: A tensor, length of inputs
            params: A dict, storing hyper params

        Returns:
            Logits tensor and predict ids tensor.
        """
        logits = tf.layers.dense(output, params['num_tags'])
        with tf.variable_scope("crf", reuse=tf.AUTO_REUSE):
            variable = tf.get_variable(
                "transition",
                shape=[params['num_tags'], params['num_tags']],
                dtype=tf.float32)
            predict_ids, _ = tf.contrib.crf.crf_decode(logits, variable, nwords)
        return logits, predict_ids

    def compute_loss(self, logits, labels, nwords, params):
        """Compute loss.

        Args:
            logits: A tensor, output of dense layer
            labels: A tensor, the ground truth label
            nwords: A tensor, length of inputs
            params: A dict, storing hyper params

        Returns:
            A loss tensor, negative log likelihood loss.
        """
        tags_str2idx = lookup_ops.index_table_from_file(
            params['tag_vocab'], default_value=0)
        actual_ids = tags_str2idx.lookup(labels)
        # get transition matrix created before
        with tf.variable_scope("crf", reuse=True):
            trans_val = tf.get_variable(
                "transition",
                shape=[params['num_tags'], params['num_tags']],
                dtype=tf.float32)
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            inputs=logits,
            tag_indices=actual_ids,
            sequence_lengths=nwords,
            transition_params=trans_val)
        loss = tf.reduce_mean(-log_likelihood)
        return loss
