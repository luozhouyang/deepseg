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

import abc

import tensorflow as tf


class AbstractModel(abc.ABC):
    """Model interface."""

    def input_fn(self, params, mode=tf.estimator.ModeKeys.TRAIN):
        """Input fn for estimator.

        Args:
            params: A dict, storing hyper params
            mode: A string, one of `tf.estimator.ModeKeys`

        Returns:
            A tuple of (features, labels)
        """
        raise NotImplementedError()

    def model_fn(self, features, labels, mode, params, config):
        """Build model fn for estimator.

        Args:
            features: Features data
            labels: Labels data
            mode: Mode
            params: Hyper params
            config: Run config
        """
        raise NotImplementedError()

    def decode(self, output, nwords, params):
        """Decode the outputs of BiLSTM.

        Args:
            output: A tensor, output of BiLSTM
            nwords: A scalar, the length of the source sequence
            params: A dict, storing hyper params

        Returns:
            Logits and predict ids.
        """
        raise NotImplementedError()

    def compute_loss(self, logits, labels, nwords, params):
        """Compute loss.

        Args:
            logits: A tensor, logits from decode
            labels: A tensor, the ground truth label
            nwords: A scalar, the length of the source sequence
            params: A dict, storing hyper params

        Returns:
            The loss of this network.
        """
        raise NotImplementedError()

    def build_predictions(self, predict_ids, params):
        """Build predictions for predict mode.

        Args:
            predict_ids: A tensor, predict ids from decode.
            params: A dict, storing hyper params

        Returns:
            A dict, storing predictions you want
        """
        raise NotImplementedError()

    def build_eval_metrics(self, predict_ids, labels, nwords, params):
        """Build evaluation ops for eval mode.

        Args:
            predict_ids: A tensor, predict ids from decode
            labels: A tensor, the ground truth labels
            nwords: A scalar, the length of the source sequence
            params: A dict, storing hyper params

        Returns:
            A metrics op for eval mode.
        """
        raise NotImplementedError()

    def build_train_op(self, loss, params):
        """Build train op for train mode.

        Args:
            loss: A tensor, loss of this network
            params: A dict, storing hyper params

        Returns:
            A train op for train mode.
        """
        raise NotImplementedError()
