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

import functools
import os

import tensorflow as tf

from .bilstm_crf_model import BiLSTMCRFModel


# TODO(luozhouyang) Add hooks
class Runner(object):
    """Train, evaluate, predict or export the model."""

    def __init__(self, params):
        self.params = params
        self.model = BiLSTMCRFModel()

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False,
            gpu_options=tf.GPUOptions(allow_growth=True))

        seed = self.params['random_seed']

        run_config = tf.estimator.RunConfig(
            model_dir=self.params['model_dir'],
            session_config=sess_config,
            tf_random_seed=seed,
            save_checkpoints_secs=None,
            save_checkpoints_steps=params['save_ckpt_steps'],
            keep_checkpoint_max=params['keep_ckpt_max'],
            log_step_count_steps=params['log_step_count_steps'])

        self.estimator = tf.estimator.Estimator(
            self.model.model_fn,
            model_dir=params['model_dir'],
            params=params,
            config=run_config)

    def train(self):
        input_fn = functools.partial(
            self.model.input_fn,
            params=self.params,
            mode=tf.estimator.ModeKeys.TRAIN)
        train_hooks = []
        train_spec = tf.estimator.TrainSpec(
            input_fn=input_fn,
            hooks=train_hooks,
            max_steps=1000)

        self.estimator.train(
            train_spec.input_fn,
            train_spec.hooks,
            max_steps=train_spec.max_steps)

    def eval(self):
        input_fn = functools.partial(
            self.model.input_fn,
            params=self.params,
            mode=tf.estimator.ModeKeys.EVAL)
        eval_hooks = []
        eval_spec = tf.estimator.EvalSpec(
            input_fn=input_fn, hooks=eval_hooks)
        self.estimator.evaluate(
            eval_spec.input_fn, hooks=eval_spec.hooks)

    def predict(self):
        """Predict from a file specified in params['predict_src_file']."""
        input_fn = functools.partial(
            self.model.input_fn,
            params=self.params,
            mode=tf.estimator.ModeKeys.PREDICT)
        predict_hooks = []
        predictions = self.estimator.predict(
            input_fn=input_fn, hooks=predict_hooks)
        return predictions

    def train_and_eval(self):
        train_input_fn = functools.partial(
            self.model.input_fn,
            params=self.params,
            mode=tf.estimator.ModeKeys.EVAL)
        train_hooks = []
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, hooks=train_hooks)
        eval_input_fn = functools.partial(
            self.model.input_fn,
            params=self.params,
            mode=tf.estimator.ModeKeys.EVAL)
        eval_hooks = []
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            hooks=eval_hooks)
        tf.estimator.train_and_evaluate(
            self.estimator,
            train_spec=train_spec,
            eval_spec=eval_spec)

    def export(self):
        """Export model as a saved model."""

        def receiver_fn():
            receiver_tensors = {
                "inputs": tf.placeholder(dtype=tf.string, shape=(None, None)),
                "inputs_length": tf.placeholder(dtype=tf.int32, shape=(None))
            }

            features = receiver_tensors.copy()
            return tf.estimator.export.ServingInputReceiver(
                features=features,
                receiver_tensors=receiver_tensors)

        self.estimator.export_savedmodel(
            export_dir_base=os.path.join(self.params['model_dir'], "export"),
            serving_input_receiver_fn=receiver_fn)
