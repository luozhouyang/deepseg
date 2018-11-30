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

import argparse
import functools
import json
import os

import tensorflow as tf

from . import utils
from .bilstm_crf_model import BiLSTMCRFModel
from .hooks import EvalSummaryHook
from .hooks import InitHook
from .hooks import SaveEvaluationPredictionHook
from .hooks import TrainSummaryHook


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

        # strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=0)

        run_config = tf.estimator.RunConfig(
            model_dir=self.params['model_dir'],
            session_config=sess_config,
            tf_random_seed=seed,
            save_checkpoints_secs=None,
            save_checkpoints_steps=params['save_ckpt_steps'],
            keep_checkpoint_max=params['keep_ckpt_max'],
            log_step_count_steps=params['log_step_count_steps'],
            train_distribute=None)

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
        train_spec = tf.estimator.TrainSpec(
            input_fn=input_fn,
            hooks=self._build_train_hooks(),
            max_steps=10000)

        self.estimator.train(
            train_spec.input_fn,
            train_spec.hooks,
            max_steps=train_spec.max_steps)

    def eval(self):
        input_fn = functools.partial(
            self.model.input_fn,
            params=self.params,
            mode=tf.estimator.ModeKeys.EVAL)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=input_fn, hooks=self._build_eval_hooks())
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
        train_spec = tf.estimator.TrainSpec(
            input_fn=train_input_fn,
            max_steps=1000000,
            hooks=self._build_train_hooks())
        eval_input_fn = functools.partial(
            self.model.input_fn,
            params=self.params,
            mode=tf.estimator.ModeKeys.EVAL)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=eval_input_fn,
            hooks=self._build_eval_hooks())
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

    def _build_train_hooks(self):
        return [InitHook(), TrainSummaryHook(self.estimator.model_dir)]

    def _build_eval_hooks(self):
        save_path = os.path.join(self.estimator.model_dir, "eval")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        output_file = os.path.join(save_path, "predictions.txt")
        eval_hooks = [InitHook(),
                      SaveEvaluationPredictionHook(output_file),
                      EvalSummaryHook(self.estimator.model_dir)]
        return eval_hooks


def check_predictions_files(params):
    if not os.path.exists(params['predict_src_file']):
        raise ValueError("predict_src_file must be provided in predict mode!")
    if not os.path.exists(params['predict_tag_file']):
        raise ValueError("predict_tag_file must be provided in predict mode!")


def check_vocab_files(params):
    utils.check_src_vocab_file(params)
    utils.check_tag_vocab_file(params)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--params_file", type=str, default="hparams.json",
                        required=True,
                        help="The params configuration in JSON format.")
    parser.add_argument("--mode", type=str, default="train_and_eval",
                        choices=["train", "eval",
                                 "train_and_eval", "predict", "export"],
                        help="run mode.")

    args, _ = parser.parse_known_args()
    config_file = args.params_file
    with open(config_file, mode="rt", encoding="utf8", buffering=8192) as f:
        params = json.load(f)

    check_vocab_files(params)
    runner = Runner(params=params)

    mode = args.mode
    if mode == "train":
        runner.train()
    elif mode == "eval":
        runner.eval()
    elif mode == "train_and_eval":
        runner.train_and_eval()
    elif mode == "predict":
        check_predictions_files(params)
        runner.predict()
    elif mode == "export":
        runner.export()
    else:
        raise ValueError("Unknown mode: %s" % mode)
