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

from . import collections
from . import utils


class InitHook(tf.train.SessionRunHook):

    def after_create_session(self, session, coord):
        table_init_op = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
        session.run(table_init_op)
        variable_init_op = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        session.run(variable_init_op)


class SaveEvaluationPredictionHook(tf.train.SessionRunHook):

    def __init__(self, output_file):
        self._output_file = output_file
        self._predictions = None
        self._global_steps = None

    def begin(self):
        self._predictions = collections.get_dict_from_collection("predictions")
        if not self._predictions:
            return
        self._global_steps = tf.train.get_global_step()
        if not self._global_steps:
            raise ValueError("You must create global_steps first.")

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self._predictions, self._global_steps])

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        predictions, global_steps = run_values.results
        output_path = "{}.{}".format(self._output_file, global_steps)
        with open(output_path, encoding="utf8", mode="a") as file:
            tags = predictions['predict_tags']
            for t in tags:
                tag_string = utils.convert_prediction_tags_to_string(t)
                file.write(tag_string + "\n")

    def end(self, session):
        tf.logging.info(
            "Evaluation predictions saved to %s" % self._output_file)


class TrainSummaryHook(tf.train.SessionRunHook):

    def __init__(self, output_dir):
        self._output_dir = output_dir
        self._summary_writer = None

    def begin(self):
        self._train_loss = tf.get_collection(collections.TRAIN_LOSS)
        self._global_step = tf.train.get_global_step()
        assert self._global_step is not None
        self._summary_writer = tf.summary.FileWriter(self._output_dir)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self._train_loss, self._global_step])

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        loss, global_step = run_values.results
        loss_summary = tf.Summary(
            value=[tf.Summary.Value(tag=collections.TRAIN_LOSS,
                                    simple_value=loss[0])])
        self._summary_writer.add_summary(loss_summary, global_step)


class EvalSummaryHook(tf.train.SessionRunHook):

    def __init__(self, output_dir):
        self._output_dir = output_dir
        self._summary_writer = None

    def begin(self):
        self._eval_loss = tf.get_collection(collections.EVAL_LOSS)
        self._global_steps = tf.train.get_global_step()
        assert self._global_steps is not None
        self._summary_writer = tf.summary.FileWriter(self._output_dir)

    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self._eval_loss, self._global_steps])

    def after_run(self,
                  run_context,  # pylint: disable=unused-argument
                  run_values):
        loss, global_steps = run_values.results
        loss_summary = tf.Summary(
            value=[tf.Summary.Value(tag=collections.EVAL_LOSS,
                                    simple_value=loss[0])])
        self._summary_writer.add_summary(loss_summary, global_steps)
