import tensorflow as tf
from .bilstm_crf_model import BiLSTMCRFModel
import functools


class Runner(object):

    def __init__(self, params):
        self.params = params
        self.model = BiLSTMCRFModel()

        sess_config = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=True,
            gpu_options=tf.GPUOptions(allow_growth=True))

        seed = None

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
        for i in range(10):
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
        def receive_fn():
            features = {
                "inputs": tf.placeholder(dtype=tf.string, shape=(None, None)),
                "inputs_length": tf.placeholder(dtype=tf.int64, shape=(None,))
            }

        self.estimator.export_saved_model()
