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
"""Build dataset for training, evaluation and predict."""

import os

import tensorflow as tf

__all__ = ["build_dataset",
           "build_train_dataset",
           "build_eval_dataset",
           "build_predict_dataset"]


def build_dataset(params, mode):
    """Build data for input_fn.

    Args:
        params: A dict, storing hparams
        mode: A string, one of `tf.estimator.ModeKeys`

    Returns:
        A tuple of (features, label), feed to model_fn.

    Raises:
        ValueError: If mode is not one of `tf.estimator.ModeKeys'.
    """
    if mode == tf.estimator.ModeKeys.TRAIN:
        return build_train_dataset(params)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return build_eval_dataset(params)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return build_predict_dataset(params)
    else:
        raise ValueError("Invalid mode %s" % mode)


def _build_dataset(src_dataset, tag_dataset, params):
    """Build dataset for training and evaluation mode.

    Args:
        src_dataset: A `tf.data.Dataset` object
        tag_dataset: A `tf.data.Dataset` object
        params: A dict, storing hyper params

    Returns:
        A `tf.data.Dataset` object, producing features and labels.
    """
    dataset = tf.data.Dataset.zip((src_dataset, tag_dataset))
    if params['skip_count'] > 0:
        dataset = dataset.skip(params['skip_count'])
    if params['shuffle']:
        dataset = dataset.shuffle(
            buffer_size=params['buff_size'],
            seed=params['random_seed'],
            reshuffle_each_iteration=params['reshuffle_each_iteration'])
    if params['repeat']:
        dataset = dataset.repeat(params['repeat']).prefetch(params['buff_size'])

    dataset = dataset.map(
        lambda src, tag: (
            tf.string_split([src], delimiter=",").values,
            tf.string_split([tag], delimiter=",").values),
        num_parallel_calls=params['num_parallel_call']
    ).prefetch(params['buff_size'])

    dataset = dataset.filter(
        lambda src, tag: tf.logical_and(tf.size(src) > 0, tf.size(tag) > 0))
    dataset = dataset.filter(
        lambda src, tag: tf.equal(tf.size(src), tf.size(tag)))

    if params['max_src_len']:
        dataset = dataset.map(
            lambda src, tag: (src[:params['max_src_len']],
                              tag[:params['max_src_len']]),
            num_parallel_calls=params['num_parallel_call']
        ).prefetch(params['buff_size'])

    dataset = dataset.map(
        lambda src, tag: (src, tf.size(src), tag),
        num_parallel_calls=params['num_parallel_call']
    ).prefetch(params['buff_size'])

    dataset = dataset.padded_batch(
        batch_size=params.get('batch_size', 32),
        padded_shapes=(
            tf.TensorShape([None]),
            tf.TensorShape([]),
            tf.TensorShape([None])),
        padding_values=(
            tf.constant(params['pad'], dtype=tf.string),
            0,
            tf.constant(params['oov_tag'], dtype=tf.string)))

    dataset = dataset.map(
        lambda src, src_len, tag: ((src, src_len), tag),
        num_parallel_calls=params['num_parallel_call']
    ).prefetch(params['buff_size'])

    return dataset


def build_train_dataset(params):
    """Build data for input_fn in training mode.

    Args:
        params: A dict

    Returns:
        A tuple of (features,labels).
    """
    src_file = params['train_src_file']
    tag_file = params['train_tag_file']

    if not os.path.exists(src_file) or not os.path.exists(tag_file):
        raise ValueError("train_src_file and train_tag_file must be provided")

    src_dataset = tf.data.TextLineDataset(src_file)
    tag_dataset = tf.data.TextLineDataset(tag_file)

    dataset = _build_dataset(src_dataset, tag_dataset, params)

    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    (src, src_len), tag = iterator.get_next()
    features = {
        "inputs": src,
        "inputs_length": src_len
    }

    return features, tag


def build_eval_dataset(params):
    """Build data for input_fn in evaluation mode.

    Args:
        params: A dict.

    Returns:
        A tuple of (features, labels).
    """
    src_file = params['eval_src_file']
    tag_file = params['eval_tag_file']

    if not os.path.exists(src_file) or not os.path.exists(tag_file):
        raise ValueError("eval_src_file and eval_tag_file must be provided")

    src_dataset = tf.data.TextLineDataset(src_file)
    tag_dataset = tf.data.TextLineDataset(tag_file)

    dataset = _build_dataset(src_dataset, tag_dataset, params)
    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    (src, src_len), tag = iterator.get_next()
    features = {
        "inputs": src,
        "inputs_length": src_len
    }

    return features, tag


def build_predict_dataset(params):
    """Build data for input_fn in predict mode.

    Args:
        params: A dict.

    Returns:
        A tuple of (features, labels), where labels are None.
    """
    src_file = params['predict_src_file']
    if not os.path.exists(src_file):
        raise FileNotFoundError("File not found: %s" % src_file)
    dataset = tf.data.TextLineDataset(src_file)
    if params['skip_count'] > 0:
        dataset = dataset.skip(params['skip_count'])

    dataset = dataset.map(
        lambda src: tf.string_split([src], delimiter=",").values,
        num_parallel_calls=params['num_parallel_call']
    ).prefetch(params['buff_size'])

    dataset = dataset.map(
        lambda src: (src, tf.size(src)),
        num_parallel_calls=params['num_parallel_call']
    ).prefetch(params['buff_size'])

    dataset = dataset.padded_batch(
        params.get('batch_size', 32),
        padded_shapes=(
            tf.TensorShape([None]),
            tf.TensorShape([])),
        padding_values=(
            tf.constant(params['pad'], dtype=tf.string),
            0))

    iterator = dataset.make_initializable_iterator()
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, iterator.initializer)
    (src, src_len) = iterator.get_next()
    features = {
        "inputs": src,
        "inputs_length": src_len
    }

    return features, None
