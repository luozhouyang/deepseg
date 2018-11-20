import tensorflow as tf
import os

__all__ = ["build_dataset",
           "build_train_dataset",
           "build_eval_dataset",
           "build_predict_dataset"]


def build_dataset(params, mode):
    if mode == tf.estimator.ModeKeys.TRAIN:
        return build_train_dataset(params)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return build_eval_dataset(params)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return build_predict_dataset(params)
    else:
        raise ValueError("Invalid mode %s" % mode)


def _build_dataset(src_dataset, tag_dataset, params):
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
    src_file = params['train_src_file']
    tag_file = params['train_tag_file']

    if not os.path.exists(src_file) or not os.path.exists(tag_file):
        raise ValueError("train_src_file and train_tag_file must be provided")

    src_dataset = tf.data.TextLineDataset(src_file)
    tag_dataset = tf.data.TextLineDataset(tag_file)

    return _build_dataset(src_dataset, tag_dataset, params)


def build_eval_dataset(params):
    src_file = params['eval_src_file']
    tag_file = params['eval_tag_file']

    if not os.path.exists(src_file) or not os.path.exists(tag_file):
        raise ValueError("eval_src_file and eval_tag_file must be provided")

    src_dataset = tf.data.TextLineDataset(src_file)
    tag_dataset = tf.data.TextLineDataset(tag_file)

    return _build_dataset(src_dataset, tag_dataset, params)


def build_predict_dataset(params):
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
    return dataset
