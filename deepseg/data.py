import tensorflow as tf
import os
import re
from . import utils
import multiprocessing

import keras

__all__ = ["build_train_dataset", "build_validation_dataset",
           "generate_training_files", "generate_training_files_parallel"]


def build_train_dataset(feat_vocab_table, label_vocab_table, hparams):
    """Generate iterator of training dataset.

        Args:
            feat_vocab_table: features vocab lookup table
            label_vocab_table: labels vocab lookup table
            hparams: hparams

        Returns:
            iterator of training dataset.
        """
    feat_dataset = tf.data.TextLineDataset(hparams.features_file)
    label_dataset = tf.data.TextLineDataset(hparams.labels_file)

    dataset = _build_dataset(feat_vocab_table,
                             label_vocab_table,
                             feat_dataset,
                             label_dataset,
                             hparams.max_len,
                             hparams.batch_size,
                             hparams.random_seed,
                             multiprocessing.cpu_count(),
                             hparams.buff_size,
                             hparams.skip_count)
    iterator = dataset.make_initializable_iterator()
    return iterator


def build_validation_dataset(feat_vocab_table, label_vocab_table, hparams):
    """Generate iterator of validation dataset.

    Args:
        feat_vocab_table: features vocab lookup table
        label_vocab_table: labels vocab lookup table
        hparams: hparams

    Returns:
        iterator of validation dataset.
    """
    feat_dataset = tf.data.TextLineDataset(hparams.validation_features_file)
    label_dataset = tf.data.TextLineDataset(hparams.validation_labels_file)
    dataset = _build_dataset(feat_vocab_table,
                             label_vocab_table,
                             feat_dataset,
                             label_dataset,
                             hparams.max_len,
                             hparams.batch_size,
                             hparams.random_seed,
                             multiprocessing.cpu_count(),
                             hparams.buff_size,
                             hparams.skip_count)

    iterator = dataset.make_initializable_iterator()
    return iterator


def _build_dataset(feat_vocab_table,
                   label_vocab_table,
                   feat_dataset,
                   label_dataset,
                   max_len,
                   batch_size,
                   random_seed,
                   num_parallel_call=multiprocessing.cpu_count(),
                   buff_size=None,
                   skip_count=None):
    if not buff_size:
        buff_size = batch_size * 1024
    dataset = tf.data.Dataset.zip((feat_dataset, label_dataset))

    # skip and shuffle
    if not skip_count:
        dataset = dataset.skip(skip_count)
    dataset = dataset.shuffle(buff_size, random_seed, True)

    # split data
    dataset = dataset.map(lambda src, tgt: (
        tf.string_split([src], ",").values, tf.string_split([tgt], ",").values),
                          num_parallel_calls=num_parallel_call).prefetch(buff_size)

    # cast string to ids
    dataset = dataset.map(lambda src, tgt: (
        tf.cast(feat_vocab_table.lookup(src), tf.int32),
        tf.cast(label_vocab_table.lookup(tgt), tf.int32)),
                          num_parallel_calls=num_parallel_call).prefetch(buff_size)

    # padding
    dataset = dataset.map(lambda src, tgt: (
        keras.preprocessing.sequence.pad_sequences(
            [src], maxlen=max_len, padding="post", truncating="post", value=0)[0],
        keras.preprocessing.sequence.pad_sequences(
            [tgt], maxlen=max_len, padding="post", truncating="post", value=0)[0]),
                          num_parallel_calls=num_parallel_call).prefetch(buff_size)

    return dataset


def generate_training_files_parallel(seg_file,
                                     output_feature_file,
                                     output_label_file,
                                     parallel=multiprocessing.cpu_count(),
                                     file_parts=128):
    """Generate features file and labels file from a tagged(segmented) file in parallel.

    Args:
        seg_file: A tagged(segmented) file.
        output_feature_file: A file to save features
        output_lable_file: A file to save labels
        parallel: The parallel number of multiprocessing
        file_parts: The number of parts the tagged file will be split
    """
    pass


def generate_training_files(seg_file, output_feature_file, output_label_file):
    """Generate features file and labels file from a tagged(segmented) file.

    Args:
        seg_file: A tagged(segmented) file
        output_feature_file: A file to save features
        output_label_file: A file to save lables
    """
    utils.check_file_exists(seg_file)

    with open(output_feature_file, mode="wt", encoding="utf8", buffering=8192) as feat:
        with open(output_label_file, mode="wt", encoding="utf8", buffering=8192) as label:
            with open(seg_file, mode="rt", encoding="utf8", buffering=8192) as r:
                for line in r:
                    if not line:
                        continue
                    line = line.strip("\n").strip()
                    if not line:
                        continue
                    feat_line, label_line = _generate_feature_and_label(line)
                    if not (feat_line and label_line):
                        continue
                    feat.write(feat_line + "\n")
                    label.write(label_line + "\n")


def _generate_feature_and_label(line):
    """Generate feature and label by line content from tagged file.

    Labels including: S[Single], B[Begin], E[End], M[middle]

    Args:
        line: Tagged file's line

    Returns:
        A string line of feature and a string line of label.
    """
    feat_line, label_line = "", ""
    words = line.split(" ")
    if len(words) <= 0:
        return feat_line, label_line
    for i in range(len(words)):
        word = words[i]
        if len(word) == 1:
            feat_line += (word + ",")
            label_line += "S,"
            continue
        if not utils.is_chinese_word(word):
            feat_line += (word + ",")
            label_line += "S,"
            continue
        for j in range(len(word)):
            char = word[j]
            if j == 0:
                feat_line += (char + ",")
                label_line += "B,"
                continue
            if j == len(word) - 1:
                feat_line += (char + ",")
                label_line += "E,"
                continue
            feat_line += (char + ",")
            label_line += "M,"

    feat_line = re.sub("[,]+", ",", feat_line)
    label_line = re.sub("[,]+", ",", label_line)
    return feat_line.strip(","), label_line.strip(",")
