import multiprocessing
import os
import re

import tensorflow as tf

from deepseg import vocab
from deepseg import utils

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
                   skip_count=0):
    if not buff_size:
        buff_size = batch_size * 1024
    dataset = tf.data.Dataset.zip((feat_dataset, label_dataset))

    # skip and shuffle
    if not skip_count:
        dataset = dataset.skip(skip_count)
    dataset = dataset.shuffle(buff_size, random_seed, True)

    # split data
    dataset = dataset.map(
        lambda src, tgt: (
            tf.string_split([src], ",").values,
            tf.string_split([tgt], ",").values),
        num_parallel_calls=num_parallel_call).prefetch(buff_size)

    # filter empty data
    dataset = dataset.filter(
        lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0)).prefetch(buff_size)

    if max_len:
        dataset = dataset.map(
            lambda src, tgt: (src[:max_len], tgt[:max_len]),
            num_parallel_calls=num_parallel_call).prefetch(buff_size)

    # cast string to ids
    dataset = dataset.map(lambda src, tgt: (
        tf.cast(feat_vocab_table.lookup(src), tf.int64),
        tf.cast(label_vocab_table.lookup(tgt), tf.int64)),
                          num_parallel_calls=num_parallel_call).prefetch(buff_size)

    def padding_func(x):
        return x.padded_batch(
            batch_size,
            padded_shapes=(tf.TensorShape([None]),
                           tf.TensorShape([None])),
            padding_values=(tf.cast(feat_vocab_table.lookup(tf.constant(vocab.UNK)), tf.int64),
                            tf.cast(label_vocab_table.lookup(tf.constant(vocab.LABEL_S)), tf.int64)))

    # padding
    dataset = padding_func(dataset)
    dataset = dataset.repeat()
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
        output_label_file: A file to save labels
        parallel: The parallel number of multiprocessing
        file_parts: The number of parts the tagged file will be split
    """
    parts = utils.split_file(seg_file, file_parts, "part")
    pool = multiprocessing.Pool(parallel)
    jobs = []
    features_files, labels_files = [], []
    for file in parts:
        prefix = str(file)[0:str(file).rfind(os.sep)]
        name = str(file).split(os.sep)[-1]
        feat_file = os.path.join(prefix, name.replace("part", "feature"))
        label_file = os.path.join(prefix, name.replace("part", "label"))
        jobs.append(pool.apply_async(generate_training_files, (file, feat_file, label_file)))
        features_files.append(feat_file)
        labels_files.append(label_file)

    for job in jobs:
        job.get()

    # TODO(luozhouyang) Fix: if features becomes empty after processing then the tags may mismatch
    # concat each part to s single file
    utils.concat_files(sorted(features_files), output_feature_file)
    utils.concat_files(sorted(labels_files), output_label_file)

    # remove tmp files
    f0 = features_files[0]
    d = str(f0)[0:str(f0).rindex(os.sep)]
    utils.remove_dir(d)


def generate_training_files(seg_file, output_feature_file, output_label_file):
    """Generate features file and labels file from a tagged(segmented) file.

    Args:
        seg_file: A tagged(segmented) file
        output_feature_file: A file to save features
        output_label_file: A file to save labels
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
