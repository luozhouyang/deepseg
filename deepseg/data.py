import multiprocessing
import os
import re

import numpy as np
import tensorflow as tf

from deepseg import utils
from deepseg import vocab

__all__ = ["build_train_dataset", "build_validation_dataset",
           "generate_training_files", "generate_training_files_parallel", "Dataset"]


class Dataset(object):

    def __init__(self,
                 hparams,
                 features_vocab_limit=None,
                 labels_vocab_limit=None):
        self.hparams = hparams

        self._sequence_length = hparams.max_len
        self._batch_size = hparams.batch_size

        self._features_file = hparams.features_file
        self._labels_file = hparams.labels_file
        self._features_vocab_file = hparams.features_vocab_file
        self._features_vocab_limit = features_vocab_limit
        if not self._features_vocab_file:
            self._generate_features_vocab_file()
        self._labels_vocab_file = hparams.labels_vocab_file
        self.labels_vocab_limit = labels_vocab_limit
        if not self._labels_vocab_file:
            self._generate_labels_vocab_file()

        self._features_vocab_size = self._count_file_lines(self.features_vocab_file)
        self._labels_vocab_size = self._count_file_lines(self.labels_vocab_file)

        self._features_string2ids_table, self._features_ids2string_table = self._create_table(self._features_vocab_file)
        self._labels_string2ids_table, self._labels_ids2string_table = self._create_table(self._labels_vocab_file)

        self._features_vocab = vocab.load_vocab(self._features_vocab_file)
        self._labels_vocab = vocab.load_vocab(self._labels_vocab_file)

        self._features_total_number = utils.count_file_lines(self._features_file)

        f = open(self._features_file, encoding="utf8")
        self._features = f.readlines()
        f.close()

        l = open(self._labels_file, encoding="utf8")
        self._labels = l.readlines()
        l.close()

    def training_generator(self):
        batch_features = np.zeros(shape=(self._batch_size, self._sequence_length))
        batch_labels = np.zeros(shape=(self._batch_size, self._sequence_length))
        idx = 0
        while True:
            for i in range(self._batch_size):
                if idx >= self._features_total_number:
                    idx = 0
                feat_line, label_line = self._features[idx], self._labels[idx]
                feat_line = self.features_ids(feat_line.split(","))
                label_line = self.labels_ids(label_line.split(","))
                batch_features[i] = np.asarray(feat_line, np.int64)
                batch_labels[i] = np.asarray(label_line, np.int64)
                idx += 1
            yield batch_features, batch_labels

    def validation_generator(self):
        pass

    def test_generator(self):
        pass

    @property
    def features_string2ids_table(self):
        return self._features_string2ids_table

    @property
    def features_ids2string_table(self):
        return self._features_ids2string_table

    @property
    def labels_string2ids_table(self):
        return self._labels_string2ids_table

    @property
    def labels_ids2string_table(self):
        return self._labels_ids2string_table

    def features_ids(self, features):
        """Convert words to ids.

        Args:
            features: A list of words or characters

        Returns:
            A list of int ids
        """
        features = self._padding_features(features, self._sequence_length)
        ids = []
        for w in features:
            if w not in self._features_string2ids_table:
                ids.append(vocab.UNK_LABEL_ID)
                continue
            ids.append(self._features_string2ids_table[w])
        return ids

    def features_string(self, ids):
        """Convert a sequence of ids to string.

        Args:
            ids: A int list of features ids

        Returns:
            A string list of features
        """
        return [self._features_ids2string_table[i] for i in ids]

    def labels_ids(self, labels):
        """Convert labels to ids.

        Args:
            labels: A list of labels

        Returns:
            A list of int ids
        """
        labels = self._padding_labels(labels, self._sequence_length)
        ids = []
        for w in labels:
            if w not in self._labels_string2ids_table:
                ids.append(vocab.UNK_LABEL_ID)
                continue
            ids.append(self._labels_string2ids_table[w])
        return ids

    def labels_string(self, ids):
        """Convert labels ids to string.

        Args:
            ids: A list of labels ids

        Returns:
            A list of labels
        """
        return [self._labels_ids2string_table[i] for i in ids]

    def features_vocab(self):
        return self._features_vocab

    def labels_vocab(self):
        return self._labels_vocab

    @property
    def features_vocab_size(self):
        return self._features_vocab_size

    @property
    def labels_vocab_size(self):
        return self._labels_vocab_size

    @staticmethod
    def _count_file_lines(file):
        return utils.count_file_lines(file)

    @property
    def features_vocab_file(self):
        return self._features_vocab_file

    @property
    def labels_vocab_file(self):
        return self._labels_vocab_file

    def _generate_features_vocab_file(self):
        self._features_vocab_file = os.path.join(self._features_file[0:str(self._features_file).rfind(os.sep)],
                                                 "vocab.features.txt")
        vocab.generate_feature_vocab(self._features_file, self._features_vocab_file, self._features_vocab_limit)

    def _generate_labels_vocab_file(self):
        self._labels_vocab_file = os.path.join(self._labels_file[0:str(self._labels_file).rfind(os.sep)],
                                               "vocab.labels.txt")
        vocab.generate_label_vocab(self._labels_file, self._labels_vocab_file, self.labels_vocab_limit)

    @staticmethod
    def _create_table(file):
        string2ids = {}
        ids2string = {}
        start = 0
        with open(file, mode="rt", encoding="utf8") as f:
            for line in f:
                if not line:
                    continue
                line = line.strip("\n").strip()
                if not line:
                    continue
                string2ids[line] = start
                ids2string[start] = line
                start += 1
        return string2ids, ids2string

    @staticmethod
    def _padding_features(seq, maxlen):
        if len(seq) > maxlen:
            return seq[0:maxlen]
        return seq + [vocab.UNK_CHARS for _ in range(maxlen - len(seq))]

    @staticmethod
    def _padding_labels(seq, maxlen):
        if len(seq) > maxlen:
            return seq[0:maxlen]
        return seq + [vocab.UNK_LABEL for _ in range(maxlen - len(seq))]


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
    return dataset


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
            padding_values=(tf.cast(feat_vocab_table.lookup(tf.constant(vocab.UNK_CHARS)), tf.int64),
                            tf.cast(label_vocab_table.lookup(tf.constant(vocab.LABEL_S)), tf.int64)),
            drop_remainder=True)

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
