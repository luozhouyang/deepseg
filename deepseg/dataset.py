import logging
import os
import re

import tensorflow as tf


def read_vocab_file(vocab_file):
    words = []
    with open(vocab_file, mode='rt', encoding='utf8') as fin:
        for line in fin:
            line = line.rstrip('\n').strip()
            if not line:
                continue
            words.append(line)
    vocab = {}
    for i, v in enumerate(words):
        vocab[v] = i
    return vocab


def build_tags(words):
    tokens, tags = [], []
    for word in words:
        if len(word) == 1:
            tokens.append(word)
            tags.append('O')
            continue
        for i, c in enumerate(word):
            if i == 0:
                tokens.append(c)
                tags.append('B')
            else:
                tokens.append(c)
                tags.append('I')
    return tokens, tags


def read_files(input_files, callback=None):
    if isinstance(input_files, str):
        input_files = [input_files]
    for f in input_files:
        if not os.path.exists(f):
            logging.warning('File %s does not exist.', f)
            continue
        with open(f, mode='rt', encoding='utf8') as fin:
            for line in fin:
                line = line.rstrip('\n')
                if not line:
                    continue
                if callback:
                    callback(line)
        logging.info('Read file %s finished.', f)
    logging.info('Read all files finished.')


def read_train_files(input_files, sep=' '):
    features, labels = [], []

    def collect_fn(line):
        tokens, tags = build_tags(re.split(sep, line))
        if len(tokens) != len(tags):
            return
        features.append(tokens)
        labels.append(tags)

    read_files(input_files, callback=collect_fn)
    return features, labels


def read_predict_files(input_files):
    features = []

    def collect_fn(line):
        tokens = [w.strip() for w in line if w.strip()]
        features.append(tokens)

    read_files(input_files, callback=collect_fn)
    return features


class LabelMapper:

    def __init__(self):
        self.label2id = {
            'O': 0,
            'B': 1,
            'I': 2
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

    def encode(self, labels):
        ids = [self.label2id.get(label, 0) for label in labels]
        return ids

    def decode(self, ids):
        labels = [self.id2label.get(_id, 'O') for _id in ids]
        return labels


class TokenMapper:

    def __init__(self, vocab_file, unk_token='[UNK]', pad_token='[PAD]'):
        self.token2id = read_vocab_file(vocab_file)
        self.id2token = {v: k for k, v in self.token2id.items()}
        assert len(self.token2id) == len(self.id2token)
        self.unk_token = unk_token
        self.unk_id = self.token2id[self.unk_token]
        self.pad_token = pad_token
        self.pad_id = self.token2id[self.pad_token]

    def encode(self, tokens):
        ids = [self.token2id.get(token, self.unk_id) for token in tokens]
        return ids

    def decode(self, ids):
        tokens = [self.id2token.get(_id, self.unk_token) for _id in ids]
        return tokens


class DatasetBuilder:

    def __init__(self, token_mapper, label_mapper, **kwargs):
        self.token_mapper = token_mapper
        self.label_mapper = label_mapper
        self.feature_pad_id = self.token_mapper.pad_id
        self.label_pad_id = self.label_mapper.label2id['O']

    def build_train_dataset(self, input_files, batch_size, buffer_size, repeat=1, **kwargs):
        features, labels = read_train_files(input_files, sep=kwargs.get('sep', ' '))
        features = [self.token_mapper.encode(x) for x in features]
        labels = [self.label_mapper.encode(x) for x in labels]
        features = tf.ragged.constant(features, dtype=tf.int32)
        labels = tf.ragged.constant(labels, dtype=tf.int32)
        x_dataset = tf.data.Dataset.from_tensor_slices(features)
        # convert ragged tensor to tensor
        x_dataset = x_dataset.map(lambda x: x)
        y_dataset = tf.data.Dataset.from_tensor_slices(labels)
        y_dataset = y_dataset.map(lambda y: y)
        dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
        dataset = dataset.repeat(repeat)
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=([None], [None]),
            padding_values=(self.token_mapper.pad_id, self.label_mapper.label2id['O'])
        )
        return dataset

    def build_valid_dataset(self, input_files, batch_size, buffer_size, repeat=1, **kwargs):
        return self.build_train_dataset(input_files, batch_size, buffer_size, repeat=repeat, **kwargs)

    def build_predict_dataset(self, input_files, batch_size, **kwargs):
        features = read_predict_files(input_files)
        features = [self.token_mapper.encode(x) for x in features]
        features = tf.ragged.constant(features, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices(features)
        dataset = dataset.map(lambda x: x)
        dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=[None],
            padding_values=self.token_mapper.pad_id
        )
        # dataset = dataset.map(lambda x: (x, None))
        return dataset
