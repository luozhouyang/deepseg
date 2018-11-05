import codecs
import os
import re
from collections import Counter

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from . import utils

PUNCTUATIONS_PATTERNS = re.compile("[#!?+\-*/.()（）｛｝{}\[\]【】]+")

LABEL_S = "S"
LABEL_B = "B"
LABEL_E = "E"
LABEL_M = "M"
UNK_LABEL = "U"
UNK_LABEL_ID = 0
UNK_CHARS = "<unk>"
UNK_CHARS_ID = 0


def word_filter_callback(word):
    return not PUNCTUATIONS_PATTERNS.findall(word)


def oov_filter_callback(word):
    return UNK_CHARS != word


def unk_label_callback(word):
    return word in ['S', 'B', 'E', 'M']


DEFAULT_GENERATE_FEATURE_CALLBACKS = [word_filter_callback, oov_filter_callback]
DEFAULT_GENERATE_LABELS_CALLBACKS = [unk_label_callback]


def generate_vocab(file, limit=None, callbacks=None):
    utils.check_file_exists(file)
    counter = Counter()
    with codecs.getreader("utf8")(tf.gfile.GFile(file, "rb")) as f:
        for line in f:
            if not line:
                continue
            line = line.strip("\n").strip()
            if not line:
                continue
            words = line.split(",")
            for w in words:
                if utils.is_chinese_word(w):
                    for c in w:
                        counter[c] += 1
                    continue
                keep = True
                if callbacks:
                    for cb in callbacks:
                        if not cb(w):
                            keep = False
                            break
                if keep:
                    counter[w] += 1
    c = counter.most_common(limit)
    vocabs = set()
    for item in c:
        vocabs.add(item[0])
    vocabs = sorted(vocabs)
    return vocabs


def generate_feature_vocab(feat_file,
                           vocab_file,
                           vocab_limit=None,
                           callbacks=DEFAULT_GENERATE_FEATURE_CALLBACKS):
    vocabs = generate_vocab(feat_file, vocab_limit, callbacks)
    with codecs.getwriter("utf8")(tf.gfile.GFile(vocab_file, "wb")) as w:
        # write UNK first to make sure UNK_ID is 0
        w.write(UNK_CHARS + "\n")
        for v in vocabs:
            w.write("%s\n" % v)


def generate_label_vocab(labels_file, vocab_file, vocab_limit=None, callbacks=DEFAULT_GENERATE_LABELS_CALLBACKS):
    vocabs = generate_vocab(labels_file, vocab_limit, callbacks)
    with codecs.getwriter("utf8")(tf.gfile.GFile(vocab_file, "wb")) as w:
        w.write(UNK_LABEL + "\n")
        for v in vocabs:
            w.write("%s\n" % v)


def generate_label_vocab_default(vocab_file):
    if os.path.exists(vocab_file):
        return
    with codecs.getwriter("utf8")(tf.gfile.GFile(vocab_file, "wb")) as f:
        f.write(LABEL_S + "\n")
        f.write(LABEL_B + "\n")
        f.write(LABEL_M + "\n")
        f.write(LABEL_E + "\n")


def create_features_vocab_table(vocab_file):
    word2idx = lookup_ops.index_table_from_file(vocab_file, default_value=UNK_CHARS_ID)
    idx2word = lookup_ops.index_to_string_table_from_file(vocab_file, default_value=UNK_CHARS)
    return word2idx, idx2word


def create_labels_vocab_table(vocab_file):
    word2idx = lookup_ops.index_table_from_file(vocab_file, default_value=UNK_LABEL_ID)
    idx2word = lookup_ops.index_to_string_table_from_file(vocab_file, default_value=UNK_LABEL)
    word2idx.lookup()
    return word2idx, idx2word


def load_vocab(file):
    vocabs = []
    with open(file, mode="rt", encoding="utf8") as f:
        for line in f:
            vocabs.append(line.strip("\n").strip())
    return vocabs
