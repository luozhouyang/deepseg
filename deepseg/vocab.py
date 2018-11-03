import codecs
import os
import re
from collections import Counter

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

from . import utils

OOV_PATTERNS = re.compile("[#!?+\-*/.()（）｛｝{}\[\]【】]+")

LABEL_S = "S"
LABEL_B = "B"
LABEL_E = "E"
LABEL_M = "M"

UNK = "<unk>"
UNK_ID = 0


def word_filter_callback(word):
    return not OOV_PATTERNS.findall(word)


DEFAULT_GENERATE_FEATURE_CALLBACKS = [word_filter_callback]


def generate_feature_vocab(feat_file,
                           vocab_file,
                           vocab_limit,
                           oov_token=UNK,
                           callbacks=DEFAULT_GENERATE_FEATURE_CALLBACKS):
    if os.path.exists(vocab_file):
        return
    counter = Counter()
    with codecs.getreader("utf8")(tf.gfile.GFile(feat_file, "rb")) as f:
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
    c = counter.most_common(vocab_limit)
    vocabs = set()
    for item in c:
        vocabs.add(item[0])
    if oov_token in vocabs:
        vocabs.remove(oov_token)
    vocabs = sorted(vocabs)
    with codecs.getwriter("utf8")(tf.gfile.GFile(vocab_file, "wb")) as w:
        # write UNK first to make sure UNK_ID is 0
        w.write(UNK + "\n")
        for v in vocabs:
            w.write("%s\n" % v)


def generate_label_vocab(vocab_file):
    if os.path.exists(vocab_file):
        return
    with codecs.getwriter("utf8")(tf.gfile.GFile(vocab_file, "wb")) as f:
        f.write(LABEL_S + "\n")
        f.write(LABEL_B + "\n")
        f.write(LABEL_M + "\n")
        f.write(LABEL_E + "\n")


def create_vocab_table(vocab_file):
    word2idx = lookup_ops.index_table_from_file(vocab_file, default_value=UNK_ID)
    idx2word = lookup_ops.index_to_string_table_from_file(vocab_file, default_value=UNK)
    return word2idx, idx2word
