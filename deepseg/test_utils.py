import os

import tensorflow as tf


def get_test_file(name):
    test_dir = os.path.join(os.path.dirname(__file__), "../testdata")
    return os.path.join(test_dir, name)


def _create_test_hparams():
    hparams = tf.contrib.training.HParams()
    hparams.add_hparam("features_file", get_test_file("feature.txt"))
    hparams.add_hparam("labels_file", get_test_file("label.txt"))
    hparams.add_hparam("features_vocab_file", get_test_file("vocab.feature.txt"))
    hparams.add_hparam("labels_vocab_file", get_test_file("vocab.label.txt"))
    hparams.add_hparam("epochs", 5)
    hparams.add_hparam("random_seed", 3)
    hparams.add_hparam("max_len", 40)
    hparams.add_hparam("batch_size", 32)
    hparams.add_hparam("buff_size", 1)
    hparams.add_hparam("skip_count", 0)
    return hparams
