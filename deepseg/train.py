import argparse
import json
import sys

import tensorflow as tf

from deepseg import utils
from deepseg import vocab, data
from deepseg.model import build_model


def add_arguments(parser):
    parser.add_argument("--config_file", type=str,
                        default=None, help="Config file in JSON format.")


def train(hparams):
    feat_vocab_table, _ = vocab.create_vocab_table(hparams.feature_vocab_file)
    label_vocab_table, _ = vocab.create_vocab_table(hparams.label_vocab_file)
    train_iter = data.build_train_dataset(feat_vocab_table, label_vocab_table, hparams)
    x, y = train_iter.get_next()
    val_iter = None
    if hparams.validation_features_file and hparams.validation_labels_file:
        val_feat_vocab_table, _ = vocab.create_vocab_table(hparams.validation_features_file)
        val_label_vocab_table, _ = vocab.create_vocab_table(hparams.validation_labels_file)
        val_iter = data.build_validation_dataset(val_feat_vocab_table, val_label_vocab_table, hparams)

    model = build_model()
    model.fit(x, y,
              batch_size=hparams.batch_size,
              epochs=hparams.epochs,
              validation_data=val_iter,
              validation_steps=hparams.validation_steps)


def create_hparams(config_file):
    utils.check_file_exists(config_file)
    hp = tf.contrib.training.HParams()
    with open(config_file, mode="rt", encoding="utf8") as f:
        for k, v in json.loads(f.read()).items():
            hp.add_hparam(k, v)

    return hp


def main(unused_argv):
    hparams = create_hparams(ARGS.config_file)
    train(hparams)


ARGS = None

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    add_arguments(arg_parser)
    ARGS, unparsed = arg_parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
