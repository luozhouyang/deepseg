import argparse
import json
import sys

import tensorflow as tf

from deepseg import utils
from deepseg.data import Dataset
from deepseg.model import build_model


def add_arguments(parser):
    parser.add_argument("--config_file", type=str,
                        default=None, help="Config file in JSON format.")


def train(hparams):
    dataset = Dataset(hparams)
    model = build_model()
    model.fit_generator(dataset.training_generator(),
                        steps_per_epoch=1,
                        epochs=hparams.epochs, )


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
