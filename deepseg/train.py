import argparse
import tensorflow as tf
from . import data
from .model import build_model
import os
import json


def add_arguments(parser):
    parser.add_argument("--config_file", type=str,
                        default=None, help="Config file in JSON format.")


def train(hparams):
    model = build_model()
    train_iter = data.build_train_dataset(hparams)
    x, y = train_iter.get_next()
    val_iter = data.build_validation_dataset(hparams)
    model.fit(x, y,
              batch_size=hparams.batch_size,
              epochs=hparams.epochs,
              validation_data=val_iter,
              validation_steps=hparams.validation_steps)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    add_arguments(arg_parser)
    args, _ = arg_parser.parse_known_args()
    if not os.path.exists(args.config_file):
        raise FileNotFoundError("File %s does not exist." % args.config_file)
    with open(args.config_file) as f:
        hp = tf.contrib.training.HParams()
        hp.parse_json(json.loads(f.read()))
    train(hp)
