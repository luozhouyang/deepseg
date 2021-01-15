import argparse
import logging
import os

from .dataset import DatasetBuilder, LabelMapper, TokenMapper
from .models import *


def add_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bilstm-crf', choices=[
        'bilstm-crf', 'bigru-crf', 'bert-crf', 'albert-crf', 'bert-bilstm-crf', 'albert-bilstm-crf'
    ])
    parser.add_argument('--vocab_file', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--pretrained_model_dir', type=str)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_input_files', type=str, help="Comma splited input files")
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--train_buffer_size', type=int, default=1000000)
    parser.add_argument('--eval_input_files', type=str, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--eval_buffer_size', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--vocab_size', type=int, default=None)
    parser.add_argument('--embedding_size', type=int, default=None)
    parser.add_argument('--hidden_size', type=int, default=256)
    args, _ = parser.parse_known_args()
    return args


def choose_model(args):
    if args.model == 'bilstm-crf':
        assert args.vocab_size is not None, "vocab_size must be set when using bilstm-crf model!"
        assert args.embedding_size is not None, "embedding_size must be set when using bilstm-crf model!"
        model = BiLSTMCRFModel(
            vocab_size=args.vocab_size,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            lr=args.lr)
        return model

    if args.model == 'bigru-crf':
        assert args.vocab_size is not None, "vocab_size must be set when using bigru-crf model!"
        assert args.embedding_size is not None, "embedding_size must be set when using bigru-crf model!"
        model = BiGRUCRFModel(
            vocab_size=args.vocab_size,
            embedding_size=args.embedding_size,
            hidden_size=args.hidden_size,
            lr=args.lr)
        return model

    if args.model == 'bert-crf':
        assert args.pretrained_model_dir is not None, "pretrained_model_dir must be set when using bert-based models"
        model = BertCRFModel(
            pretrained_model_dir=args.pretrained_model_dir,
            lr=args.lr)
        return model

    if args.model == 'bert-bilstm-crf':
        assert args.pretrained_model_dir is not None, "pretrained_model_dir must be set when using bert-based models"
        model = BertBiLSTMCRFModel(
            pretrained_model_dir=args.pretrained_model_dir,
            lr=args.lr)
        return model

    if args.model == 'albert-crf':
        assert args.pretrained_model_dir is not None, "pretrained_model_dir must be set when using albert-based models"
        model = AlbertCRFModel(
            pretrained_model_dir=args.pretrained_model_dir,
            lr=args.lr)
        return model

    if args.model == 'albert-bilstm-crf':
        assert args.pretrained_model_dir is not None, "pretrained_model_dir must be set when using albert-based models"
        model = AlbertBiLSTMCRFModel(
            pretrained_model_dir=args.pretrained_model_dir,
            lr=args.lr)
        return model


def build_dataset(args):
    token_mapper = TokenMapper(vocab_file=args.vocab_file)
    label_mapper = LabelMapper()
    dataset = DatasetBuilder(token_mapper, label_mapper)
    train_dataset = dataset.build_train_dataset(
        input_files=str(args.train_input_files).split(','),
        batch_size=args.train_batch_size,
        buffer_size=args.train_buffer_size)
    if not args.eval_input_files:
        return train_dataset, None
    valid_dataset = dataset.build_valid_dataset(
        input_files=str(args.eval_input_files).split(','),
        batch_size=args.eval_batch_size,
        buffer_size=args.eval_buffer_size)
    return train_dataset, valid_dataset


def train_model(model, train_dataset, valid_dataset, args):
    model_dir = args.model_dir
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    tensorboard_logdir = os.path.join(model_dir, 'logs')
    saved_model_dir = os.path.join(model_dir, 'export', '{epoch}')
    model.fit(
        train_dataset,
        validation_data=valid_dataset,
        epochs=args.epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(monitor='val_loss' if valid_dataset is not None else 'loss'),
            tf.keras.callbacks.TensorBoard(tensorboard_logdir),
            tf.keras.callbacks.ModelCheckpoint(
                saved_model_dir,
                save_best_only=False,
                save_weights_only=False)
        ]
    )


if __name__ == "__main__":
    args = add_arguments()
    model = choose_model(args)
    train_dataset, valid_dataset = build_dataset(args)
    train_model(model, train_dataset, valid_dataset, args)
