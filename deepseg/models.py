import argparse
import os

import tensorflow as tf
import tensorflow_addons as tfa
from keras_crf import CRF
from transformers_keras import Albert, Bert


def BiLSTMCRFModel(vocab_size, embedding_size, hidden_size=256, num_class=3, lr=5e-5):
    sequence_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='sequence_input')
    sequence_mask = tf.keras.layers.Lambda(lambda x: tf.greater(x, 0))(sequence_input)
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)(sequence_input)
    bilstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size // 2, return_sequences=True))
    outputs = bilstm(embedding)
    crf = CRF(num_class)
    outputs = crf(inputs=outputs, mask=sequence_mask)
    model = tf.keras.Model(inputs=sequence_input, outputs=outputs)
    model.compile(
        loss=crf.neg_log_likelihood,
        metrics=[
            crf.accuracy,
        ],
        optimizer=tf.keras.optimizers.Adam(lr))
    model.summary()
    return model


def BiGRUCRFModel(vocab_size, embedding_size, hidden_size=256, num_class=3, lr=5e-5):
    sequence_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='sequence_input')
    sequence_mask = tf.keras.layers.Lambda(lambda x: tf.greater(x, 0))(sequence_input)
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)(sequence_input)
    bigru = tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(hidden_size // 2, return_sequences=True))
    outputs = bigru(embedding)
    crf = CRF(num_class)
    outputs = crf(outputs)
    model = tf.keras.Model(inputs=sequence_input, outputs=outputs)
    model.compile(
        loss=crf.neg_log_likelihood,
        metrics=[
            crf.accuracy
        ],
        optimizer=tf.keras.optimizers.Adam(lr))
    model.summary()
    return model


def BertCRFModel(pretrained_model_dir, num_class=3, lr=5e-5):
    sequence_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='sequence_input')
    sequence_mask = tf.keras.layers.Lambda(lambda x: tf.greater(x, 0))(sequence_input)
    bert = Bert.from_pretrained(pretrained_model_dir)
    outputs, _, _, _ = bert(inputs=(sequence_input,))
    crf = CRF(num_class)
    outputs = crf(outputs)
    model = tf.keras.Model(inputs=sequence_input, outputs=outputs)
    model.compile(
        loss=crf.neg_log_likelihood,
        metrics=[
            crf.accuracy
        ],
        optimizer=tf.keras.optimizers.Adam(lr))
    model.summary()
    return model


def AlbertCRFModel(pretrained_model_dir, num_class=3, lr=5e-5):
    sequence_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='sequence_input')
    sequence_mask = tf.keras.layers.Lambda(lambda x: tf.greater(x, 0))(sequence_input)
    albert = Albert.from_pretrained(pretrained_model_dir)
    outputs, _, _, _ = albert(inputs=(sequence_input,))
    crf = CRF(num_class)
    outputs = crf(outputs)
    model = tf.keras.Model(inputs=sequence_input, outputs=outputs)
    model.compile(
        loss=crf.neg_log_likelihood,
        metrics=[
            crf.accuracy
        ],
        optimizer=tf.keras.optimizers.Adam(lr))
    model.summary()
    return model


def BertBiLSTMCRFModel(pretrained_model_dir, hidden_size=256, num_class=3, lr=5e-5):
    sequence_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='sequence_input')
    sequence_mask = tf.keras.layers.Lambda(lambda x: tf.greater(x, 0))(sequence_input)
    bert = Bert.from_pretrained(pretrained_model_dir)
    outputs, _, _, _ = bert(inputs=(sequence_input,))
    bilstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size // 2, return_sequences=True))
    outputs = bilstm(outputs)
    crf = CRF(num_class)
    outputs = crf(outputs)
    model = tf.keras.Model(inputs=sequence_input, outputs=outputs)
    model.compile(
        loss=crf.neg_log_likelihood,
        metrics=[
            crf.accuracy
        ],
        optimizer=tf.keras.optimizers.Adam(lr))
    model.summary()
    return model


def AlbertBiLSTMCRFModel(pretrained_model_dir, hidden_size=256, num_class=3, lr=5e-5):
    sequence_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='sequence_input')
    sequence_mask = tf.keras.layers.Lambda(lambda x: tf.greater(x, 0))(sequence_input)
    albert = Albert.from_pretrained(pretrained_model_dir)
    outputs, _, _, _ = albert(inputs=(sequence_input,))
    bilstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size // 2, return_sequences=True))
    outputs = bilstm(outputs)
    crf = CRF(num_class)
    outputs = crf(outputs)
    model = tf.keras.Model(inputs=sequence_input, outputs=outputs)
    model.compile(
        loss=crf.neg_log_likelihood,
        metrics=[
            crf.accuracy
        ],
        optimizer=tf.keras.optimizers.Adam(lr))
    model.summary()
    return model
