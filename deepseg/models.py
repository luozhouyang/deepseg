import argparse
import os

import tensorflow as tf
import tensorflow_addons as tfa
from keras_crf import CRF, CRFAccuracy, CRFLoss


def build_bilstm_crf_model(vocab_size, embedding_size, hidden_size=256, num_class=3, lr=5e-5):
    sequence_input = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name='sequence_input')
    sequence_mask = tf.keras.layers.Lambda(lambda x: tf.greater(x, 0))(sequence_input)
    embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)(sequence_input)
    bilstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(hidden_size // 2, return_sequences=True))
    outputs = bilstm(embedding)
    crf = CRF(num_class)
    outputs = crf(outputs, mask=sequence_mask)
    model = tf.keras.Model(inputs=sequence_input, outputs=outputs)
    model.compile(
        loss=CRFLoss(crf),
        metrics=[
            CRFAccuracy(crf, name='accuracy'),
        ],
        optimizer=tf.keras.optimizers.Adam(lr))
    model.summary()
    return model


def build_bigru_crf_model(vocab_size, embedding_size, hidden_size=256, num_class=3, lr=5e-5):
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
        loss=CRFLoss(crf),
        metrics=[
            CRFAccuracy(crf, name='accuracy')
        ],
        optimizer=tf.keras.optimizers.Adam(lr))
    model.summary()
    return model
