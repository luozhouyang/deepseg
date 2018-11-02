import tensorflow as tf
from keras_contrib.layers import CRF
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM
from tensorflow.keras.models import Model

SEQUENCE_LENGTH = 40
HIDDEN_SIZE = 128
EMBEDDING_SIZE = 256
VOCAB_SIZE = 25000
N_CLASS = 4

DROPOUT = 0.2


def build_model():
    inputs = Input(shape=(SEQUENCE_LENGTH,), name="input")
    embedding = Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=SEQUENCE_LENGTH, name="embedding")(inputs)
    lstm = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True), name="bilstm")(embedding)
    lstm = Dropout(DROPOUT)(lstm)

    crf = CRF(N_CLASS, name="crf")
    crf_output = crf(lstm)

    model = Model(input=[inputs], output=[crf_output])
    model.compile(loss=crf.loss_function, optimizer=tf.keras.optimizers.Adam, metrics=["accuracy"])

    return model
