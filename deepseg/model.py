import keras
from keras.layers import Dropout, ZeroPadding1D, Conv1D, Dense, TimeDistributed
from keras.layers import Input, Embedding, Bidirectional, LSTM, Concatenate
from keras.models import Model
from keras_contrib.layers import CRF

SEQUENCE_LENGTH = 40
HIDDEN_SIZE = 128
EMBEDDING_SIZE = 256
VOCAB_SIZE = 25000
N_CLASS = 4

DROPOUT = 0.2
HALF_WINDOW_SIZE = 1


def build_model():
    inputs = Input(shape=(SEQUENCE_LENGTH,), name="input")
    embedding = Embedding(VOCAB_SIZE, EMBEDDING_SIZE, input_length=SEQUENCE_LENGTH, name="embedding")(inputs)
    lstm = Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True), name="bilstm")(embedding)
    lstm_d = Dropout(DROPOUT)(lstm)

    padding = ZeroPadding1D(HALF_WINDOW_SIZE)(embedding)
    conv = Conv1D(filters=50, kernel_size=2 * HALF_WINDOW_SIZE + 1)(padding)
    conv_d = Dropout(DROPOUT)(conv)
    dense_conv = TimeDistributed(Dense(50))(conv_d)

    rcnn = Concatenate(axis=2)([lstm_d, dense_conv])
    dense = TimeDistributed(Dense(N_CLASS))(rcnn)

    crf = CRF(N_CLASS, name="crf")
    crf_output = crf(dense)

    model = Model(inputs=[inputs], outputs=[crf_output])
    model.compile(loss=crf.loss_function, optimizer=keras.optimizers.Adam(), metrics=["accuracy"])

    return model
