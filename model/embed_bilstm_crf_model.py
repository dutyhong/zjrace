import os

from keras import Input, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, TimeDistributed, Dense
from keras.optimizers import Adam
from keras.utils import plot_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy


class EmbedBilstmCrfModel(object):
    def __init__(self, config):
        self.max_len = config.max_len
        self.max_vocab_size = config.max_vocab_size
        self.embed_dim = config.embed_dim
        self.embed_trainable = config.embed_trainable
        self.text_embeddings = config.text_embeddings
        self.lstm_units = config.lstm_units
        self.saved_model_dir = config.saved_model_dir
    def embed_bilstm_crf_model(self):
        input = Input(shape=(self.max_len, ))
        word_embedding = Embedding(input_dim=self.max_vocab_size, output_dim=self.embed_dim,
                                   weights=[self.text_embeddings], trainable=self.embed_trainable,
                                   mask_zero=True)
        embed_output = word_embedding(input)
        lstm_out = Bidirectional(LSTM(self.lstm_units,return_sequences=True))(embed_output)
        dropout = Dropout(0.3)(lstm_out)
        dense = TimeDistributed(Dense(9))(dropout) ## 9是词性标注的种类个数

        crf_tags = CRF(9, sparse_target=True)(dense)
        model = Model(inputs=input, outputs=crf_tags)
        model.summary()
        model.compile(
            optimizer=Adam(lr=0.001),
            loss=crf_loss,
            metrics=[crf_accuracy]
        )
        plot_model(model, to_file=os.path.join(self.saved_model_dir, 'embed_bilstm_crf_model.png'), show_shapes=True)
        return model




