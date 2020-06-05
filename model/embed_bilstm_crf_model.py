import os

from keras import Input, Model
from keras.layers import Embedding, Bidirectional, LSTM, Dropout, TimeDistributed, Dense, Concatenate, K, Lambda, \
    Permute
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_self_attention import SeqSelfAttention
from model_config import Config


class EmbedBilstmCrfModel(object):
    def __init__(self, config):
        self.max_len = config.max_len
        self.max_vocab_size = config.max_vocab_size
        self.embed_dim = config.embed_dim
        self.embed_trainable = config.embed_trainable
        self.char_text_embeddings = config.char_text_embeddings
        self.word_text_embeddings = config.word_text_embeddings
        self.lstm_units = config.lstm_units
        self.saved_model_dir = config.saved_model_dir
        self.max_word_len = config.max_word_len
        self.max_bigram_len = config.max_bigram_len
        self.max_word_vocab_size = config.max_word_vocab_size
        self.word_embed_dim = config.word_embed_dim
        self.max_bigram_vocab_size = config.max_bigram_vocab_size
        self.bigram_embed_dim = config.bigram_embed_dim
    def embed_bilstm_crf_model(self):
        input_char = Input(shape=(self.max_len, ))
        input_word = Input(shape=(self.max_word_len, ))
        input_bigram = Input(shape=(self.max_bigram_len, ))
        char_embedding = Embedding(input_dim=self.max_vocab_size, output_dim=self.embed_dim,
                                   weights=[self.char_text_embeddings], trainable=self.embed_trainable,
                                   mask_zero=False)
        word_embedding = Embedding(input_dim=self.max_word_vocab_size, output_dim=self.word_embed_dim,
                                   weights=[self.word_text_embeddings], trainable=self.embed_trainable,
                                   mask_zero=False)
        # bigram_embedding = Embedding(input_dim=self.max_bigram_vocab_size, output_dim=self.bigram_embed_dim)

        char_embed_output = char_embedding(input_char)
        word_embed_output = word_embedding(input_word)
        # bigram_embed_output = bigram_embedding(input_bigram)
        concat_output = Concatenate(axis=1)([char_embed_output, word_embed_output])
        lstm_out = Bidirectional(LSTM(self.lstm_units,return_sequences=True))(concat_output)
        dropout = Dropout(0.3)(lstm_out)
        permute_concat_output = Permute((2,1))(dropout)
        dense_out = Dense(self.max_len, activation="relu")(permute_concat_output)
        permute_dense_out = Permute((2,1))(dense_out)
        dense = TimeDistributed(Dense(9, activation="relu"))(permute_dense_out) ## 9是词性标注的种类个数

        crf_tags = CRF(9, learn_mode="marginal", sparse_target=True)(dense)
        model = Model(inputs=[input_char,input_word], outputs=crf_tags)
        model.summary()
        model.compile(
            optimizer=Adam(lr=0.01),
            loss=crf_loss,
            metrics=[crf_accuracy]
        )
        plot_model(model, to_file=os.path.join(self.saved_model_dir, 'embed_bilstm_crf_model.png'), show_shapes=True)
        return model
    def embed_bilstm_crf_model_with_pos(self):
        input_char = Input(shape=(self.max_len, ))
        input_word = Input(shape=(self.max_word_len, ))
        input_pos = Input(shape=(self.max_word_len,))
        input_bigram = Input(shape=(self.max_bigram_len, ))
        char_embedding = Embedding(input_dim=self.max_vocab_size, output_dim=self.embed_dim,
                                   weights=[self.char_text_embeddings], trainable=self.embed_trainable,
                                   mask_zero=False)
        word_embedding = Embedding(input_dim=self.max_word_vocab_size, output_dim=self.word_embed_dim,
                                   weights=[self.word_text_embeddings], trainable=self.embed_trainable,
                                   mask_zero=False)
        pos_embedding = Embedding(input_dim=60, output_dim=self.word_embed_dim)
        # bigram_embedding = Embedding(input_dim=self.max_bigram_vocab_size, output_dim=self.bigram_embed_dim)

        char_embed_output = char_embedding(input_char)
        word_embed_output = word_embedding(input_word)
        pos_embed_output = pos_embedding(input_pos)
        # bigram_embed_output = bigram_embedding(input_bigram)
        concat_output = Concatenate(axis=1)([char_embed_output, word_embed_output, pos_embed_output])
        lstm_out = Bidirectional(LSTM(self.lstm_units,return_sequences=True))(concat_output)
        att_out = SeqSelfAttention(attention_activation="sigmoid")(lstm_out)
        dropout = Dropout(0.3)(att_out)
        permute_concat_output = Permute((2,1))(dropout)
        dense_out = Dense(self.max_len, activation="relu")(permute_concat_output)
        permute_dense_out = Permute((2,1))(dense_out)
        dense = TimeDistributed(Dense(9, activation="relu"))(permute_dense_out) ## 9是词性标注的种类个数

        crf_tags = CRF(9, learn_mode="marginal", sparse_target=True)(dense)
        model = Model(inputs=[input_char,input_word, input_pos], outputs=crf_tags)
        model.summary()
        model.compile(
            optimizer=Adam(lr=0.01),
            loss=crf_loss,
            metrics=[crf_accuracy]
        )
        plot_model(model, to_file=os.path.join(self.saved_model_dir, 'embed_bilstm_crf_model.png'), show_shapes=True)
        return model
    def embed_bilstm_crf_model_new(self):
        input_char = Input(shape=(self.max_len,))
        input_word = Input(shape=(self.max_word_len,))
        input_bigram = Input(shape=(self.max_bigram_len,))
        char_embedding = Embedding(input_dim=self.max_vocab_size, output_dim=self.embed_dim,
                                   weights=[self.char_text_embeddings], trainable=self.embed_trainable,
                                   mask_zero=False)
        word_embedding = Embedding(input_dim=self.max_word_vocab_size, output_dim=self.word_embed_dim,
                                   weights=[self.word_text_embeddings], trainable=self.embed_trainable,
                                   mask_zero=False)
        # bigram_embedding = Embedding(input_dim=self.max_bigram_vocab_size, output_dim=self.bigram_embed_dim)

        char_embed_output = char_embedding(input_char)
        word_embed_output = word_embedding(input_word)
        # bigram_embed_output = bigram_embedding(input_bigram)
        char_lstm_out = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(char_embed_output)
        char_dropout = Dropout(0.3)(char_lstm_out)

        word_lstm_out = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(word_embed_output)
        word_dropout = Dropout(0.3)(word_lstm_out)

        concat_output = Concatenate(axis=1)([char_dropout, word_dropout])

        permute_concat_output = Permute((2, 1))(concat_output)
        dense_out = Dense(self.max_len, activation="relu")(permute_concat_output)
        permute_dense_out = Permute((2, 1))(dense_out)
        dense = TimeDistributed(Dense(9, activation="relu"))(permute_dense_out)  ## 9是词性标注的种类个数

        crf_tags = CRF(9, learn_mode="marginal", sparse_target=True)(dense)
        model = Model(inputs=[input_char, input_word], outputs=crf_tags)
        model.summary()
        model.compile(
            optimizer=Adam(lr=0.001),
            loss=crf_loss,
            metrics=[crf_accuracy]
        )
        plot_model(model, to_file=os.path.join(self.saved_model_dir, 'embed_bilstm_crf_model_new.png'), show_shapes=True)
        return model
    def embed_char_bilstm_crf_model(self):
        input_char = Input(shape=(self.max_len,))
        char_embedding = Embedding(input_dim=self.max_vocab_size, output_dim=self.embed_dim,
                                   weights=[self.char_text_embeddings], trainable=self.embed_trainable,
                                   mask_zero=False)

        char_embed_output = char_embedding(input_char)
        # bigram_embed_output = bigram_embedding(input_bigram)
        lstm_out = Bidirectional(LSTM(self.lstm_units, return_sequences=True))(char_embed_output)
        dropout = Dropout(0.3)(lstm_out)
        dense = TimeDistributed(Dense(9, activation="relu"))(dropout)  ## 9是词性标注的种类个数

        crf_tags = CRF(9, learn_mode="marginal", sparse_target=True)(dense)
        model = Model(inputs=input_char, outputs=crf_tags)
        model.summary()
        model.compile(
            optimizer=Adam(lr=0.001),
            loss=crf_loss,
            metrics=[crf_accuracy]
        )
        plot_model(model, to_file=os.path.join(self.saved_model_dir, 'char_embed_bilstm_crf_model.png'), show_shapes=True)
        return model




