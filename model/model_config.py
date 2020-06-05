import numpy as np
class Config(object):
    def __init__(self):
        self.max_len = 120
        self.char_text_embeddings = np.load("../data/char_embedding.npy")
        self.embed_dim = self.char_text_embeddings.shape[1]
        self.max_vocab_size = self.char_text_embeddings.shape[0]
        self.is_cudnn = False
        self.embed_trainable = True
        self.lstm_units = 256
        self.saved_model_dir = ""
        self.max_word_len = 90
        self.max_bigram_len = 120
        self.word_text_embeddings = np.load("../data/word_embedding.npy")
        self.word_embed_dim = self.word_text_embeddings.shape[1]
        self.max_word_vocab_size = self.word_text_embeddings.shape[0]
        self.max_bigram_vocab_size = 100000
        self.bigram_embed_dim = 128
        self.is_char_embed = True
        self.is_word_embed = True