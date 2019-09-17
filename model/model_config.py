import numpy as np
class Config(object):
    def __init__(self):
        self.max_len = 120
        self.text_embeddings = np.load("../data/char_embedding.npy")
        self.embed_dim = self.text_embeddings.shape[1]
        self.max_vocab_size = self.text_embeddings.shape[0]
        self.is_cudnn = False
        self.embed_trainable = True
        self.lstm_units = 64
        self.saved_model_dir = ""