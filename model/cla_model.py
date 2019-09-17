import pickle

from keras.utils import plot_model, to_categorical
from keras_preprocessing.sequence import pad_sequences

from absa_config import Config
from absa_model import AbsaModel
import pandas as pd
config = Config()
absa_model = AbsaModel(config)
atae_model = absa_model.atae_lstm_new()
atae_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])
plot_model(atae_model, to_file='absa_atae_model')

##d读取训练数据
char_vocabulary_dict = pickle.load(open("../data/char_vocabulary_dict.pkl", "rb"))
class_term_vocabulary_dict = pickle.load(open("../data/class_terms_vocabulary_dict.pkl", "rb"))
train_data = pd.read_csv("../data/cla_train_data")
x_content_train = list(train_data["Reviews"].apply(lambda x: list(x)))
x_aspect_train = list(train_data["terms"].apply(lambda x: list(x)))
x_content_train_ids = []
for row in x_content_train:
    row_list = []
    for word in row:
        id = char_vocabulary_dict.get(word, 0)
        row_list.append(id)
    x_content_train_ids.append(row_list)
x_aspect_train_ids = []
for row in x_aspect_train:
    row_list = []
    for word in row:
        id = char_vocabulary_dict.get(word, 0)
        row_list.append(id)
    x_aspect_train_ids.append(row_list)
y_train = list(train_data["class"])
y_train_ids = []
for label in y_train:
    y_train_ids.append(class_term_vocabulary_dict.get(label, -1))
x_content_train_ids = pad_sequences(x_content_train_ids, padding="post", truncating="post", maxlen=config.max_len)
x_aspect_train_ids = pad_sequences(x_aspect_train_ids, padding="post", truncating="post", maxlen=config.aspect_max_len)
y_train_ids = to_categorical(y_train_ids, num_classes=config.n_classes)
atae_model.fit(x=[x_content_train_ids, x_aspect_train_ids], y=y_train_ids,epochs=30, verbose=1)
plot_model(atae_model, to_file="../data/atae_model.png")
atae_model.save_weights("cla_model.h5")
print("dddd")