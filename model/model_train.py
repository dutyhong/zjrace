import pickle

from keras import callbacks
from keras.callbacks import EarlyStopping
from keras_preprocessing.sequence import pad_sequences

from model.embed_bilstm_crf_model import EmbedBilstmCrfModel
from model_config import Config
import numpy as np
config = Config()
RaceModel = EmbedBilstmCrfModel(config)
model = RaceModel.embed_bilstm_crf_model()
##数据预处理
##将训练数据字符id化
char_dict = pickle.load(open("../data/char_vocabulary_dict.pkl", "rb"))
tag_dict = pickle.load(open("../data/tag_vocabulary_dict.pkl", "rb"))

train_file = open("../data/concat_train.ner.v2", "r")
x_sentence_list = list()
y_sentence_list = list()
x_word_list = list()
y_word_list = list()
for line in train_file.readlines():
    if(len(line)>2):
        columns = line.split("\t")
        x_word_list.append(char_dict.get(columns[0], 0))
        y_word_list.append(tag_dict.get(columns[1].strip("\n"), 0))
    else:
        x_sentence_list.append(x_word_list)
        y_sentence_list.append(y_word_list)
        x_word_list = list()
        y_word_list = list()
x_train = x_sentence_list
y_train = y_sentence_list
x_train = pad_sequences(x_train, maxlen=config.max_len)
y_train = pad_sequences(y_train, maxlen=config.max_len)
y_train = np.expand_dims(y_train, 2)

test_file = open("../data/concat_test.ner.v2", "r")
x_test_sentence_list = list()
y_test_sentence_list = list()
x_test_word_list = list()
y_test_word_list = list()
for line in test_file.readlines():
    if(len(line)>2):
        columns = line.split("\t")
        x_test_word_list.append(char_dict.get(columns[0], 0))
        y_test_word_list.append(tag_dict.get(columns[1].strip("\n"), 0))
    else:
        x_test_sentence_list.append(x_test_word_list)
        y_test_sentence_list.append(y_test_word_list)
        x_test_word_list = list()
        y_test_word_list = list()
x_test = x_test_sentence_list
y_test = y_test_sentence_list
x_test = pad_sequences(x_test, maxlen=config.max_len, padding="post", truncating="post")
y_test = pad_sequences(y_test, maxlen=config.max_len, padding="post", truncating="post")
y_test = np.expand_dims(y_test, 2)
callbacks = [EarlyStopping(monitor="val_loss", patience=20, verbose=1)]
EarlyStopping()
model.fit(x=x_train, y=y_train, epochs=30, batch_size=64, validation_data=(x_test, y_test),verbose=1)
model.save_weights("concat_embed_bilstm_crf_model.h5")
print("ddd")


#####根据makeup和laptop总的训练数据训练模型（因为这两个数据有很多类别相似性:价格， 物流，真伪，服务，包装，其他）




