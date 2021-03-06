import pickle

from keras import callbacks
from keras.callbacks import EarlyStopping
from keras_preprocessing.sequence import pad_sequences

from model.embed_bilstm_crf_model import EmbedBilstmCrfModel
from model.model_config import Config
import numpy as np
import pandas as pd
import jieba
import jieba.posseg as pseg
import os

from model.vary_train_model import vary_data

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

jieba.load_userdict("../src_data/usrdict.txt")
def fenci(x):
    words = jieba.lcut(x)
    return " ".join(words)
def postag(x):
    postags = pseg.cut(x)
    words = []
    poses = []
    for word, pos in postags:
        words.append(word)
        poses.append(pos)
    return " ".join(poses)
def bigram(x):
    chars = list(x)
    char_b = chars[0]
    bigram_list = []
    char_before = ""
    char_after = ""
    char_before = char_b
    for i in range(1, len(chars)):
        char_after = chars[i]
        bigram = char_before + "" + char_after
        bigram_list.append(bigram)
        char_before = char_after
    return bigram_list

##将训练集进行交叉验证
def vary_data_train(concat_train_data, x_train, validation_cnt = 5):
    train_validation_indices = vary_data(concat_train_data, validation_cnt)
    train_validation_sets = []
    for i in range(len(train_validation_indices)):
        train_validation_set = []
        for j in range(len(x_train)):
            if j in train_validation_indices[i]:
                train_validation_set.append(x_train[j])
        train_validation_sets.append(train_validation_set)
    return train_validation_sets

config = Config()

##数据预处理
##将训练数据字符id化
char_dict = pickle.load(open("../data/char_vocabulary_dict.pkl", "rb"))
tag_dict = pickle.load(open("../data/tag_vocabulary_dict.pkl", "rb"))
word_dict = pickle.load(open("../data/word_vocabulary_dict.pkl", "rb"))
pos_dict = pickle.load(open("../data/pos_vocabulary_dict.pkl", "rb"))
concat_train_data = pd.read_csv("../data/concat_train_content.csv", header=None)
concat_train_data.columns = ["id", "reviews"]
concat_test_data = pd.read_csv("../data/concat_test_content.csv", header=None)
concat_test_data.columns = ["id", "reviews"]
concat_train_data["reviews"] = concat_train_data["reviews"].apply(fenci)
concat_test_data["reviews"] = concat_test_data["reviews"].apply(fenci)
concat_train_data["poses"] = concat_train_data["reviews"].apply(postag)
concat_test_data["poses"] = concat_test_data["reviews"].apply(postag)


train_file = open("../data/concat_train.ner.v2", "r")
x_sentence_list = list()
y_sentence_list = list()
x_word_list = list()
y_word_list = list()
for line in train_file.readlines():
    if (len(line) > 2):
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
x_train_validation_sets = vary_data_train(concat_train_data, x_train, 5)
y_train_validation_sets = vary_data_train(concat_train_data, y_train, 5)



###词
x_train_words_list = []
for words in list(concat_train_data["reviews"]):
    x_word_list = []
    for word in words.split(" "):
        x_word_list.append(word_dict.get(word, 0))
    x_train_words_list.append(x_word_list)
# x_train_words_list = pad_sequences(x_train_words_list, maxlen=config.max_word_len, truncating="post", padding="post")
###标注
x_train_poses_list = []
for poses in list(concat_train_data["poses"]):
    x_pos_list = []
    for pos in poses.split(" "):
        x_pos_list.append(pos_dict.get(pos, 0))
    x_train_poses_list.append(x_pos_list)
# x_train_poses_list = pad_sequences(x_train_poses_list, maxlen=config.max_word_len, truncating="post", padding="post")
x_train_validation_sets = vary_data_train(concat_train_data, x_train, 5)
x_train_words_validation_sets = vary_data_train(concat_train_data, x_train_words_list, 5)
x_train_poses_validation_sets = vary_data_train(concat_train_data, x_train_poses_list, 5)
y_train_validation_sets = vary_data_train(concat_train_data, y_train, 5)

x_train_validation_pad_sets = []
for x_train_validation_set in x_train_validation_sets:
    x_train_validation_set = pad_sequences(x_train_validation_set, maxlen=config.max_len, truncating="post", padding="post")
    x_train_validation_pad_sets.append(x_train_validation_set)
x_train_words_validation_pad_sets = []
for x_train_words_validation_set in x_train_words_validation_sets:
    x_train_words_validation_set = pad_sequences(x_train_words_validation_set, maxlen=config.max_word_len, truncating="post", padding="post")
    x_train_words_validation_pad_sets.append(x_train_words_validation_set)
x_train_poses_validation_pad_sets = []
for x_train_poses_validation_set in x_train_poses_validation_sets:
    x_train_poses_validation_set = pad_sequences(x_train_poses_validation_set, maxlen=config.max_word_len, truncating="post", padding="post")
    x_train_poses_validation_pad_sets.append(x_train_poses_validation_set)

y_train_validation_pad_sets = []
for y_train_validation_set in y_train_validation_sets:
    y_train_validation_set = pad_sequences(y_train_validation_set, maxlen=config.max_len, truncating="post", padding="post")
    y_train_validation_pad_sets.append(np.expand_dims(y_train_validation_set,2))
# y_train_validation_pad_sets = np.expand_dims(y_train_validation_pad_sets, 3)

# x_train = pad_sequences(x_train, maxlen=config.max_len, truncating="post", padding="post")
# y_train = pad_sequences(y_train, maxlen=config.max_len, truncating="post", padding="post")
# y_train = np.expand_dims(y_train, 2)


test_validation_indices = vary_data(concat_test_data, 5)
test_file = open("../data/concat_test.ner.v2", "r")
x_test_sentence_list = list()
y_test_sentence_list = list()
x_test_word_list = list()
y_test_word_list = list()
for line in test_file.readlines():
    if (len(line) > 2):
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
###词
x_test_words_list = []
for words in list(concat_test_data["reviews"]):
    x_word_list = []
    for word in words.split(" "):
        x_word_list.append(word_dict.get(word, 0))
    x_test_words_list.append(x_word_list)
x_test_words_list = pad_sequences(x_test_words_list, maxlen=config.max_word_len, truncating="post", padding="post")
###标注
x_test_poses_list = []
for poses in list(concat_test_data["poses"]):
    x_pos_list = []
    for pos in poses.split(" "):
        x_pos_list.append(pos_dict.get(pos, 0))
    x_test_poses_list.append(x_pos_list)
x_test_poses_list = pad_sequences(x_test_poses_list, maxlen=config.max_word_len, truncating="post", padding="post")
# callbacks = [EarlyStopping(monitor="val_loss", patience=20, verbose=1)]
# EarlyStopping()
RaceModel = EmbedBilstmCrfModel(config)
# model = RaceModel.embed_bilstm_crf_model()
# model = RaceModel.embed_bilstm_crf_model()
# model = RaceModel.embed_char_bilstm_crf_model()
model = RaceModel.embed_bilstm_crf_model_with_pos()
for i in range(5):
    # model.fit(x=[x_train,x_train_words_list, x_train_poses_list], y=y_train, epochs=10, batch_size=64,
    #           validation_data=([x_test,x_test_words_list, x_test_poses_list], y_test), verbose=1)
    model.fit(x=[x_train_validation_pad_sets[i], x_train_words_validation_pad_sets[i], x_train_poses_validation_pad_sets[i]],
              y=y_train_validation_pad_sets[i], epochs=5, batch_size=64, validation_data=([x_test,x_test_words_list, x_test_poses_list], y_test),
              verbose=1)
    model.save_weights("concat_embed_bilstm_crf_model{}.h5".format(i))
print("ddd")

#####根据makeup和laptop总的训练数据训练模型（因为这两个数据有很多类别相似性:价格， 物流，真伪，服务，包装，其他）
