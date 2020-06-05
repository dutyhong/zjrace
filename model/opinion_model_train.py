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

config = Config()
##数据预处理
##将训练数据字符id化
char_dict = pickle.load(open("../data/char_vocabulary_dict.pkl", "rb"))
tag_dict = pickle.load(open("../data/tag_vocabulary_dict.pkl", "rb"))
##把tag-dict改一下
tag_dict_tmp = set()
for tag, id in tag_dict.items():
    if "O" not in tag:
        tag = 'U'
    tag_dict_tmp.add(tag)
tag_dict.clear()
for i, tag in enumerate(tag_dict_tmp):
    tag_dict[tag] = i
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


train_file = open("../data/concat_O_train.ner.v2", "r")
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



###词
x_train_words_list = []
for words in list(concat_train_data["reviews"]):
    x_word_list = []
    for word in words.split(" "):
        x_word_list.append(word_dict.get(word, 0))
    x_train_words_list.append(x_word_list)
x_train_words_list = pad_sequences(x_train_words_list, maxlen=config.max_word_len, truncating="post", padding="post")
###标注
x_train_poses_list = []
for poses in list(concat_train_data["poses"]):
    x_pos_list = []
    for pos in poses.split(" "):
        x_pos_list.append(pos_dict.get(pos, 0))
    x_train_poses_list.append(x_pos_list)
x_train_poses_list = pad_sequences(x_train_poses_list, maxlen=config.max_word_len, truncating="post", padding="post")

x_train = pad_sequences(x_train, maxlen=config.max_len, truncating="post", padding="post")
y_train = pad_sequences(y_train, maxlen=config.max_len, truncating="post", padding="post")
y_train = np.expand_dims(y_train, 2)


test_file = open("../data/concat_O_test.ner.v2", "r")
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

RaceModel = EmbedBilstmCrfModel(config)
# model = RaceModel.embed_bilstm_crf_model()
# model = RaceModel.embed_bilstm_crf_model()
# model = RaceModel.embed_char_bilstm_crf_model()
model = RaceModel.embed_bilstm_crf_model_with_pos()
model.fit(x=[x_train, x_train_words_list, x_train_poses_list],
              y=y_train, epochs=5, batch_size=64, validation_data=([x_test,x_test_words_list, x_test_poses_list], y_test),
              verbose=1)
model.save_weights("O_concat_embed_bilstm_crf_model.h5")