import pickle

from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
from model.embed_bilstm_crf_model import EmbedBilstmCrfModel
from model.model_config import Config
import numpy as np
import jieba
import jieba.posseg as pseg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

config = Config()
MyModel = EmbedBilstmCrfModel(config)
# model = MyModel.embed_bilstm_crf_model()
# model = MyModel.embed_char_bilstm_crf_model()
model = MyModel.embed_bilstm_crf_model_with_pos()
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

def id2char(sentence, dict):
    id2char_dict = {}
    result = list()
    for ch, id in dict.items():
        id2char_dict[id] = ch
    for ch in sentence:
        if ch=="-1" or ch==-1:
            id1 = -3
        else:
            id1 = id2char_dict.get(ch, -2)
        result.append(id1)
    return result

test_file = open("../data/concat_O_test.ner.v2", "r")
x_test_sentence_list = list()
y_test_sentence_list = list()
x_test_word_list = list()
y_test_word_list = list()
for line in test_file.readlines():
    if(len(line)>2):
        columns = line.split("\t")
        x_test_word_list.append(char_dict.get(columns[0], 0))
        y_test_word_list.append(tag_dict.get(columns[1].strip("\n"), 0))
        # x_test_word_list.append(columns[0])
        # y_test_word_list.append(columns[1])
    else:
        x_test_sentence_list.append(x_test_word_list)
        y_test_sentence_list.append(y_test_word_list)
        x_test_word_list = list()
        y_test_word_list = list()
x_test = x_test_sentence_list
y_test = y_test_sentence_list
##直接测试laptop的测试集
# laptop_x_test = list()
# laptop_test_reviews = pd.read_csv("../src_data/Test_laptop_reviews.csv")
# laptop_test_reviews["char_content_list"] = laptop_test_reviews["Reviews"].apply(lambda x: list(x))
# for row in list(laptop_test_reviews["char_content_list"]):
#     row_list = list()
#     for char in row:
#         row_list.append(char_dict.get(char, 0))
#     laptop_x_test.append(row_list)


x_test = pad_sequences(x_test, maxlen=config.max_len, padding="post", truncating="post", value=-1)
y_test = pad_sequences(y_test, maxlen=config.max_len, padding="post", truncating="post")
y_test = np.expand_dims(y_test, 2)
x_test_words_list = []
concat_test_data = pd.read_csv("../data/concat_test_content.csv", header=None)
concat_test_data.columns = ["id", "reviews"]
concat_test_data["reviews"] = concat_test_data["reviews"].apply(fenci)
concat_test_data["poses"] = concat_test_data["reviews"].apply(postag)
for words in list(concat_test_data["reviews"]):
    x_word_list = []
    for word in words.split(" "):
        x_word_list.append(word_dict.get(word, 0))
    x_test_words_list.append(x_word_list)
x_test_words_list = pad_sequences(x_test_words_list, maxlen=config.max_word_len, truncating="post", padding="post")
x_test_poses_list = []
for poses in list(concat_test_data["poses"]):
    x_pos_list = []
    for pos in poses.split(" "):
        x_pos_list.append(pos_dict.get(pos, 0))
    x_test_poses_list.append(x_pos_list)
x_test_poses_list = pad_sequences(x_test_poses_list, maxlen=config.max_word_len, truncating="post", padding="post")
# model = load_model("embed_bilstm_crf_model.h5")
# result = model.evaluate(x=x_test, y=y_test, verbose = 1)
model.load_weights("O_concat_embed_bilstm_crf_model.h5")
results = model.predict(x=[x_test,x_test_words_list, x_test_poses_list],batch_size=64, verbose=1)
results1 = np.argmax(results,axis=2)
x_test_sentence_str = list()
y_test_sentence_str = list()
write_file = open("O_result_ner", "w")
for sentence, tags in zip(x_test, results1):
    sentence_str = id2char(sentence, char_dict)
    # tags = [i+1 for i in tags]
    tags_str = id2char(tags, tag_dict)
    x_test_sentence_str.append(sentence_str)
    y_test_sentence_str.append(tags_str)
    for word, tag in zip(sentence_str, tags_str):
        if word=="-3" or word==-3:
            continue
        write_file.write(str(word)+"\t"+str(tag)+"\n")
    write_file.write("\n")
write_file.close()
print("ddd")