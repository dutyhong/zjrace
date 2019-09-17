import pickle

from keras.models import load_model
from keras_preprocessing.sequence import pad_sequences
import pandas as pd
from model.embed_bilstm_crf_model import EmbedBilstmCrfModel
from model_config import Config
import numpy as np
config = Config()
MyModel = EmbedBilstmCrfModel(config)
model = MyModel.embed_bilstm_crf_model()
model.load_weights("concat_embed_bilstm_crf_model.h5")
##将训练数据字符id化
char_dict = pickle.load(open("../data/char_vocabulary_dict.pkl", "rb"))
tag_dict = pickle.load(open("../data/tag_vocabulary_dict.pkl", "rb"))
def id2char(sentence, dict):
    id2char_dict = {}
    result = list()
    for ch, id in dict.items():
        id2char_dict[id] = ch
    for ch in sentence:
        id1 = id2char_dict.get(ch, -1)
        result.append(id1)
    return result

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


x_test = pad_sequences(x_test, maxlen=config.max_len, padding="post", truncating="post")
y_test = pad_sequences(y_test, maxlen=config.max_len, padding="post", truncating="post")
y_test = np.expand_dims(y_test, 2)
# model = load_model("embed_bilstm_crf_model.h5")
# result = model.evaluate(x=x_test, y=y_test, verbose = 1)
results = model.predict(x=x_test,batch_size=32, verbose=1)
results1 = np.argmax(results,axis=2)
x_test_sentence_str = list()
y_test_sentence_str = list()
write_file = open("result_ner", "w")
for sentence, tags in zip(x_test, results1):
    sentence_str = id2char(sentence, char_dict)
    # tags = [i+1 for i in tags]
    tags_str = id2char(tags, tag_dict)
    x_test_sentence_str.append(sentence_str)
    y_test_sentence_str.append(tags_str)
    for word, tag in zip(sentence_str, tags_str):
        write_file.write(str(word)+" "+str(tag)+"\n")
    write_file.write("\n")
write_file.close()
print("ddd")