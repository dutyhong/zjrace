from keras_preprocessing.sequence import pad_sequences

from absa_config import Config
from absa_model import AbsaModel
import pandas as pd
import pickle
import numpy as np
config = Config()
absa_model = AbsaModel(config)
atae_model = absa_model.atae_lstm_new()
atae_model.load_weights("cla_model.h5")
##从result.ner里面读取预测样本
cla_data = pd.read_csv("../data/result.csv")
##d读取测试数据
char_vocabulary_dict = pickle.load(open("../data/char_vocabulary_dict.pkl", "rb"))
class_term_vocabulary_dict = pickle.load(open("../data/class_terms_vocabulary_dict.pkl", "rb"))
test_data = pd.read_csv("../data/result.csv")
x_content_test = list(test_data["sen_content"].apply(lambda x: list(x)))
x_aspect_test = list(test_data["aspect_content"].apply(lambda x: list(str(x))))
x_content_test_ids = []
for row in x_content_test:
    row_list = []
    for word in row:
        id = char_vocabulary_dict.get(word, 0)
        row_list.append(id)
    x_content_test_ids.append(row_list)
x_aspect_test_ids = []
for row in x_aspect_test:
    row_list = []
    for word in row:
        id = char_vocabulary_dict.get(word, 0)
        row_list.append(id)
    x_aspect_test_ids.append(row_list)

x_content_test_ids = pad_sequences(x_content_test_ids, padding="post", truncating="post", maxlen=config.max_len)
x_aspect_test_ids = pad_sequences(x_aspect_test_ids, padding="post", truncating="post", maxlen=config.aspect_max_len)

predict_result = atae_model.predict(x=[x_content_test_ids,x_aspect_test_ids],verbose=1)
predict_result = np.argmax(predict_result,axis=-1)
##从id2char class
class_term_vocabulary_dict = pickle.load(open("../data/class_terms_vocabulary_dict.pkl", "rb"))
class_id_char_dict = {}
for char, id in class_term_vocabulary_dict.items():
    class_id_char_dict[id] = char
predict_class = []
for id in predict_result:
    class_char = class_id_char_dict.get(id)
    predict_class.append(class_char)
predict_file = open("../data/predict_file.csv", "w")
cla_cate = {"使用场景":"SYCJ",
            "价格": "JG",
            "物流": "WL",
            "真伪": "ZW",
            "硬件&性能": "YJXN",
            "软件&性能":"RJXN",
            "服务": "FW",
            "外观": "WG",
            "其他": "QT",
            "整体": "ZT",
            "包装": "BZ"}
cla_cate_inverse = {}
for name, code in cla_cate.items():
    cla_cate_inverse[code] = name

cla_pola = {"正面":"ZM",
            "负面": "FM",
            "中性": "ZX"}
cla_pola_inverse = {}
for name, code in cla_pola.items():
    cla_pola_inverse[code] = name
predict_file.write("id,aspect_content,sen_content,Categories,Polarities\n")
for value, tmp_value in zip(predict_class, test_data.values):
    # map(str, tmp_value)
    values = str(value).split("_")
    predict_file.write(str(tmp_value[0]+1)+","+str(tmp_value[1])+","+tmp_value[2]+","+str(cla_cate_inverse.get(values[0]))
                       +","+str(cla_pola_inverse.get(values[1]))+"\n")
predict_file.close()

# pd.DataFrame.to_csv(predict_class, "../data/predict_class.csv")
print("ddd")