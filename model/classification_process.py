import pickle

import pandas as pd
##将ner的识别结果进行组装成分类输入数据
##先根据训练数据找到所有的标签组合
# train_file = open("../data/train_reviews.csv")
# train_label_data = pd.read_csv("/Users/duty/downloads/zhijiang_race/复赛训练集2019-09-01/TRAIN/Train_makeup_labels.csv")
train_label_data = pd.read_csv("../data/concat_train_data.csv")
categories = set(train_label_data['Categories'])
polarities = set(train_label_data['Polarities'])
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
# makeup_cat = {"功效":"GX",
#               "成分": "CF",
#               "气味": "QW",
#               "包装": "BZ",
#               "新鲜度": "XXD",
#               "价格": "JG",
#               "其他": "QT",
#               "使用体验": "SYTY",
#               "整体": "ZT",
#               "尺寸": "CC",
#               "服务": "FW",
#               "物流": "WL",
#               "真伪": "ZW",
#               }
concat_cat = {"使用场景": "SYCJ",
              "价格": "JG",
              "物流": "WL",
              "真伪": "ZW",
              "硬件&性能": "YJXN",
              "软件&性能": "RJXN",
              "服务": "FW",
              "外观": "WG",
              "其他": "QT",
              "整体": "ZT",
              "包装": "BZ",
              "功效": "GX",
              "成分": "CF",
"气味": "QW",
"新鲜度": "XXD",
"使用体验": "SYTY",
"尺寸": "CC"}
cla_pola = {"正面":"ZM",
            "负面": "FM",
            "中性": "ZX"}
##读取训练数据
# train_reviews_data = pd.read_csv("/Users/duty/downloads/zhijiang_race/复赛训练集2019-09-01/TRAIN/Train_laptop_reviews.csv")
train_reviews_data = pd.read_csv("../data/train_label_data.csv")
train_label_data = train_label_data[["id", "AspectTerms", "OpinionTerms", "Categories", "Polarities"]]
join_data = pd.merge(train_reviews_data, train_label_data, on="id")
##将aspectterm和opinionterm何在一起，把类别也合在一起
join_data["terms"] = join_data["AspectTerms"]+join_data["OpinionTerms"]
join_data["terms"] = join_data["terms"].apply(lambda x: x.replace("_", "") if "_" in x else x )
join_data["Categories"] = join_data["Categories"].apply(lambda x: cla_cate.get(x))
join_data["Polarities"] = join_data["Polarities"].apply(lambda x: cla_pola.get(x))
join_data["class"] = join_data["Categories"]+"_"+join_data["Polarities"]
pd.DataFrame.to_csv(join_data[["id", "Reviews", "terms", "class"]], "../data/cla_train_data", header=True, index=None)
##构建id2char的字典
join_data["content_chars"] = join_data["Reviews"].apply(lambda x: list(x))
char_vocabs = set()
for row in list(join_data["content_chars"]):
    for char in row:
        char_vocabs.add(char)
char_vocabs_dict = {}
for i, char in enumerate(char_vocabs):
    char_vocabs_dict[char] = i+1
pickle.dump(char_vocabs_dict, open("../data/cla_char_vocabulary_dict.pkl", "wb"))
##构建类别id2char字典
df = join_data["class"]
new_df = list(df.drop_duplicates())
class_vocabs_dict = {}
for i, row in enumerate(new_df):
    class_vocabs_dict[row] = i+1
pickle.dump(class_vocabs_dict, open("../data/class_terms_vocabulary_dict.pkl", "wb"))
print("ddd")

