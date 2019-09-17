import pandas as pd
makeup_train_data = pd.read_csv("/Users/duty/downloads/zhijiang_race/复赛训练集2019-09-01/TRAIN/Train_makeup_reviews.csv")
laptop_train_data = pd.read_csv("/Users/duty/downloads/zhijiang_race/复赛训练集2019-09-01/TRAIN/Train_laptop_reviews.csv")
makeup_data_cnt = makeup_train_data.count()##13663
laptop_data_cnt = laptop_train_data.count()
laptop_train_data["id"] = laptop_train_data["id"].apply(lambda x: x+makeup_data_cnt)
makeup_label_data = pd.read_csv("/Users/duty/downloads/zhijiang_race/复赛训练集2019-09-01/TRAIN/Train_makeup_labels.csv")
laptop_label_data = pd.read_csv("/Users/duty/downloads/zhijiang_race/复赛训练集2019-09-01/TRAIN/Train_laptop_labels.csv")
laptop_label_data["id"] = laptop_label_data["id"].apply(lambda x:x+makeup_data_cnt)
concat_train_data = pd.concat([makeup_train_data, laptop_train_data])
concat_label_data = pd.concat([makeup_label_data, laptop_label_data])
pd.DataFrame.to_csv(concat_train_data, "../src_data/concat_train_data.csv", index=False)
pd.DataFrame.to_csv(concat_label_data, "../src_data/concat_label_data.csv", index=False)
print("dddd")