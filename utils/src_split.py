"""
################################split.py################################
程序名称:     split.py
功能描述:     数据分割
创建人名:     wuxinhui
创建日期:     2019-08-08
版本说明:     v1.0
################################split.py################################
"""
from sklearn.model_selection import train_test_split
import random

file1 = "../src_data/Train_reviews.csv"
file2 = "../src_data/Train_labels.csv"
file3 = "../data/train_content.csv"
file4 = "../data/train_label.csv"
file5 = "../data/test_content.csv"
file6 = "../data/test_label.csv"

dictionary = {}
with open(file1, "r", encoding="utf-8") as fid1:
	for i, line in enumerate(fid1.readlines()):
		if i == 0:
			continue
		line = line.strip()
		ids = line.split(",")[0]
		sen = "".join(line.split(",")[1])
		dictionary[ids]={"sen":line}
		dictionary[ids]["label"] = []
with open(file2, "r", encoding="utf-8") as fid2:
	for i, line in enumerate(fid2.readlines()):
		if i == 0:
			continue
		line = line.strip()
		ids = line.split(",")[0]
		dictionary[ids]["label"].append(line)

fid3 =  open(file3, "w", encoding="utf-8")
fid4 =  open(file4, "w", encoding="utf-8")
fid5 =  open(file5, "w", encoding="utf-8")
fid6 =  open(file6, "w", encoding="utf-8")

keys =list(dictionary.keys())
random.shuffle(keys)
train_key = keys[0:int(0.9*len(keys))]
#train_key = keys
test_key = keys[int(0.9*len(keys))+1:]
for k in train_key:
	sen = dictionary[k]["sen"]
	lab = dictionary[k]["label"]
	fid3.write(sen+"\n")
	for l in lab:
		fid4.write(l+"\n")

for k in test_key:
	sen = dictionary[k]["sen"]
	lab = dictionary[k]["label"]
	fid5.write(sen+"\n")
	for l in lab:
		fid6.write(l+"\n")

fid3.close()
fid4.close()
fid5.close()
fid6.close()
