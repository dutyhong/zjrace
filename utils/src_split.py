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

file1 = "../src_data/concat_train_data.csv"
file2 = "../src_data/concat_label_data.csv"
file3 = "../data/concat_train_content.csv"
file4 = "../data/concat_train_label.csv"
file5 = "../data/concat_test_content.csv"
file6 = "../data/concat_test_label.csv"

# 读取数据
dictionary = {}
with open(file1, "r", encoding="utf-8") as fid1:
	for i, line in enumerate(fid1.readlines()):
		if i == 0:
			continue
		line = line.strip()
		columns = line.split(",")
		ids = columns[0]
		sen = []
		# sen.append(columns[0])
		for i in range(1,len(columns)):
			word = columns[i].replace('"', '')
			sen.append(word)
		tmp_line = "，".join(sen)
		line = ids+","+tmp_line
		dictionary[ids]={"sen":line}
		dictionary[ids]["label"] = []
with open(file2, "r", encoding="utf-8") as fid2:
	for i, line in enumerate(fid2.readlines()):
		if i == 0:
			continue
		line = line.strip()
		ids = line.split(",")[0]
		dictionary[ids]["label"].append(line)

# 校正数据
for ids in dictionary.keys():
	sen = ",".join(dictionary[ids]["sen"].split(",")[1:])
	label = dictionary[ids]["label"]
	for l in range(len(label)):
		ids, asp, asp_head, asp_tail, opi, opi_head, opi_tail, cla, pos = label[l].split(",")

		asp_head = int(asp_head) if asp_head != " " else asp_head
		asp_tail = int(asp_tail) if asp_tail != " " else asp_tail
		opi_head = int(opi_head) if opi_head != " " else opi_head
		opi_tail = int(opi_tail) if opi_tail != " " else opi_tail

		if asp != "_" and sen[asp_head:asp_tail] != asp:
			bias = -1
			while(1):
				if sen[asp_head+bias:asp_tail+bias] == asp:
					asp_head = asp_head+bias
					asp_tail = asp_tail+bias
					break
				bias = bias-1
				if asp_head+bias < 0:
					break

		if opi != "_" and sen[opi_head:opi_tail] != opi:
			bias = -1
			while(1):
				if sen[opi_head+bias:opi_tail+bias] == opi:
					opi_head = opi_head+bias
					opi_tail = opi_tail+bias
					break
				bias = bias-1
				if opi_head+bias < 0:
					break

		dictionary[ids]["label"][l] = ",".join([ids, asp, str(asp_head), str(asp_tail), \
												opi, str(opi_head), str(opi_tail), cla, pos])

# 写入数据
fid3 =  open(file3, "w", encoding="utf-8")
fid4 =  open(file4, "w", encoding="utf-8")
fid5 =  open(file5, "w", encoding="utf-8")
fid6 =  open(file6, "w", encoding="utf-8")

keys =list(dictionary.keys())
random.shuffle(keys)
train_key = keys[0:int(0.85*len(keys))]
#train_key = keys
test_key = keys[int(0.85*len(keys))+1:]
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

# 关闭文件
fid1.close()
fid2.close()
fid3.close()
fid4.close()
fid5.close()
fid6.close()
