"""
################################ner_utils.py################################
程序名称:     ner_utils.py
功能描述:     数据处理
创建人名:     wuxinhui
创建日期:     2019-07-31
版本说明:     v1.0
################################ner_utils.py################################
"""

import os
import sys
from tqdm import tqdm
import jieba.posseg as psg
import json
import pickle
import random
import numpy as np
from sklearn.model_selection import train_test_split

class Data_utils(object):

	"""docstring for Data_utils"""

	def __init__(self, file1, file2):
		super(Data_utils, self).__init__()
		self._min_count = 1
		self._sample_file = file1
		self._out = file2
		self._vocab = {}
		self._tag = {}
		self._sample = []
	
	def _reader(self):
		fid = open(self._sample_file, "r", encoding="utf-8")
		senl = []
		tagl = []
		for line in tqdm(fid.readlines()):
			try:
				sen,tag = line.strip().split("\t")
				senl.append(self._strQ2B(sen).lower())
				tagl.append(tag)
			except:
				self._sample.append(("".join(senl),tagl))
				senl = []
				tagl = []
		fid.close()
		return
		
	def _strQ2B(self, ustring):
		rstring = ''
		for uchar in ustring:
			inside_code = ord(uchar)
			if inside_code == 12288:
				inside_code = 32
			elif (inside_code >= 65281 and inside_code <= 65374):
				inside_code -= 65248
			rstring += chr(inside_code)
		return rstring
		
	def _one_hot(self, labels, dim):
		results = np.zeros((len(labels), dim))
		for i, label in enumerate(labels):
			results[i][label] = 1
		return results
		
	def _vocab_build(self):
		tags = []
		for l in tqdm(self._sample):
			senl, tagl = l
			tags.extend(tagl)
		self._tag = {t:1 for t in set(tags)}
		del self._tag["U"]
		self._id2tag = {i+1:j for i,j in enumerate(self._tag)}
		self._id2tag[0] = "U"
		self._tag2id = {v:k for k,v in self._id2tag.items()}
		self._tag["U"] = 1
		if "train" in self._out:
			pickle.dump((self._tag,self._id2tag,self._tag2id), open("../saved_models/ner_tag_dict.pl", "wb"), -1)
		return
		
	def _tag2id_func(self, tagl):
		return [self._tag2id[t] for t in tagl]

	def _id2tag_func(self, ids):
		return [self._id2tag[i] for i in ids]
	
	def _sample_build(self, shuffle=True):
		y = []
		X = []
		self.pos_dict = {"PAD":0}
		for line in tqdm(self._sample):
			senl,tagl = line
			posl = []
			for i in psg.cut(senl):
				w = i.word
				t = i.flag
				if t not in self.pos_dict.keys():
					self.pos_dict[t] = len(self.pos_dict)
					self.pos_dict[t+"_beg"] = len(self.pos_dict)
					self.pos_dict[t+"_mid"] = len(self.pos_dict)
					self.pos_dict[t+"_end"] = len(self.pos_dict)
				if len(w) == 1:
					posl.append(t)
				else:
					posl.extend([t+"_beg"]+[t+"_mid"]*(len(w)-2)+[t+"_end"])
			X.append((senl,posl))
			y.append(tagl)
		pickle.dump((X,y), open(self._out, "wb"), -1)
		if "train" in self._out:
			pickle.dump(self.pos_dict, open("../saved_models/ner_pos_dict.pl", "wb"), -1)
		return
		
if __name__ == "__main__":
	file1 = sys.argv[1]
	file2 = sys.argv[2]
	dhelp = Data_utils(file1, file2)
	dhelp._reader()
	dhelp._vocab_build()	
	dhelp._sample_build()
