"""
################################Opinion_extract.py################################
程序名称:     Opinion_extract.py
功能描述:     观点抽取主函数
创建人名:     wuxinhui
创建日期:     2019-08-15
版本说明:     v1.0
################################Opinion_extract.py################################
"""

import numpy as np
import os
import argparse
import copy
import time
import random
import re
import sys
import pickle
import jieba.posseg as psg
import jieba
from model.bert_bilstm_crf_model import bert_bilstm_crf_model
from model.bert_bilstm_model import bert_bilstm_model
from Config import Config
from pyltp import Postagger
from pyltp import Parser

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

sys.path.append("./model")
posC = {"包装":"_BZ",
		"成分":"_CF",
		"尺寸":"_CC",
		"服务":"_FW",
		"功效":"_GX",
		"价格":"_JG",
		"气味":"_QW",
		"使用体验":"_SY",
		"物流":"_WL",
		"新鲜度":"_XX",
		"真伪":"_ZW",
		"整体":"_ZT",
		"其他":"_QT"}
posD = {"正面":"_P",
		"负面":"_N",
		"中性":"_M"}
posC = {v:k for k,v in posC.items()}
posD = {v:k for k,v in posD.items()}


# functions
def strQ2B(ustring):
	rstring = ''
	for uchar in ustring:
		inside_code = ord(uchar)
		if inside_code == 12288:
			inside_code = 32
		elif (inside_code >= 65281 and inside_code <= 65374):
			inside_code -= 65248
		rstring += chr(inside_code)
	return rstring.strip().lower()

def tag_process(senl, tagl):
	
	for l in range(len(tagl)):
		"""
		for i in range(len(tagl[l])-2):
			if "_beg" in tagl[l][i] and "U" == tagl[l][i+1] and "_mid" in tagl[l][i+2]:
				tagl[l][i+1] = tagl[l][i][0]+"_mid"
			if "_mid" in tagl[l][i] and "U" == tagl[l][i+1] and "_mid" in tagl[l][i+2]:
				tagl[l][i+1] = tagl[l][i][0]+"_mid"
			if "_mid" in tagl[l][i] and "U" == tagl[l][i+1] and "_end" in tagl[l][i+2]:
				tagl[l][i+1] = tagl[l][i][0]+"_mid"
			if "U" == tagl[l][i] and "U" == tagl[l][i+1] and "_mid" in tagl[l][i+2]:
				tagl[l][i+1] = tagl[l][i+2][0]+"_beg"
			if "_mid" in tagl[l][i] and "U" == tagl[l][i+1] and "U" == tagl[l][i+2]:
				tagl[l][i+1] = tagl[l][i][0]+"_end"	
			if "U" == tagl[l][i] and "U" == tagl[l][i+1] and "_end" in tagl[l][i+2]:
				tagl[l][i+1] = tagl[l][i+2][0]+"_beg"
			if "_beg" in tagl[l][i] and "U" == tagl[l][i+1] and "U" == tagl[l][i+2]:
				tagl[l][i+1] = tagl[l][i][0]+"_end"	
		"""
		for i in range(len(tagl[l])):
			if tagl[l][i] != "U" and senl[l][i] in [",",'.',"?","!"]:
				tagl[l][i] = "U"
			if tagl[l][i] != "U" and re.search("[^\\u4E00-\\u9FA5a-zA-Z0-9]+",senl[l][i],re.I) != None:
				tagl[l][i] = "U"
	return tagl

def parserS(tag):
	idL,idO = [],[]
	for j in range(len(tag)):
		if "beg" in tag[j]:
			idL = []
			idL.append(j)
		if "mid" in tag[j]:
			idL.append(j)
		if "end" in tag[j]:
			if idL != []:
				idL.append(j)
				idO.append(idL)
			idL = []
		if tag[j] in ["A","O"]:
			idO.append([j])
			idL = []
		if tag[j] == "U":
			idL = []
	return idO

def parserQ(idO, sen, tag):
	idQ = 0
	segl = []
	for l in idO:
		head = l[0]
		tail = l[-1]
		segl.extend(jieba.lcut(sen[idQ:head], cut_all=False))
		segl.extend([sen[head:tail+1]])
		idQ = tail+1
	segl.extend(jieba.lcut(sen[idQ:len(sen)], cut_all=False))
	labl,idQ = [],0
	for l in segl:
		head = idQ
		tail = idQ+len(l)
		if list(range(head, tail)) in idO:
			labl.append(tag[head][0])
		else:
			labl.append("U")
		idQ = idQ+len(l)
	return segl, labl


def parserE(dic, segl, labl, ids, rely_id, relation):
	
	rely_list = []
	for l in range(len(segl)-1):
		if labl[l] == "A" and labl[l+1] == "O":
			rely_list.append((segl[l],segl[l+1]))
			labl[l], labl[l+1] = "U", "U"
	
	for l in range(len(segl)-2):
		if labl[l] == "A" and labl[l+1] == "U" and labl[l+2] == "O" and segl[l+1] not in [",",".","!","?"]:
			rely_list.append((segl[l],segl[l+2]))
			labl[l], labl[l+2] = "U", "U"
	
	for l in range(len(segl)):
		cur_id, rel_id = l, rely_id[l]-1
		if rel_id < 0:
			continue
		if (labl[cur_id], labl[rel_id]) == ("A","O"):
			rely_list.append((segl[cur_id], segl[rel_id]))
			labl[cur_id], labl[rel_id] = "U", "U"
		if (labl[cur_id], labl[rel_id]) == ("O","A"):
			rely_list.append((segl[rel_id], segl[cur_id]))
			labl[cur_id], labl[rel_id] = "U", "U"
	
	for l in range(len(segl)):
		if labl[l] == "A":
			rely_list.append((segl[l], "_"))
		if labl[l] == "O":
			rely_list.append(("_", segl[l]))

	for l in rely_list:
		if [l[0], l[1], "_", "_"] not in dic[ids]["opi"]:
			dic[ids]["opi"].append([l[0], l[1], "_", "_"])
	return dic


class Ner(bert_bilstm_crf_model):
	"""docstring for Ner"""
	def __init__(self, Config):
		Config.embeds_dim = Config.ner_embeds_dim
		Config.layerid = Config.ner_layerid
		super(Ner, self).__init__(Config)
		if Config.if_train_ner == True:
			self._model_train_(pickle.load(open(Config.ner_train_data, "rb")))
			self._model_test_(pickle.load(open(Config.ner_test_data, "rb")))
		return

class Cla(bert_bilstm_model):
	"""docstring for Cla"""
	def __init__(self, Config):
		Config.embeds_dim = Config.cla_embeds_dim
		Config.layerid = Config.cla_layerid
		super(Cla, self).__init__(Config)
		if Config.if_train_cla == True:
			self._model_train_(pickle.load(open(Config.cla_train_data, "rb")))
			self._model_test_(pickle.load(open(Config.cla_test_data, "rb")))
		return

class Opinion_extract(object):
	"""docstring for Opinion_extract"""
	def __init__(self,):
		super(Opinion_extract, self).__init__()

	def _reader(self, file, label):
		if label == "file":
			self.senl = []
			self.idsl = []
			with open(file, "r", encoding="utf-8") as fid:
				for line in fid.readlines():
					line = line.strip()
					ids, sen = line.split(",")
					sen = strQ2B(sen).replace(" ","")
					self.idsl.append(ids)
					self.senl.append(sen)
			del self.idsl[0]
			del self.senl[0]
		if label == "char":
			self.idsl = "0"
			self.senl = [file]
		return

	def _pos_compile(self):
		self.data = []
		for sen in self.senl:
			pos = []
			for i in psg.cut(sen):
				w, t = i.word, i.flag
				if len(w) == 1:
					pos.append(t)
				else:
					pos.extend([t+"_beg"]+[t+"_mid"]*(len(w)-2)+[t+"_end"])
			self.data.append((sen,pos))
		return

	def _ner_compile(self, result):
		# self.tagl = Ner._model_predict_batch(self.data)
		self.tagl = result
		self.tagl = tag_process(self.senl, self.tagl)
		with open("ner.res", "w", encoding="utf-8") as fid:
			for i in range(len(self.data)):
				sen, pos = self.data[i]
				fid.write(sen+"\n")
				fid.write(",".join(self.tagl[i])+"\n")
		return

	def _cla_compile(self, Cla):
		self.label = Cla._model_predict_batch_(self.cla_data)
		return

	def _par_compile_(self):
		self.dic = {}
		LTP_DATA_DIR = "./ltp_data_v3.4.0"
		postagger = Postagger() 
		parser = Parser()
		par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model') 
		pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
		postagger.load(pos_model_path)
		parser.load(par_model_path)
		fid = open("ner.pos", "w", encoding="utf-8")
		for i in range(len(self.idsl)):
			self.dic[self.idsl[i]] = {}
			self.dic[self.idsl[i]]["sen"] = self.senl[i]
			self.dic[self.idsl[i]]["opi"] = []
			idO  = parserS(self.tagl[i])
			segl, labl = parserQ(idO, self.senl[i], self.tagl[i]) 
			postags = postagger.postag(segl)
			arcs = parser.parse(segl, postags)
			rely_id = [arc.head for arc in arcs] 
			relation = [arc.relation for arc in arcs]
			fid.write("\t".join(segl)+"\n")
			fid.write("\t".join([str(w) for w in rely_id])+"\n")
			fid.write("\t".join(relation)+"\n")
			fid.write("///////"+"\n")
			self.dic = parserE(self.dic, segl, labl, self.idsl[i], rely_id, relation)
		self.cla_data = []
		self.src_data = []
		for i in self.dic.keys():
			sens = self.dic[i]["sen"]
			labs = self.dic[i]["opi"]
			if labs == []:
				self.src_data.append((i, sens, "_", "_", "_", "_"))
				self.cla_data.append((("_"+"_").strip("_"),sens))
				continue
			for l in labs:
				self.src_data.append((i, sens, l[0], l[1], l[2], l[3]))
				self.cla_data.append(((l[0]+l[1]).strip("_"),sens))
		return
		
	def _writer(self, file):
		fid = open("./result/Result.csv", "w", encoding="utf-8")
		for i in range(len(self.label)):
			ids, sen, asp, opi, cla, pos = self.src_data[i]
			cla = posC[self.label[i][:3]]
			pos = posD[self.label[i][-2:]]
			if asp == "_" and opi == "_":
				cla, pos = "_", "_"
			fid.write(ids+","+asp+","+opi+","+cla+","+pos+"\n")
		fid.close()
		return
		
if __name__ == "__main__":
	Config = Config()
	Opi = Opinion_extract()
	Opi._reader("./src_data/Test_reviews.csv", "file")
	#Opi._reader("./data/test_content.csv", "file")
	Opi._pos_compile()
	Opi._ner_compile()
	Opi._par_compile_()
	Opi._cla_compile(Cla(Config))
	Opi._writer("./result/Result.csv")
		
		
