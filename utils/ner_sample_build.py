
"""
################################ner_sample_build.py################################
程序名称:     ner_sample_build.py
功能描述:     标注样本构建
创建人名:     wuxinhui
创建日期:     2019-08-08
版本说明:     v1.0
################################ner_sample_build.py################################
"""
import sys

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

def tagger(tag, head, tail, pos, ids, lab):
	head = int(head)
	tail = int(tail)
	if tail == head+1:
		tag[head] = lab
	else:
		tag[head] = lab+"_beg" 
		tag[tail-1] = lab+"_end"
		for i in range(head+1, tail-1):
			tag[i] = lab+"_mid"
	return tag

class logger(object):
	
	"""docstring for logger"""
	def __init__(self):
		super(logger, self).__init__()

	def reader(self, file1, file2):
		self.dictionary = {}
		with open(file1, "r", encoding="utf-8") as fid1:
			for i, line in enumerate(fid1.readlines()):
				line = line.strip()
				ids = line.split(",")[0]
				sen = "".join(line.split(",")[1])
				self.dictionary[ids]={"sen":sen}
				self.dictionary[ids]["lab"] = []
		with open(file2, "r", encoding="utf-8") as fid2:
			for i, line in enumerate(fid2.readlines()):
				line = line.strip()
				ids, asp, asp_head, asp_tail, opi, opi_head, opi_tail, cla, pos = line.split(",")
				self.dictionary[ids]["lab"].append((asp,asp_head,asp_tail,opi,opi_head,opi_tail,cla,pos))
		return

	def parser(self, file):
		# posC = {"包装":"_BZ",
		# 		"成分":"_CF",
		# 		"尺寸":"_CC",
		# 		"服务":"_FW",
		# 		"功效":"_GX",
		# 		"价格":"_JG",
		# 		"气味":"_QW",
		# 		"使用体验":"_SY",
		# 		"物流":"_WL",
		# 		"新鲜度":"_XX",
		# 		"真伪":"_ZW",
		# 		"整体":"_ZT",
		# 		"其他":"_QT"}
		# posD = {"正面":"_P",
		# 		"负面":"_N",
		# 		"中性":"_M"}
		posC = {"功效": "GX",
				"成分": "CF",
				"气味": "QW",
				  "包装": "BZ",
				  "新鲜度": "XXD",
				  "价格": "JG",
				  "其他": "QT",
				  "使用体验": "SYTY",
				  "整体": "ZT",
				  "尺寸": "CC",
				  "服务": "FW",
				  "物流": "WL",
				  "真伪": "ZW"}
		posD = {"正面": "ZM",
					"负面": "FM",
					"中性": "ZX"}
		fid = open(file, "w", encoding="utf-8")
		for i in self.dictionary.keys():
			sen = self.dictionary[i]["sen"]
			# sen = sen.lstrip()
			# sen =sen.rstrip()
			# sen = sen.replace("\"", "")
			sen = sen.replace(',', '，')
			sen = sen.replace('"', '')
			print(sen)
			print(i)
			lab = self.dictionary[i]["lab"]
			tag = ["U"]*len(sen)
			iterN = 0
			for l in lab:
				asp,asp_head,asp_tail,opi,opi_head,opi_tail,cla,pos = l
				if asp != "_":
					tag = tagger(tag, asp_head, asp_tail, posC, cla, "A")
				if opi != "_":
					tag = tagger(tag, opi_head, opi_tail, posD, pos, "O")
			idSet = [ids for ids in range(len(sen)) if sen[ids] == " "]
			sen = strQ2B(sen).replace(" ","")
			tag = [tag[i] for i in range(len(tag)) if i not in idSet]
			for l in range(len(sen)):
				fid.write(sen[l]+"\t"+tag[l]+"\n")
			fid.write("\n")
			
if __name__ == "__main__":

	# file1 = sys.argv[1]
	# file2 = sys.argv[2]
	# file3 = sys.argv[3]
	# test = '"少打了,多两节课",dsfdsf'
	# result = test.split(",")
	# test = test.replace(',', '，')
	# test = test.replace('"', '')
	# print(test)
	log = logger()
	file1 = "../data/concat_test_content.csv"
	file2 = "../data/concat_test_label.csv"
	file3 = "../data/concat_test.ner.v2"
	log.reader(file1, file2)
	log.parser(file3)
