
"""
################################cla_sample_build.py################################
程序名称:     cla_sample_build.py.py
功能描述:     分类样本构建
创建人名:     wuxinhui
创建日期:     2019-08-08
版本说明:     v1.0
################################cla_sample_build.py################################
"""
import sys

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
		fid = open(file, "w", encoding="utf-8")
		for i in self.dictionary.keys():
			sen = self.dictionary[i]["sen"]
			lab = self.dictionary[i]["lab"]
			for l in lab:
				asp,asp_head,asp_tail,opi,opi_head,opi_tail,cla,pos = l
				label = posC[cla]+posD[pos]
				sen1 = asp+opi
				sen1 = sen1.strip("_")
				sen2 = sen
				fid.write(label+"\t"+sen1+"@@"+sen2+"\n")
			
if __name__ == "__main__":

	file1 = sys.argv[1]
	file2 = sys.argv[2]
	file3 = sys.argv[3]
	log = logger()
	log.reader(file1, file2)
	log.parser(file3)
