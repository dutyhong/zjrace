"""
################################Config.py################################
程序名称:     Config.py
功能描述:     配置文件
创建人名:     wuxinhui
创建日期:     2019-08-15
版本说明:     v1.0
################################Config.py################################
"""

class Config(object):

	"""docstring for Config"""

	def __init__(self):
		super(Config, self).__init__()
		self.ner_train_data = "./data/train.ner.pl"
		self.ner_test_data = "./data/test.ner.pl"
		self.cla_train_data = "./data/train.cla.pl"
		self.cla_test_data = "./data/test.cla.pl"
		self.if_train_ner = False
		self.if_train_cla = False
		self.dict_dir = "./saved_models"
		self.saved_models_dir = "./saved_models"
		self.pretrain_model_dir = "./pretrain_model/chinese_wwm_ext_L-12_H-768_A-12"
		self.layerid = 12
		self.ner_layerid = 12
		self.cla_layerid = 12
		self.embeds_dir = None
		self.embeds_dim = 64
		self.ner_embeds_dim = 64
		self.cla_embeds_dim = 128
		self.batch_size = 32
		self.epochs = 50
		self.max_seq_len = 100
		
if __name__ == "__main__":
	Config()



