"""
################################bert_bilstm_model.py################################
程序名称:     bert_bilstm_model.py
功能描述:     BERT预训练模型+双向长短时记忆网络
创建人名:     wuxinhui
创建日期:     2019-08-03
版本说明:     v1.0
################################bert_bilstm_model.py################################
"""

import numpy as np
import os, sys, datetime, pickle
import pickle
import keras
import jieba.posseg as psg
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from keras.utils import plot_model
from keras.engine.topology import Layer
from keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import Tokenizer
from keras_bert.layers import TokenEmbedding
from keras_pos_embd import PositionEmbedding
from keras_layer_normalization import LayerNormalization
from keras_multi_head import MultiHeadAttention
from keras import regularizers
from keras import backend as K

class bert_bilstm_model(object):

	"""docstring for bert_bilstm_model"""

	def __init__(self, args):
		super(bert_bilstm_model, self).__init__()
		self.embeds_dim = args.embeds_dim
		self.embeds_dir = args.embeds_dir
		self.dict_dir = args.dict_dir
		self.pretrain_model_dir = args.pretrain_model_dir
		self.layerid = args.layerid
		self.max_seq_len = args.max_seq_len
		self.batch_size = args.batch_size
		self.epochs = args.epochs
		self.saved_models_dir = args.saved_models_dir
		self.layer_dict = [7, 15, 23, 31, 39, 47, 55, 63, 71, 79, 87, 95, 103]
		self._loader_dict_()
		self.n_classes = len(self._tag)
		self.model = self._model_compile_()

	def _loader_dict_(self):
		self.vocab_dict = {}
		with open(os.path.join(self.pretrain_model_dir,"vocab.txt"),"r",encoding="utf-8") as fid:
			for line in fid.readlines():
				self.vocab_dict[line.strip()] = len(self.vocab_dict)
		tag_dict = pickle.load(open(os.path.join(self.dict_dir, "cla_tag_dict.pl"), 'rb'))
		self._tag, self._id2tag, self._tag2id = tag_dict
		return
		
	def _model_compile_(self):
		layerN = 12
		bert_model = load_trained_model_from_checkpoint(
			os.path.join(self.pretrain_model_dir, "bert_config.json"),
			os.path.join(self.pretrain_model_dir, "bert_model.ckpt"),
			seq_len=None
		)

		for l in bert_model.layers:
			l.trainable = True
		
		x = Lambda(lambda x: x[:, 0])(bert_model.output)
		prob = Dense(self.n_classes, activation='softmax')(x)
		model = Model(inputs=bert_model.inputs, outputs=prob)
		model.summary()
		model.compile(
					optimizer=Adam(lr=0.001),
					loss='categorical_crossentropy',
					metrics =['accuracy']
			)
		plot_model(model, to_file=os.path.join(self.saved_models_dir,'bert_bilstm_model.png'), show_shapes=True)
		return model
		
	def _model_train_(self, train_text):
		X, y = train_text
		input_ids, input_mask, input_type = self._text_process(X)
		trainy = self._label_encoder(y)
		callbacks =[
				ReduceLROnPlateau(),
				ModelCheckpoint(filepath=os.path.join(self.saved_models_dir, "bert_bilstm_model.h5"), \
								save_best_only=True)
			]
		self.model.fit(x=[input_ids,input_mask], 
					y=trainy, 
					batch_size=self.batch_size, 
					epochs=self.epochs, 
					verbose=1, 
					callbacks=callbacks,
					validation_split=0.1,  
					shuffle=True)
		return
		
	def _model_test_(self, test_text):
		X, y = test_text
		input_ids, input_mask, input_type = self._text_process(X)
		testy = self._label_encoder(y)
		self.model.load_weights(os.path.join(self.saved_models_dir, "bert_bilstm_model.h5"))
		score = self.model.evaluate(x=[input_ids,input_mask], y=testy, verbose=1)
		print("the test loss for model: %s" %score[0])
		print("the test accuracy for model: %s" %score[1])
		return
		
	def _model_predict_(self, talk):
		input_ids, input_mask, input_type = self._text_process([talk])
		self.model.load_weights(os.path.join(self.saved_models_dir, "bert_bilstm_model.h5"))
		prob = self.model.predict([input_ids,input_mask])[0]
		label = self._id2tag[np.argmax(prob)]
		return label

	def _model_predict_batch_(self, batch):
		input_ids, input_mask, input_type = self._text_process(batch)
		self.model.load_weights(os.path.join(self.saved_models_dir, "bert_bilstm_model.h5"))
		tag = self.model.predict([input_ids,input_mask])
		label = [self._id2tag[i] for i in self._label_decoder(tag)]
		return label
		
	def _text_process(self, text):
		Tokener = Tokenizer(self.vocab_dict)
		encoder = [Tokener.encode(first=doc[0],second=doc[1], max_len=self.max_seq_len) for doc in text]
		input_ids = [i[0] for i in encoder]
		input_type = [i[1] for i in encoder]
		input_mask = [[0 if l==0 else 1 for l in i] for i in input_ids]
		return (input_ids,input_mask,input_type)
		
	def _char2id_func(self, senl):
		return [self._char2id.get(s,1) for s in senl]
		
	def _id2char_func(self, ids):
		return [self._id2char[i] for i in ids]
		
	def _label_decoder(self, tag):
		return [np.argmax(ids) for ids in tag]

	def _label_encoder(self, labels):
		results = np.zeros((len(labels), self.n_classes))
		for i, label in enumerate(labels):
			results[i][label] = 1
		return results
