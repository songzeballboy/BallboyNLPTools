#encoding=utf8

import gensim
import numpy as np
import time
import codecs

import sys
import os

from multiprocessing import cpu_count
from functools import reduce

import platform
import requests

import jieba

import DataUtil.util as datahelper
import DataUtil.Lookup as lookup

# class Seg(Core):
# 	def __init__(self):
# 		super(Seg,self).__init__()

class Jieba_end(object):
	def __init__(self):
		pass

	def get_sent_cut(self,sentDataset, if_cut_all = True):
		sent_cut_list = []
		for line in sentDataset:
			words = jieba.cut(line,cut_all = if_cut_all)
			cur_line = " ".join(words)
			sent_cut_list.append(cur_line)
		return sent_cut_list

	def cut_one_sent(self,sent,if_cut_all = True):
		words = jieba.cut(sent,cut_all=if_cut_all)
		ret_line = " ".join(words)
		return ret_line

class Seg(object):
	def __init__(self,backEnd = 'jieba'):
		self.backEnd = backEnd
		self.model = 'None'
		if self.backEnd == 'jieba':
			self.model = Jieba_end()
		else:
			print("The other backEnd is not ready current !!")
			return False
		print("Seg init finished, using the %s as the tokenizer backend" % backEnd)
		return True

	def load_stop_word():


	def init_backEnd(backEnd):
		self.__init__(backEnd)
		return True

	def cut(self,dataset,if_cut_all = True):
		return model.get_sent_cut(dataset,if_cut_all)

	def cut_line(self,dataset, if_cut_all = True):
		return model.cut_one_sent(dataset, if_cut_all)

class W2V(object):
	def __init__(self,dim=300):
		self.w2v_model = None
		self.dim = dim

	def get_the_model(self):
		return self.w2v_model

	def get_the_dim(self):
		return self.dim

	def load_word2vec_model(model_file):
		self.w2v_model = gensim.models.Word2Vec.load(model_file)
		return True

	def save_word2vec_model(model_file):
		if not self.w2v_model:
			print("Error : the model is not training yet")
			return False
		self.w2v_model.save(model_file)
		return True

	def save_embedding_result(filename):
		if not self.w2v_model:
			print("Error : the model is not training yet")
			return False
		word_list = self.w2v_model.wv.index2word
		embedding_matrix = self.w2v_model.wc.syn0

		ret = []
		assert(len(word_list) == embedding_matrix.shape[0])

		for i in range(len(word_list)):
			wd = word_list[i]
			embedding_vec = list(embedding_matrix[i])
			toWrite = str(wd)
			for cur in embedding_vec:
				toWrite = toWrite + " " + str(cur)
			ret.append(toWrite)

		datahelper.dump_data(filename,ret)
		return True

	def pre_the_sentence(self,corpus_filename):
		dataset = datahelper.read_data(corpus_filename)
		
		Seg_handle = Seg("jieba")
		dataset = Seg_handle.cut(dataset,if_cut_all=False)
		ret = []
		for line in dataset:
			ret.append(line.split(" "))
		return ret

	def traing_with_preprocess(self, dataset):
		print("begin to training the word2vec model")
		model = gensim.models.Word2Vec(dataset,min_count=3,size=self.dim)
		print("training the word2vec finished !!")
		return model

	def trainng_with_not_preprocess(self, corpus_filename, save_embedding_path = None, model_save_path = None):
		dataset = self.pre_the_sentence(corpus_filename)
		self.w2v_model = traing_with_preprocess(dataset)

		if not save_data_path:
			pass
		else:
			self.save_embedding_result(save_embedding_path)

		if not model_save_path:
			pass
		else:
			self.save_word2vec_model(model_save_path)
		return True