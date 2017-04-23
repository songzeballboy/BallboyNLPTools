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

from DataUtil.util import read_data
from DataUtil.util import read_data_wtih_limit

def load_embedding(filename,number_to_load = 100000, embedding_sz = 300):
	print("Begin to load the embedding")
	time0 = time.time()

	word2index = {}
	embedding_matrix = []
	index = 1
	embedding_matrix.append([0.0] * embedding_sz)

	dataset = read_data_wtih_limit(filename,number_to_load)
	for line in dataset:
		tokens = line.spilt("\t")
		word = tokens[0]
		word_embedding = list(map(float,tokens[1:]))
		word2index[word] = index

		index += 1
		embedding_matrix.append(word_embedding)
	
	time1 = time.time()
	print("Runnig time is == %.3f" % (time1 - time0))
	print("Finish load the embedding")
	return embedding_matrix, word2index

def load_pattern_embedding(filename,number_to_load = 100000, embedding_sz = 300):
	pattern_dict = {
		"what":u"什么",
		"degree":u"程度",
		"how":u"怎样",
		"why":u"为什么",
		"disease":u"疾病",
		"symptom":u"症状",
		"where":u"哪里",
		"treat":u"治疗",
		"whether":u"是否",
		"could":u"可否",
		"when":u"何时",
		"surgery":u"手术",
		"examination":u"检查",
		"location" : u"北京",
		"medicine" : u"中药"
	}

	print("Begin to load the embedding")
	time0 = time.time()

	word2index = {}
	embedding_matrix = []
	index = 1
	embedding_matrix.append([0.0] * embedding_sz)
	
	dataset = read_data_wtih_limit(filename,number_to_load)
	for line in dataset:
		tokens = line.split("\t")
		word = tokens[0]
		word_embedding = list(map(float,tokens[1:]))
		word2index[word] = index

		index += 1
		embedding_matrix.append(word_embedding)

	for key,value in pattern_dict.items():
		if value not in word2index:
			print("Mismatch the word is %s" % key)
			continue
		word_embedding = embedding_matrix[word2index[value]][:]
		word2index[key] = index
		index += 1
		embedding_matrix.append(word_embedding)

	time1 = time.time()
	print("Running time is == %.3f" % (time1 - time0))
	print("Finish load the embedding")
	return embedding_matrix,word2index

def training_the_word2vec_embedding(corpus_filename, model_save_file):
	dataset = get_dataset()
	w2v_model =  gensim.models.Word2Vec(dataset, min_count=20, window=10, workers=3)
	w2v_model.save(model_save_file)
	return True

def dump_word2vec_embedding(w2v_model_file,out_file):
	model = gensim.models.Word2Vec.load(w2v_model_file)
	## get the gensim word2vec model information into the counter

