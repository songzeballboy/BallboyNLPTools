#encoding=utf8

import sys
import os
import numpy as np
import random

import codecs
from collections import Counter

import math

def read_data(filename):
	ret = []
	with codecs.open(filename,"r",encoding="utf8",errors='ignore') as f:
		while 1:
			line = f.readline()
			if not line :
				break
			line.replace("\n","").replace("\r","").rstrip()
			ret.append(line)
	return ret

def read_data_wtih_limit(filename,number_to_load = 100000):
	ret = []
	with codecs.open(filename,"r",encoding="utf8",errors='ignore') as f:
		line_cnt = 0
		while line_cnt < number_to_load:
			line = f.readline()
			if not line:
				break
			line = line.replace("\n","").replace("\r","").rstrip()
			ret.append(line)
	return ret

def dump_data(filename, dataset):
	with codecs.open(filename,"w",encoding="utf8",errors='ignore') as f:
		for line in dataset:
			f.write(line + "\n")
	return True

def numpy_vector_padding(vec,dim,padding_x):
	if len(vec) > dim:
		return vec[ : dim]
	elif len(vec) < dim:
		return np.append(vec, [padding_x] * (dim - len(vec)))
	else:
		return vec

def sub_sample_data(dataset, sample_size):
	sub_dataset = []
	random_list = random.sample(range(len(dataset)), sample_size)
	for _ in random_list:
		sub_dataset.append(dataset[_])
	return sub_dataset

def random_shuffle(dataset):
	indices = np.random.permutation(np.arange(dataset.shape[0]))
	dataset = dataset[indices]
	return dataset

def index_to_one_hot(index, dim):
	one_hot = [0] * dim
	one_hot[index] = 1
	return one_hot

def cal_norm(vec):
	return math.sqrt(np.dot(vec,vec))

def cos_sim(vec1, vec2):
	if vec1.shape[0] != vec2.shape[0]:
		print("Error: the two vector is not same length")
		raise "cal the cos_sim_error"

	xx = np.dot(vec1, vec2)
	yy1 = cal_norm(vec1)
	yy2 = cal_norm(vec2)
	if xx == 0.0:
		xx = 0.1
	if yy1 == 0.0:
		yy1 = 0.1
	if yy2 == 0.0:
		yy2 = 0.1

	return xx / (yy1 * yy2)
