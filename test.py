import numpy as np
import collections
import csv

from keras.utils import np_utils
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.layers import Flatten, TimeDistributed
from keras.optimizers import Adam, RMSprop, SGD

num_words = 25000
max_length = 30

dictionary = dict()
with open('dictionary.txt', 'r', errors='ignore') as file:
	un_parse = file.readlines()
	for line in un_parse:
		line = line.strip('\n').split()
		dictionary[line[0]] = int(line[1])

model = load_model('../model/sentiment_classifier.h5')
while True:
	seq_input = input(">> Input you sentence: ")
	data = [0]*max_length
	word_count = 0
	for word in seq_input.split():
		if word_count>=max_length:
			break;
		if word in dictionary:
			data[word_count] = dictionary[word]
		else:
			data[word_count] = 0
		word_count += 1	
	print (data)
	prediction = model.predict( np.array([data]) )
	print (prediction)