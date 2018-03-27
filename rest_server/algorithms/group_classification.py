import pickle
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from gensim.models import KeyedVectors
import numpy as np
import os
import sys


def initialization():
	current_directory = os.getcwd() + '\\algorithms'
	classification_filename = current_directory + '\\classification_dictionary.pickle'
	with open(classification_filename, 'rb') as f:
		global classification 
		classification = pickle.load(f)
	global model
	model = KeyedVectors.load_word2vec_format(current_directory + '\\GoogleNews-vectors-negative300.bin', binary=True)
	with open(current_directory + '\\trained_pca.pickle', "rb") as f:
		global trained_pca
		trained_pca = pickle.load(f)
	with open(current_directory + '\\trained_kmean.pickle', "rb") as f:
		global trained_kmean
		trained_kmean = pickle.load(f)



def topic_classification(topic):
	print('topic_classification')
	if topic not in classification:
		group = classify_new_topic(topic)
		print("{0} in group {1}".format(topic, group))
	else:
		group = classification[topic]
		print("{0} in group {1}".format(topic, group))

	return group


def classify_new_topic(topic):
	print('classify_new_topic')
	topic_vector_temp = model[topic]
	topic_vector = np.empty([1,300])
	topic_vector[0,]=topic_vector_temp
	topic_vector = trained_pca.transform(topic_vector)
	group = trained_kmean.predict(topic_vector)[0]
	print("{0} in group {1}".format(topic, group))
	return group

