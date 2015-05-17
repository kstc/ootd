import pickle
import numpy as np

import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.preprocessing import scale
from scipy.spatial.distance import euclidean
from PIL import Image

from feature_extraction import get_dominant_color

from os.path import isfile

def file_exists(fname):
    return isfile(fname)

def get_data(filename):
	with open(filename, 'rb') as input_file:
		data = [i.strip().split(',') for i in input_file.readlines()]

	data = np.array(data)
	X = np.array(data[:, 1:], dtype=float)
	X[:, 0] = X[:, 0]/(256**3)
	X = scale(X)

	return data, X

def load_kmeans(filename):
	if not file_exists(filename):
		raise Exception(filename + " does not exist!")
	# print "Loading Kmeans model..."
	kmeans = pickle.load(open(filename, 'rb'))

	return kmeans

def train_kmeans(X, n, filename):
	# print "Performing K-means on dataset X..."
	kmeans = cluster.KMeans(n_clusters=n)
	kmeans.fit(X)

	pickle.dump(kmeans, open(filename, 'w+'))

	return kmeans

def plot_kmeans(X, labels, num_clusters):
	colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']
	for i in range(0, num_clusters):
		plt.scatter(X[labels==i][:, 0], X[labels==i][:, 1], color=colors[i])
	plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
	plt.ylabel('Type')
	plt.xlabel('Color')
	plt.show()

def save_clusters(data, labels, num_clusters, sex):
	for i in range(0, num_clusters):
		np.savetxt(sex + '_data_' + str(i) + '.csv', data[labels==i], fmt='%s', delimiter=',')

def get_similar(input_data, kmeans, dataset):
	# Get data and labels
	raw_data, X = get_data(dataset)
	labels = kmeans.labels_

	# Append new input to entire dataset and scale
	temp = raw_data[:, 1:]
	temp = np.vstack((temp, input_data))
	temp = np.array(temp, dtype=float)
	temp[:, 0] = temp[:, 0]/(256**3)
	temp = scale(temp)

	# Get data from scaled samples
	data = temp[-1]

	label = kmeans.predict(data)

	# Get scaled samples from cluster
	temp_2 = temp[labels==label]

	# Get raw samples from cluster
	cluster = raw_data[labels==label]

	distances = [(euclidean(data, temp_2[i]), cluster[i][0]) for i in range(len(cluster))]
	same = [d[1] for d in sorted(distances, key=lambda tup: tup[0])]
	return same[:7]


# raw_data, X = get_data("data_set_women.csv")

# kmeans = load_kmeans("womens.km")
# # kmeans = train_kmeans(X, 6, 'womens.km')

# # # plot_kmeans(X, kmeans.labels_, 6)

# # save_clusters(raw_data, kmeans.labels_, 6, 'women')

# color = get_dominant_color(Image.open('gw2.jpg'))
# clothing_type = '3'

# get_similar(raw_data, kmeans.labels_, [color, clothing_type], kmeans)




