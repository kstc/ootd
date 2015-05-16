import pickle
import numpy as np

import matplotlib.pyplot as plt

from sklearn import cluster, datasets
from sklearn.preprocessing import scale

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

def create_kmeans(load, X=None, n=6):
	if not load and X and n is None:
		raise Exception("Number of clusters must be specified if kmeans model is not loaded")

	if load and isfile("womens.km"):
		print "Loading Kmeans model..."
		kmeans = pickle.load(open('womens.km', 'rb'))
	else:
		print "Performing K-means on dataset X..."
		kmeans = cluster.KMeans(n_clusters=n)
		kmeans.fit(X)

	return kmeans

def plot_kmeans(X, labels, num_clusters):
	colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow']
	for i in range(0, num_clusters):
		plt.scatter(X[labels==i][:, 0], X[labels==i][:, 1], color=colors[i])
	plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
	plt.ylabel('Type')
	plt.xlabel('Color')
	plt.show()

def save_clusters(data, labels, num_clusters):
	for i in range(0, num_clusters):
		np.savetxt('data_' + i + '.csv', data[labels==i], fmt='%s', delimiter=',')

raw_data, X = get_data("data_set_women.csv")

kmeans = create_kmeans(load=True)

plot_kmeans(X, kmeans.labels_, 6)

save_clusters(raw_data, kmeans.labels_, 6)




