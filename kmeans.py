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

def create_kmeans(X=None, load=True):
	if isfile("womens.km") and X is None:
		kmeans = pickle.load(open('womens.km', 'rb'))
	else:
		kmeans = cluster.KMeans(n_clusters=6)
		kmeans.fit(X)

	return kmeans

def plot_kmeans(X, labels):
	plt.scatter(X[labels==0][:, 0], X[labels==0][:, 1], color='r')
	plt.scatter(X[labels==1][:, 0], X[labels==1][:, 1], color='b')
	plt.scatter(X[labels==2][:, 0], X[labels==2][:, 1], color='g')
	plt.scatter(X[labels==3][:, 0], X[labels==3][:, 1], color='purple')
	plt.scatter(X[labels==4][:, 0], X[labels==4][:, 1], color='orange')
	plt.scatter(X[labels==5][:, 0], X[labels==5][:, 1], color='y')
	plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='black')
	plt.ylabel('Type')
	plt.xlabel('Color')
	plt.show()

def save_clusters(data, labels):
	np.savetxt('data_0.csv', data[labels==0], fmt='%s', delimiter=',')
	np.savetxt('data_1.csv', data[labels==1], fmt='%s', delimiter=',')
	np.savetxt('data_2.csv', data[labels==2], fmt='%s', delimiter=',')
	np.savetxt('data_3.csv', data[labels==3], fmt='%s', delimiter=',')
	np.savetxt('data_4.csv', data[labels==4], fmt='%s', delimiter=',')
	np.savetxt('data_5.csv', data[labels==5], fmt='%s', delimiter=',')

raw_data, X = get_data("data_set_women.csv")

kmeans = create_kmeans(None)

plot_kmeans(X, kmeans.labels_)

save_clusters(raw_data, kmeans.labels_)




