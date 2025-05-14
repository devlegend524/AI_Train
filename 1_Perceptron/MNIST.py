import pylab
from matplotlib import gridspec
from sklearn.datasets import make_classification
import numpy as np
from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import pickle
import os
import gzip

np.random.seed(1)
import random

with gzip.open('../data/mnist.pkl.gz', 'rb') as mnist_pickle:
    MNIST = pickle.load(mnist_pickle, encoding='latin1')

print(MNIST[0][0][0][130:180])
print(MNIST[0][0][0])
features = MNIST[0][0].astype(np.float32) / 256.0
lables = MNIST[0][0]
fig = pylab.figure(figsize=(10, 5))
for i in range(10):
    ax = fig.add_subplot(1, 10, i+1)
    pylab.imshow(features[i].reshape(28, 28))
pylab.show()

def set_mnist_pos_neg(positive_label, negative_label):
    positive_indices = [i for i, j in enumerate(MNIST[0][0]) 
                            if j == positive_label]
    negative_indices = [i for i, j in enumerate(MNIST[0][0]) 
                            if j == negative_label]

    positive_images = MNIST[0][0][positive_indices]
    negative_images = MNIST[0][0][negative_indices]

    fig = pylab.figure()
    ax = fig.add_subplot(1, 2, 1)
    pylab.imshow(positive_images[0].reshape(28,28), cmap='gray', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax = fig.add_subplot(1,2,2)
    pylab.imshow(negative_images[0].reshape(28, 28), cmap='gray', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    return positive_images, negative_images

pos1, neg1 = set_mnist_pos_neg(1, 0)