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

n = 50
X,Y = make_classification(n_samples = n, n_features = 2, n_redundant=0, n_informative=2, flip_y=0)

Y = Y*2 - 1 # convert initial 0/1 values into -1/1
X = X.astype(np.float32); Y = Y.astype(np.int32) # features - float, label - int

# Split the dataset into training and test
train_x, test_x = np.split(X, [ n*8//10])
train_labels, test_labels = np.split(Y, [n*8//10])
print("Features:\n", train_x[0:4])
print("Labels: \n", train_labels[0:4])

def plot_dataset(suptitle, features, labels):
    # prepare the plot
    fig, ax = pylab.subplots(1,1)
    # pylab.subplots_adjust(bottom=0.2, wspace=0.4)
    fig.suptitle(suptitle, fontsize = 16)
    ax.set_xlabel('$x_i[0]$ -- (feature 1)')
    ax.set_ylabel('$x_i[1]$ -- (feature 2)')

    colors = ['r' if l > 0 else 'b' for l in labels]
    ax.scatter(features[:, 0], features[:, 1], marker='o', c=colors, s=100, alpha=0.5)
    fig.show()

plot_dataset('Training data', train_x, train_labels)


pos_examples = np.array([ [t[0], t[1], 1] for i, t in enumerate(train_x) if train_labels[i] > 0])
neg_examples = np.array([ [t[0], t[1], 1] for i,t in enumerate(train_x) if train_labels[i] < 0])

print(pos_examples[0:3])


def train(positive_examples, negative_examples, num_interations = 200):
    num_dims = positive_examples.shape[1]


    # Initialize weights
    # We initialize with 0 for simplicity, but random initialization is also a good idea

    weights = np.zeros((num_dims, 1))

    pos_count = positive_examples.shape[0]
    neg_count = negative_examples.shape[0]

    report_frequency = 10

    for i in range(num_interations):
        # Pick one positive and one negative example
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)

        z = np.dot(pos, weights)
        if z < 0: # positive example was classified as negative
            weights = weights + pos.reshape(weights.shape)

        z = np.dot(neg, weights)
        if z >= 0: # negative example was classified as positive
            weights = weights - neg.reshape(weights.shape)

        # Periodically, print out the current accuracy on all examples
        if i % report_frequency == 0:
            pos_out = np.dot(positive_examples, weights)
            neg_out = np.dot(negative_examples, weights)
            pos_correct = (pos_out >= 0).sum() / float(pos_count)
            neg_correct = (neg_out < 0).sum() / float(neg_count)
            print("Iteration={}, pos correct={}, neg ccorrect={}".format(i, pos_correct, neg_correct))

    return weights

wts = train(pos_examples, neg_examples)
print(wts.transpose())

def plot_boundary(positive_examples, negative_examples, weights):
    if np.isclose(weights[1], 0):
        if np.isclose(weights[0], 0):
            x = y = np.array([-6, 6], dtype = 'float32')
        else:
            y = np.array([-6, 6], dtype='float32')
            x = -(weights[1] * y + weights[2])/weights[0]
    else:
        x = np.array([-6, 6], dtype='float32')
        y = -(weights[0] * x + weights[2])/weights[1]
    
    pylab.xlim(-6, 6)
    pylab.ylim(-6, 6)
    pylab.plot(positive_examples[:, 0], positive_examples[:, 1], 'bo')
    pylab.plot(negative_examples[:, 0], negative_examples[:, 1], 'ro')
    pylab.plot(x, y, 'g', linewidth=2.0)
    pylab.show()

plot_boundary(pos_examples, neg_examples, wts)


def accuracy(weights, test_x, test_labels):
    res = np.dot(np.c_[test_x, np.ones(len(test_x))], weights)
    return (res.reshape(test_labels.shape)*test_labels >=0).sum() / float(len(test_labels))

accuracy(wts, test_x, test_labels)

def train_graph(positive_examples, negative_examples, num_iterations = 100):
    num_dims = positive_examples.shape[1]
    weights = np.zeros((num_dims, 1)) # initialize weights

    pos_count = positive_examples.shape[0]
    neg_count = negative_examples.shape[0]

    report_frequency = 15
    snapshots = []

    for i in range(num_iterations):
        pos = random.choice(positive_examples)
        neg = random.choice(negative_examples)

        z = np.dot(pos, weights)
        if z < 0:
            weights = weights + pos.reshape(weights.shape)

        z = np.dot(neg, weights)
        if z > 0:
            weights = weights - neg.reshape(weights.shape)

        if i % report_frequency == 0:
            pos_out = np.dot(positive_examples, weights)
            neg_out = np.dot(negative_examples, weights)
            pos_correct = (pos_out >= 0).sum() / float(pos_count)
            neg_correct = (neg_out < 0).sum() / float(neg_count)
            # make correction a list so it is homogeneous to weights list then numpy array accepts

            snapshots.append((np.concatenate(weights), [(pos_correct+neg_correct)/2.0,0,0]))

    return np.array(snapshots)

snapshots = train_graph(pos_examples, neg_examples)

def plotit(pos_examples, neg_examples, snapshots, step):
    fig = pylab.figure(figsize=(10, 4))
    fig.add_subplot(1, 2, 1)
    plot_boundary(pos_examples, neg_examples, snapshots[step][0])
    fig.add_subplot(1,2,2)
    pylab.plot(np.arange(len(snapshots[:,1])), snapshots[:, 1])
    pylab.ylabel('Accuracy')
    pylab.xlabel('Iteration')
    pylab.plot(step, snapshots[step, 1][0], 'bo')
    pylab.show()

def pl1(step): plotit(pos_examples, neg_examples, snapshots, step)

interact(pl1, step=widgets.IntSlider(value=0, min=0, max=len(snapshots) - 1))

