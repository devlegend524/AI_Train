# Multi-Layered Perceptrons

import  matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.datasets import make_classification
import numpy as np
# pick the seed for reproducibility - change it to explore the effects of random variations
np.random.seed(0)
import random

n = 100
X,Y = make_classification(n_samples= n, n_features=2, n_redundant=0, n_informative=2, flip_y=0.2)

X = X.astype(np.float32)
Y = Y.astype(np.int32)

# Split into train and test dataset

train_x, test_x = np.split(X, [n*8//10])
train_labels, test_labels = np.split(Y, [n*8//10])

def plot_dataset(suptitle, features, labels):
    # prepare the plot
    fig, ax = plt.subplots(1,1)
    #pylab.subplots_adjust(bottom=0.2, wspace=0.4)
    fig.suptitle(suptitle, fontsize = 16)
    ax.set_xlabel('$X_i[0]$ -- (feature 1)')
    ax.set_ylabel('$X_i[1]$ -- (feature 2)')

    colors = ['r' if l else 'b' for l in labels]
    ax.scatter(features[:, 0], features[:, 1], marker='o', c=colors, s=100, alpha=0.5)
    fig.show()

plot_dataset('Scatterplot of the training data', train_x, train_labels)
plt.show()

print(train_x[:5])
print(train_labels[:5])


# helper function for plotting various loss functions
def plot_loss_functions(suptitle, functions, ylabels, xlabel):
    fig, ax = plt.subplots(1, len(functions), figsize=(9, 3))
    plt.subplots_adjust(bottom=0.2, wspace=0.4)
    fig.suptitle(suptitle)
    for i, fun in enumerate(functions):
        ax[i].set_xlabel(xlabel)
        if len(ylabels) > i:
            ax[i].set_ylabel(ylabels[i])
        ax[i].plot(x, fun)
    plt.show()

x = np.linspace(-2, 2, 101)

# zero one loss function
def zero_one(d):
    if d.any() < 0.5:
        return 0
    return 1

# logistic loss function
def logistic_loss(fx):
    # assumes y == 1
    return -np.log(fx)

plot_loss_functions(
    suptitle="Common loss functions for regression",
    functions = [np.abs(x), np.power(x, 2)],
    ylabels=['$\mathcal{L}_{abs}$ (absolute loss)', '$\mathcal{L}_{sq}$ (squared loss)'],
    xlabel= '$y - f(x_i)$'
)