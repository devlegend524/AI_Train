# Example 2: Classfication

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
import random
import torch
torch.__version__

np.random.seed(0) # pick the seed for reproducibility - change it to explore the effects of random variations

n = 100
X, Y = make_classification(n_samples=n, n_features=2, n_redundant=0, n_informative=2, flip_y=0.1, class_sep=1.5)

X = X.astype(np.float32)
Y = Y.astype(np.int32)

split = [ 70*n//100, (15+70)*n//100 ]
train_x, valid_x, test_x = np.split(X, split)

train_labels, valid_labels, test_labels = np.split(Y, split)

def plot_dataset(features, labels, W=None, b=None):
    # prepare the plot
    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$X_i[0]$ -- (feature 1)')
    ax.set_ylabel('$x_i[1]$ -- (feature 2)')
    colors = ['r' if l else 'b' for l in labels]
    ax.scatter(features[:, 0], features[:, 1], marker='o', c=colors, s=100, alpha=0.5)
    if W is not None:
        min_x = min(features[:, 0])
        max_x = max(features[:, 1])
        min_y = min(features[:, 1]) * (1-.1)
        max_y = max(features[:, 1])*(1+.1)
        cx = np.array([min_x, max_x], dtype=np.float32)
        cy = (0.5-W[0]*cx-b)/W[1]
        ax.plot(cx, cy, 'g')
        ax.set_ylim(min_y, max_y)
    fig.show()

plot_dataset(train_x, train_labels)


class Network():
    def __init__(self):
        self.W = torch.randn(size=(2,1), requires_grad=True)
        self.b = torch.zeros(size=(1,), requires_grad=True)

    def forward(self, x):
        return torch.matmul(x, self.W) + self.b
    
    def zero_grad(self):
        self.W.data.zero_()
        self.b.data.zero_()

    def update(self, lr=0.1):
        self.W.data.sub_(lr*self.W.grad)
        self.b.data.sub_(lr*self.b)

net = Network()

def train_on_batch(net, x, y):
    z = net.forward(x).flatten()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(input=z, target=y)
    net.zero_grad()
    loss.backward()
    net.update()
    return loss

# Create a tf.data.Dataset object for easy batched iteration
dataset = torch.utils.data.TensorDataset(torch.tensor(train_x), torch.tensor(train_labels, dtype=torch.float32))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

list(dataloader)[0]


for epoch in range(15):
    for (x, y) in dataloader:
        loss = train_on_batch(net, x, y)
    print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

print(net.W, net.b)

plot_dataset(train_x, train_labels, net.W.detach().numpy(), net.b.detach().numpy())

pred = torch.sigmoid(net.forward(torch.tensor(valid_x)))
torch.mean(((pred.view(-1)>0.5) == (torch.tensor(valid_labels)>0.5)).type(torch.float32))