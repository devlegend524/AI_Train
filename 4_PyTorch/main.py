import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
import random
import torch
import torch.version
torch.__version__


np.random.seed(13) # pick the seed for erproducibility -  change it to explore the effects of random variations

train_x = np.linspace(0, 3, 120)
train_labels = 2 * train_x + 0.9 + np.random.randn(*train_x.shape) * 0.5

plt.scatter(train_x, train_labels)


input_dim = 1
output_dim = 1
learning_rate = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.is_available())
print(torch.version.cuda)
print('Doing computations on ' + device)


# Initialize weights and bias
w = torch.tensor([0.0], requires_grad=True, dtype=torch.float32, device=device)  # Changed initial value from [100,0] to [0.0]
b = torch.tensor(0.0, requires_grad=True, device=device)    # Simplified bias initialization

def f(x):
    return torch.matmul(x, w) + b

def compute_loss(labels, predictions):
    return torch.mean(torch.square(labels - predictions))

def train_on_batch(x, y):
    predictions = f(x)
    loss = compute_loss(y, predictions)
    loss.backward()
    # with torch.no_grad():  # Ensure we don't track these operations in the computation graph
    w.data.sub_(learning_rate * w.grad)
    b.data.sub_(learning_rate * b.grad)
    w.grad.zero_()
    b.grad.zero_()
    return loss

# Shuffle the data.
indices = np.random.permutation(len(train_x))
features = torch.tensor(train_x[indices], dtype=torch.float32).view(-1, 1)
labels = torch.tensor(train_labels[indices], dtype=torch.float32)


batch_size = 4
for epoch in range(10):
    for i in range(0, len(features), batch_size):
        loss = train_on_batch(features[i:i+batch_size].view(-1, 1), labels[i:i+batch_size].to(device))
    print('Epoch %d: last batch loss = %.4f' % (epoch, float(loss)))

plt.scatter(train_x, train_labels)
x = np.array([min(train_x), max(train_x)])
with torch.no_grad():
    y = w.numpy()*x + b.numpy()

plt.plot(x,y, color='red')

