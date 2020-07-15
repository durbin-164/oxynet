'''
http://yann.lecun.com/exdb/mnist/
'''

import parent_package
import oxynet as onet 
import cupy as cp 
import numpy as np
from oxynet.modules import Module, Conv2d, Linear, Flatten, CrossEntropyLoss
from oxynet.optims import SGD 
from oxynet.modules import tanh
import gzip
from oxynet import Tensor
import time 

start_time = time.time()

root_dir = ".datasets/MNIST/"

train_data='train-images-idx3-ubyte.gz'
train_label='train-labels-idx1-ubyte.gz'
test_data='t10k-images-idx3-ubyte.gz'
test_label='t10k-labels-idx1-ubyte.gz'

def _load_mnist( path, header_size):
    path = root_dir + path
    with gzip.open(path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=header_size)
    return cp.array(data, dtype=cp.uint8)
base_size=10000
data_size = base_size*28*28
x_train = _load_mnist(train_data, header_size=16)[:data_size,].reshape((-1,1, 28, 28)).astype(float)/255
# x_test = _load_mnist(test_data, header_size=16).reshape((-1, 1,28, 28)).astype(float)/255
y_train = _load_mnist(train_label, header_size=8)[:base_size]
# y_test = _load_mnist(test_label, header_size=8).reshape((-1,1))

print(y_train.shape)
def one_hot(Y, num_classes):
    r"""Perform one-hot encoding on input Y.

    .. math::

        \text{Y'}_{i, j} =
                    \begin{cases}
                      1, &\quad if \quad Y_i = 0 \\
                      0, &\quad else
                    \end{cases}

    Args:
        Y (Tensor): 1D tensor of classes indices of length :math:`N`
        num_classes (int): number of classes :math:`c`

    Returns:
        Tensor: one hot encoded tensor of shape :math:`(N, c)`
    """
    batch_size = len(Y)
    Y_tilde = cp.zeros((batch_size, num_classes))
    Y_tilde[cp.arange(batch_size), Y] = 1
    return Y_tilde

def accuracy(pred, actual):
    pred_ = cp.argmax(pred, axis=-1)
    actual_ = cp.argmax(actual, axis=-1)
    # print(pred.shape)
    # print(actual.shape)

    match = (pred_ == actual_).astype(int).sum()

    acc = match/len(pred)

    return acc

class Model(Module):
    def __init__(self,in_channel, out_channel):
        self.conv1 = Conv2d(in_channels=in_channel, out_channels= 4, kernel_size=(5,5), stride=2)
        self.fc1 = Linear(12*12*4, 64)
        self.fc2 = Linear(64,32)
        self.fc3 = Linear(32,out_channel)
        self.flat = Flatten()


    def forward(self, input):
        x1 = tanh(self.conv1(input))
        # x1 = input
        x2 = self.flat(x1)
        x3 = tanh(self.fc1(x2))
        x4 = tanh(self.fc2(x3))
        x5 = self.fc3(x4)

        return x5

model = Model(1, 10)
optimizer = SGD(lr=0.0001)
criterion = CrossEntropyLoss()
batch_size =256
out_class = 10


starts = cp.arange(0, x_train.shape[0], batch_size)
for epoch in range(500):
    epoch_loss = 0.0
    epoch_accuracy = 0.0

    cp.random.shuffle(starts)
    for start in starts:
        end = start + batch_size

        model.zero_grad()

        inputs = Tensor(x_train[start:end], requires_grad = True)
        actual = Tensor(one_hot(y_train[start:end],out_class), requires_grad = True)
        
        predicted = model(inputs)
        
        
        loss = criterion(predicted, actual)

        loss.backward()
        optimizer.step(model)

        epoch_loss += loss.data
        epoch_accuracy += accuracy(predicted.data, actual.data)
        

    epoch_loss = epoch_loss/(len(starts))
    epoch_accuracy /= (len(starts))
    if(epoch % 10 == 0):
        print("Epoch : ",epoch, "  Loss: ",epoch_loss, " Acc: ", epoch_accuracy)
        


print("--- %s minutes ---" % ((time.time() - start_time)/60.0))