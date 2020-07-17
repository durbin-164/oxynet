import parent_package
import numpy as np 
from torch.nn import Module, Conv2d, Linear, Flatten, CrossEntropyLoss
from torch.optim import SGD 
from torch import Tensor
import torch
import gzip
import time 
import torch.nn.functional as F
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
    return np.asarray(data, dtype=np.uint8)

data_size = 10000*28*28
x_train = _load_mnist(train_data, header_size=16)[:data_size,].reshape((-1,1, 28, 28)).astype(float)/255
x_test = _load_mnist(test_data, header_size=16).reshape((-1, 1,28, 28)).astype(float)/255
y_train = _load_mnist(train_label, header_size=8)[:10000]
y_test = _load_mnist(test_label, header_size=8).reshape((-1,1))


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
    Y_tilde = np.zeros((batch_size, num_classes))
    Y_tilde[np.arange(batch_size), Y] = 1
    return Y_tilde

def accuracy(pred, actual):
    pred_ = np.argmax(pred, axis=1)
    # actual_ = np.argmax(actual, axis=-1)
    # print("pd",pred_.shape)
    # print(actual.shape)

    match = (pred_ == actual).astype(int).sum()

    acc = match/len(pred)

    return acc

class Model(Module):
    def __init__(self,in_channel, out_channel):
        super().__init__()
        self.conv1 = Conv2d(in_channels=in_channel, out_channels= 4, kernel_size=(5,5), stride=2)
        self.fc1 = Linear(12*12*4, 64)
        self.fc2 = Linear(64,32)
        self.fc3 = Linear(32,out_channel)
        self.flat = Flatten()


    def forward(self, input):
        x1 = F.tanh(self.conv1(input))
        # x1 = input
        x2 = self.flat(x1)
        x3 = F.tanh(self.fc1(x2))
        x4 = F.tanh(self.fc2(x3))
        x5 = self.fc3(x4)

        return x5

device = torch.device("cuda:0")

model = Model(1, 10)
optimizer = SGD(model.parameters(), lr=0.01)
criterion = CrossEntropyLoss()
batch_size =64
out_class = 10

print(x_train.shape)
print(y_train.shape)

x_train_t = torch.Tensor(x_train)
y_train_t = torch.Tensor(y_train)
y_train_t = y_train_t.type(torch.LongTensor)
train = torch.utils.data.TensorDataset(x_train_t, y_train_t)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)


for epoch in range(100):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    
    for i, data in enumerate(train_loader, 0):
        
        inputs, actual = data
        inputs.to(device)
        actual.to(device)
        model.zero_grad()

        predicted = model(inputs)
        loss = criterion(predicted, actual)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_accuracy += accuracy(predicted.detach().numpy(), actual.detach().numpy())
        

    epoch_loss = epoch_loss/(len(train_loader))
    epoch_accuracy /= (len(train_loader))
    if(epoch % 10 == 0):
        print("Epoch : ",epoch, "  Loss: ",epoch_loss, " Acc: ", epoch_accuracy)
        
print("--- %s minutes ---" % ((time.time() - start_time)/60.0))