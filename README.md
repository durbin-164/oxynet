# oxynet

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_oxynet&metric=alert_status)](https://sonarcloud.io/dashboard?id=durbin-164_oxynet)
[![Coverage](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_oxynet&metric=coverage)](https://sonarcloud.io/dashboard?id=durbin-164_oxynet)
[![Bugs](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_oxynet&metric=bugs)](https://sonarcloud.io/dashboard?id=durbin-164_oxynet)
[![Code Smells](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_oxynet&metric=code_smells)](https://sonarcloud.io/dashboard?id=durbin-164_oxynet)
[![Lines of Code](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_oxynet&metric=ncloc)](https://sonarcloud.io/dashboard?id=durbin-164_oxynet)
[![Maintainability Rating](https://sonarcloud.io/api/project_badges/measure?project=durbin-164_oxynet&metric=sqale_rating)](https://sonarcloud.io/dashboard?id=durbin-164_oxynet)

Don't it feel amazing if you are a deep learning library creator like Tensorflow, Pytorch, etc?  

This is a learning purpose toy Deep Learning Framework from scratch to understand how automatic differentiation
work and how a deep learning framework like Pytorch, Tensorflow, etc, work. If you want to learn how modern deep learning frameworks work, hope this repository will help you a lot. 

For starting see **example** folder and **tests** folder to understand functionality.

## Dependency
Only **Numpy**

## Core Functionality
### Tensor
Tensor is the main auto differentiable multidimensional variable. It supports almost all frequently used function such that

### Supported Operation
1. add
2. sub
3. mul
4. div
5. matmul
6. pow
7. sum
8. slice
9. transpose

### Supported Math Operation
1. exp
2. log
3. max
4. he_initialization

### Optimizer
1. SGD

### Activation Function
1. softmax
2. tanh

### Loss Function
1. CrossEntropyLoss

### Module
1. Liner
2. Flatten
3. Conv2d

## Model Creation and Training Example

```python
import oxynet as onet 
import numpy as np 
from oxynet.modules import Module, Conv2d, Linear, Flatten, CrossEntropyLoss
from oxynet.optims import SGD 
from oxynet.modules import tanh
import gzip
from oxynet import Tensor

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

data_size = 1000*28*28
x_train = _load_mnist(train_data, header_size=16)[:data_size,].reshape((-1,1, 28, 28)).astype(float)/255
x_test = _load_mnist(test_data, header_size=16).reshape((-1, 1,28, 28)).astype(float)/255
y_train = _load_mnist(train_label, header_size=8)[:1000]
y_test = _load_mnist(test_label, header_size=8).reshape((-1,1))
print(y_train.shape)

def one_hot(Y, num_classes):
    batch_size = len(Y)
    Y_tilde = np.zeros((batch_size, num_classes))
    Y_tilde[np.arange(batch_size), Y] = 1
    return Y_tilde

def accuracy(pred, actual):
    pred_ = np.argmax(pred, axis=-1)
    actual_ = np.argmax(actual, axis=-1)
    match = (pred_ == actual_).astype(int).sum()
    acc = match/len(pred)
    return acc

#Model Definetion
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

# Create Model
model = Model(1, 10)
optimizer = SGD(lr=0.0001)
criterion = CrossEntropyLoss()
batch_size =64
out_class = 10

# Training Model
starts = np.arange(0, x_train.shape[0], batch_size)
for epoch in range(500):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    np.random.shuffle(starts)
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

```


## Supported Operation Example

#### Creation

```python
t1 = Tensor(10, requires_grad=True)
t2 = Tensor([1, 2, 3], requires_grad=True)
t3 = Tensor([[1, 2, 3],[4, 5, 6]], requires_grad=True)
```

### Operations

#### Addition

```python
t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
t2 = Tensor([[7, 8, 9]], requires_grad=True)
t3 = t1 + t2
assert t3.data.tolist() == [[8, 10, 12], [11, 13, 15]]
t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))
assert t1.grad.data.tolist() == [[1, 1, 1], [1, 1, 1]]
assert t2.grad.data.tolist() == [[2, 2, 2]]
```

#### Multiplication 

```python
t1 = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
t2 = Tensor([[7, 8, 9]], requires_grad=True)
t3 = t1 * t2
assert t3.data.tolist() == [[7,16, 27], [28, 40, 54]]
t3.backward(Tensor([[1, 1, 1], [1, 1, 1]]))
assert t1.grad.data.tolist() == [[7, 8, 9], [7, 8, 9]]
assert t2.grad.data.tolist() == [[5, 7, 9]]
```

##### Matmul

````python
# t1 is (3, 2)
t1 = Tensor([[1, 2], [3, 4], [5, 6]], requires_grad=True)

# t2 is a (2, 1)
t2 = Tensor([[10], [20]], requires_grad=True)
t3 = t1 @ t2
assert t3.data.tolist() == [[50], [110], [170]]
grad = Tensor([[-1], [-2], [-3]])
t3.backward(grad)

np.testing.assert_array_equal(t1.grad.data,
                              grad.data @ t2.data.T)
np.testing.assert_array_equal(t2.grad.data,
                              t1.data.T @ grad.data)
````

#### Div

```python
t1 = Tensor(10, requires_grad=True)
t2 = Tensor(20, requires_grad=True)
t3 = t2/t1
assert t3.data == 2. 
t3.backward()
assert t1.grad.data == 20* (-1./10**2)
assert t2.grad.data == 1./10
```

#### Sum

```python
t1 = Tensor([1,2,3], requires_grad=True)
t2 = t1.sum()
t2.backward(Tensor(3))
assert t1.grad.data.tolist() == [3,3,3]
```

#### Slice

```python
data = np.random.randn(10,10)
t1 = Tensor(data, requires_grad=True)
t2 = t1[2:5, 5:]
assert t2.shape == (3,5)
t2.backward(Tensor(1))
assert t1.grad.shape == (10,10)
```

## Supported Math Operation

#### exp

```python
t1 = onet.Tensor([1,2,3], requires_grad=True)
t2 = onet.exp(t1)
assert t2.data.tolist() == np.exp([1,2,3]).tolist()
t2.backward(onet.Tensor(1))
assert t1.grad.data.tolist() == np.exp([1,2,3]).tolist()
```

#### log
```python
t1 = onet.Tensor([1,2,3], requires_grad=True)
t2 = onet.log(t1)
assert t2.data.tolist() == np.log([1,2,3]).tolist()
t2.backward(onet.Tensor([10,10,12]))
assert t1.grad.data.tolist() == [10,5,4]
```

#### max

```python
t1 = onet.Tensor([[2,4,8,10],[3,15,4,5]], requires_grad=True)
t2 = onet.max(t1, keepdims=True)
assert t2.data == [[15]]
t2.backward(onet.Tensor([[20]]))
outdata = np.zeros((2,4))
outdata[1][1]=20
np.testing.assert_array_almost_equal(t1.grad.data, outdata)
```