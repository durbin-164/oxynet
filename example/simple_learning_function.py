import parent_package
import numpy as np 

from oxynet import Tensor
from oxynet.layer import Parameter
from oxynet.model import Module
from oxynet.optim import SGD


x_data = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-1, +3, -2], dtype= np.float))

y_data = x_data @ coef + 5

class Model(Module):
    def __init__(self) -> None:
        self.w = Parameter(3)
        self.b = Parameter()

    def predict(self, inputs: Tensor) -> Tensor:
        return inputs @ self.w + self.b


model = Model()
optimizer = SGD(model.parameters, lr = 0.001)
batch_size = 32

for epoch in range(100):
    epoch_loss = 0.0

    for start in range(0, 100, batch_size):
        end = start+batch_size

        model.zero_grad()

        inputs = x_data[start:end]

        predicted = model.predict(inputs)

        actual = y_data[start:end]

        errors = predicted -actual

        loss = (errors * errors).sum()

        loss.backward()

        epoch_loss += loss.data  

        optimizer.step()

    print(epoch, epoch_loss)
