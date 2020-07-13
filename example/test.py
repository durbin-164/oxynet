import parent_package
from typing import List

import numpy as np
import inspect

from oxynet import Tensor
from oxynet.modules import Parameter,tanh, Module
from oxynet.optims import SGD

class FizzBuzzModel(Module):
    def __init__(self, num_hidden: int = 50) -> None:
        self.w1 = Parameter(10, num_hidden)
        self.b1 = Parameter(num_hidden)

        self.w2 = Parameter(num_hidden, 4)
        self.b2 = Parameter(4)

    def forward(self, inputs: Tensor) -> Tensor:
        # inputs will be (batch_size, 10)
        x1 = inputs @ self.w1 + self.b1  # (batch_size, num_hidden)
        x2 = tanh(x1)                    # (batch_size, num_hidden)
        x3 = x2 @ self.w2 + self.b2      # (batch_size, 4)

        return x3


model = FizzBuzzModel()

for name, value in inspect.getmembers(model):
    if isinstance(value, Parameter):
        # for name, value in inspect.getmembers(value):
            print(name)