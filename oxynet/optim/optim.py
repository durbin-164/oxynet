from oxynet.model import Module 
from oxynet.layer import Parameter
from typing import Callable, Iterator



class SGD:
    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr 

    def step(self, model: Module) -> None:
        for parameter in model.parameters():
            parameter -= parameter.grad * self.lr #type:ignore
