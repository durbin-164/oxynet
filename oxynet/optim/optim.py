from oxynet.model import Module 
from oxynet.layer import Parameter
from typing import Callable, Iterator



class SGD:
    def __init__(self,parameters: Callable , lr: float = 0.001) -> None:
        self.lr = lr 
        self.parameters = parameters

    def step(self) -> None:
        for parameter in self.parameters():
            parameter -= parameter.grad * self.lr
