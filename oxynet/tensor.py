from __future__ import annotations
from typing import List, NamedTuple, Callable, Optional, Union, Tuple

import numpy as np 
import oxynet.tensor_ops as ops

class Dependency(NamedTuple):
    tensor: 'Tensor'
    grad_fn: Callable[[np.ndarray], np.ndarray]


Arrayable = Union[float, list, np.ndarray]

def ensure_array(arrayable: Arrayable) -> np.ndarray:
    if isinstance(arrayable, np.ndarray):
        return arrayable
    else:
        return np.array(arrayable)

Tensorable = Union['Tensor', list , np.ndarray]

def ensure_tensor(tensorable: Tensorable) -> 'Tensor':
    if isinstance(tensorable, Tensor):
        return tensorable
    else:
        return Tensor(tensorable)


class Tensor:
    def __init__(self,
                data: Arrayable,
                requires_grad: bool = False,
                depends_on: List[Dependency] = None) -> None:

        self._data = ensure_array(data)
        self.requires_grad = requires_grad
        self.depends_on = depends_on or []
        self.grad: Optional['Tensor'] = None

        if self.requires_grad:
            self.zero_grad()
    
    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @data.setter
    def data(self, new_data: np.ndarray) -> None:
        self._data = new_data
        # Setting the data manually means we invalidate the gradient.
        # self.grad = None

    @property
    def shape(self) -> Tuple:
        return self._data.shape
        
    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def __repr__(self) -> str:
        return f"Tensor ({self.data}, requires_grad=({self.requires_grad}))"  

    def __add__(self, other) -> 'Tensor':
        """gets called if I do t + other"""
        return ops._add(self, ensure_tensor(other))
    
    def __radd__(self, other) -> 'Tensor':
        """gets called if I do other + t"""
        return ops._add(ensure_tensor(other), self)
    
    def __mul__(self, other) -> 'Tensor':
        return ops._mul(self, ensure_tensor(other))

    def __rmul__(self, other) -> 'Tensor':
        return ops._mul(ensure_tensor(other), self)
        
    def __neg__(self) -> 'Tensor':
        return ops._neg(self)
    
    def __sub__(self, other) -> 'Tensor':
        return ops._sub(self, ensure_tensor(other))
        
    def __rsub__(self, other) -> 'Tensor':
        return ops._sub( ensure_tensor(other), self)

    
    def __iadd__(self, other) -> 'Tensor':
        """when we do t += other"""
        self.data = self.data + ensure_tensor(other).data
        return self

    def __isub__(self, other) -> 'Tensor':
        """when we do t -= other"""
        self.data = self.data - ensure_tensor(other).data
        return self

    def __imul__(self, other) -> 'Tensor':
        """when we do t *= other"""
        self.data = self.data * ensure_tensor(other).data
        return self

    def __matmul__(self, other) -> 'Tensor':
        return ops._matmul(self, other)

    def __getitem__(self, idxs) -> 'Tensor':
        return ops._slice(self, idxs)
    
    def __pow__(self, to_power) -> "Tensor":
        return ops._pow(self, to_power)

    def __ipow__(self, to_power:int) -> "Tensor":
        self.data = self.data ** to_power
        return self
        

    def backward(self, grad: 'Tensor' = None) ->None:
        assert self.requires_grad, "called backward on non-requires-grad tensor" 

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("grad must be specified for non-0-tensor")

        self.grad.data = self.grad.data + grad.data # type: ignore

        for dependency in self.depends_on:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.tensor.backward(Tensor(backward_grad)) 

    def sum(self) -> 'Tensor':
        return ops._tensor_sum(self)

    
    def reshape(self, shape:Tuple) -> "Tensor":
        return ops._tensor_reshape(self, shape)