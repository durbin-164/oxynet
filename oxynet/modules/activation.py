import cupy as cp 
from oxynet.tensor import Tensor, Dependency 
import oxynet as onet 
from oxynet.modules import Module

class Softmax(Module):
    '''
    stablesoftmax
    '''
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, axis: int = -1)->Tensor:
        max_vals = onet.max(input, axis=axis, keepdims=True)
        # avoid overflow/underflow issue
        exp_vals = onet.exp(input - max_vals)
        sum_vals = exp_vals.sum( axis=axis, keepdims=True)
        return exp_vals / sum_vals



def tanh(tensor: Tensor) -> Tensor:
    data = cp.tanh(tensor.data)
    requires_grad = tensor.requires_grad

    if requires_grad:
        def grad_fn(grad: cp.ndarray) -> cp.ndarray:
            return grad * (1 - data * data)

        depends_on = [Dependency(tensor, grad_fn)]
    else:
        depends_on = []


    return Tensor(data, 
              requires_grad,
              depends_on)

