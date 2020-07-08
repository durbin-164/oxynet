from __future__ import annotations
from typing import List, Tuple
import oxynet as onet
import numpy as np
from .utils import handle_array_broadcasting

def tensor_sum(t: onet.Tensor) -> onet.Tensor:
    """
    Takes a tensor and returns the 0-tensor
    that's the sum of all its elements.
    """

    data = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            """
            grad is necessarily a 0-tensor, so each
            input element contributes that much
            """

            return grad * np.ones_like(t.data)
        
        depends_on = [onet.Dependency(t, grad_fn)]
    
    else:
        depends_on = []

    return onet.Tensor(data,
                 requires_grad,
                 depends_on)



def add(t1: onet.Tensor, t2: onet.Tensor) ->onet.Tensor:
    '''
    y = a + b
    we have dL/dy
    So, dL/da = dL/dy * dy/da = dL/dy * 1 = dL/dy

    or
    let think y gradient change by eps
    So, y = a+eps + b = a + b + eps
    '''
    data = t1.data + t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[onet.Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:

            grad = handle_array_broadcasting(grad, t1.data)

            return grad
        depends_on.append(onet.Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = handle_array_broadcasting(grad, t2.data)
                    
            return grad

        depends_on.append(onet.Dependency(t2, grad_fn2))

    return onet.Tensor(data, 
                  requires_grad,
                  depends_on)


def mul(t1: onet.Tensor, t2: onet.Tensor) ->onet.Tensor:
    '''
    y = a * b
    we have dL/dy
    So, dL/da = dL/dy * dy/da = dL/dy * b
    And dL/db = DL/dy * dy/db = dL/dy * a

    or 

    Let y change by y = (a+eps) * b
    So, y = a * b + (eps * b)
    So gradient change (by eps * b)
    '''
    data = t1.data * t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[onet.Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:

            grad = grad * t2.data

            grad = handle_array_broadcasting(grad, t1.data)

            return grad
        depends_on.append(onet.Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:

            grad = grad * t1.data

            grad = handle_array_broadcasting(grad, t2.data)
                    
            return grad

        depends_on.append(onet.Dependency(t2, grad_fn2))

    return onet.Tensor(data, 
                  requires_grad,
                  depends_on)


def neg(t1: onet.Tensor) -> onet.Tensor:
    data = -t1.data
    requires_grad  = t1.requires_grad

    if requires_grad:
        depends_on = [onet.Dependency(t1, lambda x : -x)]
    else:
        depends_on = []

    return onet.Tensor(data,
                 requires_grad,
                 depends_on)

def sub(t1: onet.Tensor, t2: onet.Tensor) -> onet.Tensor:
    return t1 +  -t2


def matmul(t1: onet.Tensor, t2: onet.Tensor) -> onet.Tensor:
    """
    if t1 is (n1, m1) and t2 is (m1, m2), then t1 @ t2 is (n1, m2)
    so grad3 is (n1, m2)
    if t3 = t1 @ t2, and grad3 is the gradient of some function wrt t3, then
        grad1 = grad3 @ t2.T
        grad2 = t1.T @ grad3
    """
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[onet.Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T

        depends_on.append(onet.Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad
        depends_on.append(onet.Dependency(t2, grad_fn2))

    return onet.Tensor(data,
                  requires_grad,
                  depends_on)


def slice(t:onet.Tensor, idxs) -> onet.Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(t.data)
            bigger_grad[idxs] = grad

            return bigger_grad
        depends_on = [onet.Dependency(t, grad_fn)]

    else:
        depends_on = []

    return onet.Tensor(data,
                  requires_grad,
                  depends_on)




def tensor_reshape(t: onet.Tensor, shape: Tuple) -> onet.Tensor:
    data = t.data.reshape(shape)

    requires_grad = t.requires_grad 

    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            if grad.shape == t.data.shape:
                return grad
            return grad * np.ones_like(t.data)

        depends_on = [onet.Dependency(t, grad_fn)]

    return onet.Tensor(data,
                        requires_grad,
                        depends_on)


def pow(t: onet.Tensor, to_power:int) -> onet.Tensor:

    data = np.power(t.data, to_power)
    requires_grad = t.requires_grad 
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * to_power * np.power(t.data, to_power-1) 

        depends_on = [onet.Dependency(t, grad_fn)]

    return onet.Tensor(data, 
                         requires_grad,
                         depends_on)



