from __future__ import annotations
from typing import List
import oxynet.tensor as tensor
import numpy as np

def _tensor_sum(t: tensor.Tensor) -> tensor.Tensor:
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
        
        depends_on = [tensor.Dependency(t, grad_fn)]
    
    else:
        depends_on = []

    return tensor.Tensor(data,
                 requires_grad,
                 depends_on)



def _add(t1: tensor.Tensor, t2: tensor.Tensor) ->tensor.Tensor:
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

    depends_on: List[tensor.Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:

            grad = __handle_array_broadcasting(grad, t1.data)

            return grad
        depends_on.append(tensor.Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = __handle_array_broadcasting(grad, t2.data)
                    
            return grad

        depends_on.append(tensor.Dependency(t2, grad_fn2))

    return tensor.Tensor(data, 
                  requires_grad,
                  depends_on)


def _mul(t1: tensor.Tensor, t2: tensor.Tensor) ->tensor.Tensor:
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

    depends_on: List[tensor.Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:

            grad = grad * t2.data

            grad = __handle_array_broadcasting(grad, t1.data)

            return grad
        depends_on.append(tensor.Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:

            grad = grad * t1.data

            grad = __handle_array_broadcasting(grad, t2.data)
                    
            return grad

        depends_on.append(tensor.Dependency(t2, grad_fn2))

    return tensor.Tensor(data, 
                  requires_grad,
                  depends_on)


def _neg(t1: tensor.Tensor) -> tensor.Tensor:
    data = -t1.data
    requires_grad  = t1.requires_grad

    if requires_grad:
        depends_on = [tensor.Dependency(t1, lambda x : -x)]
    else:
        depends_on = []

    return tensor.Tensor(data,
                 requires_grad,
                 depends_on)

def _sub(t1: tensor.Tensor, t2: tensor.Tensor) -> tensor.Tensor:
    return t1 +  -t2


def _matmul(t1: tensor.Tensor, t2: tensor.Tensor) -> tensor.Tensor:
    """
    if t1 is (n1, m1) and t2 is (m1, m2), then t1 @ t2 is (n1, m2)
    so grad3 is (n1, m2)
    if t3 = t1 @ t2, and grad3 is the gradient of some function wrt t3, then
        grad1 = grad3 @ t2.T
        grad2 = t1.T @ grad3
    """
    data = t1.data @ t2.data
    requires_grad = t1.requires_grad or t2.requires_grad

    depends_on: List[tensor.Dependency] = []

    if t1.requires_grad:
        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T

        depends_on.append(tensor.Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad
        depends_on.append(tensor.Dependency(t2, grad_fn2))

    return tensor.Tensor(data,
                  requires_grad,
                  depends_on)


def _slice(t:tensor.Tensor, idxs) -> tensor.Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(t.data)
            bigger_grad[idxs] = grad

            return bigger_grad
        depends_on = [tensor.Dependency(t, grad_fn)]

    else:
        depends_on = []

    return tensor.Tensor(data,
                  requires_grad,
                  depends_on)



def __handle_array_broadcasting(grad, data):
    ndims_added = grad.ndim - data.ndim

    for _ in range(ndims_added):
        grad = grad.sum(axis = 0)

    for i, dim in enumerate(data.shape):
        if dim == 1:
            grad = grad.sum(axis = i, keepdims = True)

    return grad