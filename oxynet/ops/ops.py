from __future__ import annotations
from typing import List, Tuple
import oxynet as onet
import cupy as cp
from .utils import handle_array_broadcasting

def sum(t: onet.Tensor, axis = None, keepdims = False) -> onet.Tensor:
    """
    Takes a tensor and returns the 0-tensor
    that's the sum of all its elements.
    """

    data = t.data.sum(axis=axis, keepdims = keepdims)
    requires_grad = t.requires_grad
    depends_on = []
    if requires_grad:
        def grad_fn(grad: cp.ndarray) -> cp.ndarray:
            """
            grad is necessarily a 0-tensor, so each
            input element contributes that much
            """
            # We need to keep the information on which axis the sum was made (to be broadcasting compatible)
            # We always reshape the gradient in the same axis for back-propagation
            data_keepdims = t.data.sum(axis=axis, keepdims=True)
            return grad.reshape(data_keepdims.shape) + cp.zeros_like(t.data)

        depends_on = [onet.Dependency(t, grad_fn)]
    
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
        def grad_fn1(grad: cp.ndarray) -> cp.ndarray:

            grad = handle_array_broadcasting(grad, t1.data)

            return grad
        depends_on.append(onet.Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: cp.ndarray) -> cp.ndarray:
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
        def grad_fn1(grad: cp.ndarray) -> cp.ndarray:

            grad = grad * t2.data

            grad = handle_array_broadcasting(grad, t1.data)

            return grad
        depends_on.append(onet.Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: cp.ndarray) -> cp.ndarray:

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
        def grad_fn1(grad: cp.ndarray) -> cp.ndarray:
            return grad @ t2.data.T

        depends_on.append(onet.Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad: cp.ndarray) -> cp.ndarray:
            return t1.data.T @ grad
        depends_on.append(onet.Dependency(t2, grad_fn2))

    return onet.Tensor(data,
                  requires_grad,
                  depends_on)


def slice(t:onet.Tensor, idxs) -> onet.Tensor:
    data = t.data[idxs]
    requires_grad = t.requires_grad

    if requires_grad:
        def grad_fn(grad: cp.ndarray) -> cp.ndarray:
            bigger_grad = cp.zeros_like(t.data)
            bigger_grad[idxs] = grad

            return bigger_grad
        depends_on = [onet.Dependency(t, grad_fn)]

    else:
        depends_on = []

    return onet.Tensor(data,
                  requires_grad,
                  depends_on)


def multiply(t1: onet.Tensor, t2: onet.Tensor) -> onet.Tensor:
    r"""Elementwise multiplication of two tensors-like object.

    .. math::

        T_{out} = T_1 \times T_2

    Args:
        t1 (Tensor like): tensor to multiply
        t2 (Tensor like): second tensor to multiply with

    Returns:
        Tensor
    """
    data = cp.multiply(t1.data, t2.data)
    requires_grad = t1.requires_grad or t2.requires_grad
    depends_on = []

    if t1.requires_grad:
        def grad_fn1(grad):
            r"""Update the gradient from t1 for the the multiplication operation, :math:`grad = grad \times T_2`.

            Shape:
                - inputs (np.ndarray): upstream gradient with shape the same shape as inputs data :math:`T_1`.
                - outputs (np.ndarray): downstream gradient with shape the same shape as inputs data :math:`T_1`.
            """
            grad = grad * t2.data
            # Sum out added dims
            ndims_added = grad.ndim - t1.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(onet.Dependency(t1, grad_fn1))

    if t2.requires_grad:
        def grad_fn2(grad):
            r"""Update the gradient from t2 for the the multiplication operation, :math:`grad = grad \times T_1`.

            Shape:
                - inputs (np.ndarray): upstream gradient with shape the same shape as inputs data :math:`T_2`.
                - outputs (np.ndarray): downstream gradient with shape the same shape as inputs data :math:`T_2`.
            """
            grad = grad * t1.data
            # Sum out added dims
            ndims_added = grad.ndim - t2.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            # Sum across broadcasted (but non-added dims)
            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            return grad

        depends_on.append(onet.Dependency(t2, grad_fn2))

    return onet.Tensor(data, requires_grad, depends_on)


def inverse(t:onet.Tensor) -> onet.Tensor:
    r"""Inverse a tensor-like object.

    .. math::

        T_{out} = \frac{1}{T}

    Args:
        t (Tensor like): tensor to inverse.

    Returns:
        Tensor
    """
    data = 1.0/t.data
    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad):
            r"""Update the gradient for the inverse operation, :math:`grad = grad \times \frac{-1}{T^2}`.

            Shape:
                - inputs (np.ndarray): upstream gradient.
                - outputs (np.ndarray): downstream gradient.
            """
            return - 1. / (t.data ** 2) * grad

        depends_on = [onet.Dependency(t, grad_fn)]

    return onet.Tensor(data,
                       requires_grad,
                       depends_on)


def div(t1: onet.Tensor, t2: onet.Tensor)-> onet.Tensor:
    r"""Divide two tensor-like object.

    .. math::

        T_{out} = T_1 \times \frac{1}{T_2}

    Args:
        t1 (Tensor like): tensor to multiply
        t2 (Tensor like): tensor to invert

    Returns:
        Tensor
    """
    return multiply(t1, inverse(t2))



def pow(t: onet.Tensor, to_power:int) -> onet.Tensor:

    data = cp.power(t.data, to_power)
    requires_grad = t.requires_grad 
    depends_on = []

    if requires_grad:
        def grad_fn(grad: cp.ndarray) -> cp.ndarray:
            return grad * to_power * cp.power(t.data, to_power-1) 

        depends_on = [onet.Dependency(t, grad_fn)]

    return onet.Tensor(data, 
                         requires_grad,
                         depends_on)



