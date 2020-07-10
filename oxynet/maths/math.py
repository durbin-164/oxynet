from oxynet import Tensor , Dependency
import numpy as np 
from .utils import scatter

def exp(tensor: Tensor) -> Tensor:

    data = np.exp(tensor.data)
    requires_grad = tensor.requires_grad

    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray)->np.ndarray:
            return grad * data  
        
        depends_on = [Dependency(tensor, grad_fn)]

    return Tensor(data,
                  requires_grad,
                  depends_on)


def log(t:Tensor)-> Tensor:
    data = np.log(t.data)

    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.divide(1, t.data)
        depends_on = [Dependency(t, grad_fn)]
    return Tensor(data,
                  requires_grad,
                  depends_on)


def max(t: Tensor, axis = None, keepdims = False)->Tensor:
    data = np.max(t.data, axis, keepdims=keepdims)

    requires_grad = t.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            bigger_grad = np.zeros_like(t.data)
            if keepdims:
                grad = np.squeeze(grad)
            if axis  is None:
               # If there is no axis, the argmax is the location of he maximum single element
                max_indices = np.unravel_index(np.argmax(t.data), t.shape)
                bigger_grad[max_indices] = grad
            else:
                # If there is an axis, we reconstruct the bigger matrix by 'rolling' on this axis
                max_indices = np.argmax(t.data, axis=axis)
                bigger_grad = scatter(bigger_grad, axis, max_indices, grad)

            return bigger_grad

        depends_on.append(Dependency(t, grad_fn))

    return Tensor(data,
                  requires_grad,
                  depends_on)