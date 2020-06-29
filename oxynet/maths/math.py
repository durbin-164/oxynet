from oxynet import Tensor , Dependency
import numpy as np 

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