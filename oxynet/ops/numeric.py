from __future__ import annotations
from typing import Tuple
import oxynet as onet
import numpy as np 
from .utils import inv_permutation

def transpose(t:onet.Tensor, indices:Tuple=None)-> onet.Tensor:
    '''
     By default, reverse the dimensions,
     otherwise permute the axes according to the values given.

     if indices = None : numpy reverse the dimensions.
    '''
    if indices is None or len(indices)==0 :
        indices = tuple(range(t.data.ndim-1, -1, -1))

    data = np.transpose(t.data, indices)
    requires_grad = t.requires_grad 

    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray)-> np.ndarray:
            '''
            indices.index(v) -> return: index of value for v. 
            if indices = [2,1,4,0,3], then
            revert_indices = [3,1,0,4,2]
            '''
            revert_indices = tuple(inv_permutation(indices))

            return np.transpose(grad,revert_indices)

        depends_on = [onet.Dependency(t, grad_fn)]

    return onet.Tensor(data, 
                       requires_grad,
                       depends_on)