import oxynet as onet
import numpy as np 

def transpose(t:onet.Tensor, indices=None)-> onet.Tensor:
    '''
     By default, reverse the dimensions,
     otherwise permute the axes according to the values given.

     if indices = None : numpy reverse the dimensions.
    '''
    data = np.transpose(t.data, indices)
    requires_grad = t.requires_grad 

    depends_on = []

    if requires_grad:
        def grad_fn(grad: np.ndarray)-> np.ndarray:
            if indices is None:
                grad = grad.transpose()
            else:
                '''
                indices.index(v) -> return: index of value for v. 
                if indices = [2,1,4,0,3], then
                revert_indices = [3,1,0,4,2]
                '''
                revert_indices = tuple([indices.index(v) for v in range(len(indices))])

                grad = grad.transpose(revert_indices)

            return grad

            depends_on = [onet.Dependency(t, grad_fn)]

        return onet.Tensor(data, 
                           requires_grad,
                           depends_on)