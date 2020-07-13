import numpy as np 

def scatter(source: np.ndarray, axis:int, indices: np.ndarray, values)->np.ndarray:
    '''
    https://pytorch.org/docs/stable/tensors.html#torch.Tensor.scatter_

    Writes all values from the tensor src into self at the indices
     specified in the index tensor. For each value in src, its output
      index is specified by its index in src for dimension != dim 
      and by the corresponding value in index for dimension = dim.
    '''
    for i, roll in enumerate(np.rollaxis(source, axis)):
        roll += (indices == i).astype(int) * values
    return source