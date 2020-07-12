from oxynet import Tensor
from typing import Tuple
import numpy as np 

class Parameter(Tensor):
    def __init__(self, *shape: Tuple, data = None) ->None:
        
        self.data = np.random.randn(*shape)
        if data is not None:
            self.data = data
        super().__init__(self.data, requires_grad=True)

    def __repr__(self):
        return 'Parameter containing:\n' + super().__repr__()
