import unittest 
import numpy as np 
from oxynet import Tensor 

class TestTensorTranspose(unittest.TestCase):

    def test_default_transpose(self):
        data = np.random.randn(2,3,4,5,6,7)
        t1 = Tensor(data, requires_grad=True)

        t2 = t1.transpose()

        assert t2.shape == (7,6,5,4,3,2)
        t2.backward(t2)

        assert t1.grad.shape == (2,3,4,5,6,7)

    def test_with_indices_transpose(self):
        data = np.random.randn(2,3,4,5,6)
        t1 = Tensor(data, requires_grad=True)

        t2 = t1.transpose(2,1,4,0,3)

        assert t2.shape == (4,3,6,2,5)
        t2.backward(t2)
        '''
        t2= grad = (4,3,6,2,5) and revert_indices = [3,1,0,4,2]
        == [2,3,4,5,6]
        '''
        assert t1.grad.shape == (2,3,4,5,6)