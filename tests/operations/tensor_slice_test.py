import unittest 
import numpy as np 
from oxynet import Tensor 

class TestTensorSlice(unittest.TestCase):

    def test_simple_slice(self):
        data = np.random.randn(10,10)
        t1 = Tensor(data, requires_grad=True)

        t2 = t1[2:5, 5:]

        assert t2.shape == (3,5)
        t2.backward(Tensor(1))

        assert t1.grad.shape == (10,10)