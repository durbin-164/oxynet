import unittest 
import numpy as np 
from oxynet import Tensor 

class TestTensorReshape(unittest.TestCase):

    def test_simple_reshape(self):
        data = np.random.randn(3,10,10)
        t1 = Tensor(data, requires_grad=True)
        print(t1.shape)

        t2 = t1.reshape(3, 100)

        assert t2.shape == (3,100)
        t2.backward(t2)

        assert t1.grad.shape == (3,10,10)