from oxynet import Tensor
from oxynet.layer import tanh 
import numpy as np 
import unittest 

class TestTanh(unittest.TestCase):
    def test_simple_tanh(self):
        a = Tensor([1,2,3], requires_grad=True)
        b = tanh(a)

        np.testing.assert_array_almost_equal(b.data, np.tanh(a.data))

        b.backward(Tensor(1))
        np.testing.assert_array_almost_equal(a.grad.data, 
                                        (1- np.tanh(a.data) *  np.tanh(a.data)))