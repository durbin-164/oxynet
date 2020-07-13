from oxynet import Tensor
from oxynet.modules import tanh , Softmax
import numpy as np 
import unittest 
import oxynet.modules.function as F

class TestSoftmax(unittest.TestCase):
    def test_simple_softmax(self):
        # softmax = Softmax()

        input = Tensor([1,2,3], requires_grad=True)
        output = F.softmax(input)

        expected = np.array([0.090031, 0.244728, 0.665241])
        np.testing.assert_array_almost_equal(output.data, expected)

        #TODO: Test backward
        # grad_data = Tensor([1,1,1])
        # output.backward(grad_data)

        
        # expected = input.data * (1-input.data)
        # np.testing.assert_array_almost_equal(input.grad.data, expected)


class TestTanh(unittest.TestCase):
    def test_simple_tanh(self):
        a = Tensor([1,2,3], requires_grad=True)
        b = tanh(a)

        np.testing.assert_array_almost_equal(b.data, np.tanh(a.data))

        b.backward(Tensor(1))
        np.testing.assert_array_almost_equal(a.grad.data, 
                                        (1- np.tanh(a.data) *  np.tanh(a.data)))


                        