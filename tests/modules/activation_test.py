from oxynet import Tensor
from oxynet.modules import tanh , Softmax
import cupy as cp 
import unittest 
import oxynet.modules.function as F

class TestSoftmax(unittest.TestCase):
    def test_simple_softmax(self):
        # softmax = Softmax()

        input = Tensor([1,2,3], requires_grad=True)
        output = F.softmax(input)

        expected = cp.array([0.090031, 0.244728, 0.665241])
        cp.testing.assert_array_almost_equal(output.data, expected)

        assert output.data.sum().tolist() == 0.9999999999999998

        #TODO: Test backward
        # grad_data = Tensor([1,1,1])
        # output.backward(grad_data)

        
        # expected = input.data * (1-input.data)
        # np.testing.assert_array_almost_equal(input.grad.data, expected)

    def test_2d_softmax(self):
        # softmax = Softmax()

        input = Tensor([[1,2,3],[1,2,3]], requires_grad=True)
        output = F.softmax(input)

        expected = cp.array([[0.090031, 0.244728, 0.665241],[0.090031, 0.244728, 0.665241]])
        cp.testing.assert_array_almost_equal(output.data, expected)

        assert output.data.sum(axis=-1).tolist() == [0.9999999999999998,0.9999999999999998]


class TestTanh(unittest.TestCase):
    def test_simple_tanh(self):
        a = Tensor([1,2,3], requires_grad=True)
        b = tanh(a)

        cp.testing.assert_array_almost_equal(b.data, cp.tanh(a.data))

        b.backward(Tensor(1))
        cp.testing.assert_array_almost_equal(a.grad.data, 
                                        (1- cp.tanh(a.data) *  cp.tanh(a.data)))


                        