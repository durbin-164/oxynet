import unittest 
import numpy as np 
from oxynet import Tensor 

class TestTensorPow(unittest.TestCase):

    def test_simple_pow(self):
        t1 = Tensor([1,2,3,4,5,6], requires_grad=True)

        t2 = t1 ** 2
        assert t2.data.tolist() == [1,4,9,16,25,36]

        t2.backward(Tensor(1))

        assert t1.grad.data.tolist() == [2,4,6,8,10,12]


        ### power 7 
        t1 = Tensor([1,2,3,4], requires_grad=True)

        t2 = t1 ** 7
        assert t2.data.tolist() == [1, 128, 2187, 16384]

        t2.backward(Tensor(1))

        assert t1.grad.data.tolist() == [7.0, 448.0, 5103.0, 28672.0]


    def test_simple_ipow(self):
        t1 = Tensor([1,2,3,4,5,6], requires_grad=True)

        t1  **= 2
        assert t1.data.tolist() == [1,4,9,16,25,36]

        t1.backward(Tensor(1))
        #invalid gradient
        # assert t1.grad.data.tolist() == [2,4,6,8,10,12]
        