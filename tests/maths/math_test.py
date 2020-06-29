import oxynet as onet 
from unittest import TestCase
import numpy as np 

class TestExp(TestCase):
    def test_simple_exp(self):
        t1 = onet.Tensor([1,2,3], requires_grad=True)
        t2 = onet.exp(t1)

        assert t2.data.tolist() == np.exp([1,2,3]).tolist()

        t2.backward(onet.Tensor(1))

        assert t1.grad.data.tolist() == np.exp([1,2,3]).tolist()
