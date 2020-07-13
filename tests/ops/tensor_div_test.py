from unittest import TestCase
from oxynet import Tensor 

class TestDiv(TestCase):
    def test_simple_div(self):
        t1 = Tensor(10, requires_grad=True)
        t2 = Tensor(20, requires_grad=True)
        t3 = t2/t1
        
        assert t3.data == 2. 

        t3.backward()

        assert t1.grad.data == 20* (-1./10**2)
        assert t2.grad.data == 1./10
        