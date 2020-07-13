from unittest import TestCase
from oxynet.modules import Linear 
from oxynet import Tensor
import numpy as np 

class TestLinear(TestCase):
    def test_simple_linear_forward(self):
        fc1 = Linear(20, 10)

        input = Tensor( np.random.randn(100, 20), requires_grad= True)

        output = fc1(input)

        assert output.shape == (100, 10)




