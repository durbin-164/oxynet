from unittest import TestCase 
from oxynet.modules import Conv2d , Flatten
from oxynet import Tensor 
import numpy as np 

class TestConv2d(TestCase):
    def test_simple_cov2d(self):
        input = Tensor(np.random.randn(2,8,8,3), requires_grad= True)
        
        conv = Conv2d(3, 5,5)

        output = conv(input)

        assert output.shape == (2,4,4,5)

        ######
        input = Tensor(np.random.randn(3,8,9,3), requires_grad= True)
        
        conv = Conv2d(3, 10,3)

        output = conv(input)

        assert output.shape == (3,6,7,10)


class TestFlatten(TestCase):
    def test_sample_flatten(self):
        input = Tensor(np.random.randn(3,8,9,3), requires_grad= True)
        
        conv = Conv2d(3, 10,3)
        flat = Flatten()

        output = conv(input)
        output = flat(output)

        assert output.shape == (3, 420)
