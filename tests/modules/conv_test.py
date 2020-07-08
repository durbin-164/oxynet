from unittest import TestCase 
from oxynet.modules import Conv2d , Flatten
from oxynet import Tensor 
import numpy as np 

class TestConv2d(TestCase):
    def test_simple_cov2d(self):
        input = Tensor(np.random.randn(2,3,8,8), requires_grad= True)

        # print(input.shape)
        
        conv = Conv2d(3, 5,(5,5))

        output = conv(input)

        assert output.shape == (2,5,4,4)

        output.backward(output)

        assert input.grad.shape == (2,3,8,8)

        ######
        input = Tensor(np.random.randn(3,3,8,9), requires_grad= True)
        
        conv = Conv2d(3, 10,(3,3), 2)

        output = conv(input)

        assert output.shape == (3,10,3,4)

        output.backward(output)

        assert input.grad.shape == (3,3,8,9)


# class TestFlatten(TestCase):
#     def test_sample_flatten(self):
#         input = Tensor(np.random.randn(3,8,9,3), requires_grad= True)
        
#         conv = Conv2d(3, 10,3)
#         flat = Flatten()

#         output = conv(input)
#         output = flat(output)

#         assert output.shape == (3, 420)
