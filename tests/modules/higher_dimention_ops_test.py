from unittest import TestCase
import numpy as np 
from oxynet import Tensor 
import oxynet as onet 
from oxynet.modules.higher_dimention_ops import img2col

class TestImg2Col(TestCase):
    def test_img2col_simple(self):
        data = np.random.randn(4,3,28,28)
        t1 = Tensor(data, requires_grad=True)
        t2 = img2col(t1, ksize=(3,3), stride=2, pad=0)
        # data shape = (N*out_h*out_w, C*k_h*k_w)
        assert t2.data.shape == (4*13*13, 3* 3*3)

        t2.backward(Tensor(t2.data))
        #t1 grad shape == input shape
        assert t1.grad.shape == (4, 3, 28, 28)