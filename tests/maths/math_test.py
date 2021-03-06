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


class TestLog(TestCase):
    def test_simple_log(self):
        #Float test
        t1 = onet.Tensor(1, requires_grad=True)
        t2 = onet.log(t1)
        assert t2.data == np.log(1)

        t2.backward(onet.Tensor(1))

        assert t1.grad.data == 1

        ## Array test
        t1 = onet.Tensor([1,2,3], requires_grad=True)
        t2 = onet.log(t1)

        assert t2.data.tolist() == np.log([1,2,3]).tolist()
        t2.backward(onet.Tensor([10,10,12]))
        assert t1.grad.data.tolist() == [10,5,4]


class TestMax(TestCase):
    def test_simple_max(self):
        #test with axis = None
        t1 = onet.Tensor([[2,4,8,10],[3,15,4,5]], requires_grad=True)
        t2 = onet.max(t1, keepdims=True)

        assert t2.data == [[15]]
        t2.backward(onet.Tensor([[20]]))
        outdata = np.zeros((2,4))
        outdata[1][1]=20
        np.testing.assert_array_almost_equal(t1.grad.data, outdata)


    def test_max_when_axis_0(self):
        #test with axis = 0
        t1 = onet.Tensor([[2,4,8,10],[3,15,4,5]], requires_grad=True)
        t2 = onet.max(t1, axis=0, keepdims=True)

        assert t2.data.tolist() == [[3,15,8,10]]
        t2.backward(onet.Tensor([[10,20,30,40]]))
        outdata = np.zeros((2,4))
        outdata[1][0]=10
        outdata[1][1]=20
        outdata[0][2]=30
        outdata[0][3]=40
        np.testing.assert_array_almost_equal(t1.grad.data, outdata)

    def test_max_when_axis_1(self):
        #test with axis = 0
        t1 = onet.Tensor([[2,4,8,10],[3,15,4,5]], requires_grad=True)
        t2 = onet.max(t1, axis=1)
    
        assert t2.data.tolist() == [10,15]
        t2.backward(onet.Tensor([10,20]))
        outdata = np.zeros((2,4))
        outdata[0][3]=10
        outdata[1][1]=20
        np.testing.assert_array_almost_equal(t1.grad.data, outdata)