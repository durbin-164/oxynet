import oxynet as onet 
from oxynet import Tensor 
from oxynet.modules import CrossEntropyLoss 
from unittest import TestCase 
import numpy as np 

class TestCrossEntropyLoss(TestCase):
    def test_simple_cross_entropy(self):
        criterion = CrossEntropyLoss()

        pred_data = np.random.randn(10, 10)
        label_data = np.random.randn(10, 10)

        pred = Tensor(pred_data, requires_grad=True)
        label = Tensor(label_data, requires_grad=True)

        loss = criterion(pred, label)

        assert loss.data.shape == ()

        loss.backward()

        assert pred.grad.shape == (10,10)

    def test_with_label_n_1_cross_entropy(self):
        criterion = CrossEntropyLoss()

        pred_data = np.random.randn(10, 10)
        label_data = np.arange(10).reshape(10,1)

        pred = Tensor(pred_data, requires_grad=True)
        label = Tensor(label_data, requires_grad=True)

        loss = criterion(pred, label)

        assert loss.data.shape == ()

        loss.backward()

        assert pred.grad.shape == (10,10)