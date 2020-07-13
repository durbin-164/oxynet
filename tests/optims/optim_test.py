import unittest 
from oxynet.modules import Module , Parameter
from oxynet.optims import SGD

class TestSGD(unittest.TestCase):
    def test_simple_SGD(self):
        model = OptimHelp()

        optimizer = SGD( 0.01)

        model.zero_grad()

        optimizer.step(model)
        
        assert model.a.shape == (3, 4)
        assert model.b.data.shape ==()



class OptimHelp(Module):
    def __init__(self):
        self.a = Parameter(3,4)
        self.b = Parameter()