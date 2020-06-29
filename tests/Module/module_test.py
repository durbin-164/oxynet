import unittest 
from oxynet.model import Module 
from oxynet.layer import Parameter

class TestModule(unittest.TestCase):
    def test_simple_module(self):
        model = ModuleTestHelp()

        model.zero_grad()
        
        assert model.a.shape == (3, 4)
        assert model.b.data.shape ==()

    def test_simple_call(self):
        model = ModuleTestHelp()

        model.zero_grad()
        
        self.assertRaises(NotImplementedError, model)



class ModuleTestHelp(Module):
    def __init__(self):
        self.a = Parameter(3,4)
        self.b = Parameter()