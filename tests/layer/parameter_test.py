from oxynet.layer import Parameter
import unittest

class TestParameter(unittest.TestCase):
    def test_simple_parameter(self):
        p = Parameter(3,4)

        assert p.shape == (3,4)

        assert type(repr(p)) == str