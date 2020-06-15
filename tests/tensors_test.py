import numpy as np 
import unittest 
from oxynet.tensors import TensorBase

class TestTensorBase(unittest.TestCase):

    def test_convert_in_numpy(self):
        tensorBase = TensorBase()
        
        self.assertEqual( type(tensorBase.convert_in_numpy([1,2,3])), np.ndarray)
        self.assertEqual( type(tensorBase.convert_in_numpy(np.array([1,2,3]))) , np.ndarray)

        with self.assertRaises(ValueError) as context:
            tensorBase.convert_in_numpy()
        self.assertTrue("Arguments must be of type 'list' or 'np.ndarray'" in str(context.exception))
