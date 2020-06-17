import numpy as np 
import unittest 
from oxynet.tensors import TensorBase

class TestTensorBase(unittest.TestCase):

    def test_tensor_value_and_node_uid(self):
        # Test tensor value
        tensorBase = TensorBase([1,2,3], "tensor")
        self.assertEqual( type(tensorBase.value), np.ndarray)
        
        tensorBase = TensorBase(np.array([1,2,3]), "tensor")
        self.assertEqual( type(tensorBase.value) , np.ndarray)

        with self.assertRaises(ValueError) as context:
            tensorBase = TensorBase(10, "tensor")
        self.assertTrue("Arguments must be of type 'list' or 'np.ndarray'" in str(context.exception))


        # Test tensor node_uid

        tensorBase = TensorBase([1,2,3], "tensor")
        self.assertEqual( tensorBase.node_uid, 'tensor')

        tensorBase = TensorBase([1,2,3])
        self.assertEqual( type(tensorBase.node_uid), str)

        with self.assertRaises(ValueError) as context:
            tensorBase = TensorBase([1], 10)
        self.assertTrue("Argument 'name' must be type 'str'" in str(context.exception))


        #Test property shape
        tensorBase = TensorBase([[1,2,3], [4,5,6]])
        self.assertEqual( tensorBase.shape, (2,3))

        #Test __repr__ 
        tensorBase = TensorBase([1,2,3])
        self.assertEqual( repr(tensorBase), 'Tensor( \n [1 2 3]\n')
