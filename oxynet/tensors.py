import numpy as np 

class TensorBase:
    """
    Base tensor class
    """

    node_uid = ''
    value = ''

    def __init__(self, md_array, name=None):
       self.value = self.__get_node_value(md_array = md_array)
       self.node_uid = self.__get_node_uid(name = name)
    
    def __get_node_value(self, md_array):

        if type(md_array) == list:
            return np.array(md_array)
        elif type(md_array) == np.ndarray:
            return md_array
        else:
            raise ValueError("Arguments must be of type 'list' or 'np.ndarray'")
    

    def __repr__(self):
        return f"Tensor( \n {self.value.__str__()}\n"
    

    def __get_node_uid(self, name):
        if name:
            if type(name) == str:
                 return name
            else:
                raise ValueError("Argument 'name' must be type 'str'")

        else:
            return  self.__get_hash_number()

    
    def __get_hash_number(self):
        return str(hash(np.random.random()))


    @property
    def shape(self):
        return self.value.shape



class Tensor(TensorBase):
    name = 'Tensor'

class TensorConstant(TensorBase):
    name = 'Tensor Constant'