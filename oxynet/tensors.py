import numpy as np 

class TensorBase:
    """
    Base tensor class
    """

    node_uid = ''

    def __init__(self, md_array=None, name=None):
        pass



    def convert_in_numpy(self, md_array=None):

        if type(md_array) == list:
            return np.array(md_array)
        elif type(md_array) == np.ndarray:
            return md_array
        else:
            raise ValueError("Arguments must be of type 'list' or 'np.ndarray'")
