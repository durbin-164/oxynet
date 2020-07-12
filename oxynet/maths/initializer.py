import numpy as np 

def he_initialization(*shape):
    data = np.random.randn(*shape) * np.sqrt(2.0/shape[0])
    return data