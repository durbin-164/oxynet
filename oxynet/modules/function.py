import oxynet as onet 
from oxynet import Tensor 
from .activation import Softmax 

def softmax(input: Tensor, axis:int = -1) -> Tensor:
    soft = Softmax()
    return soft(input, axis)