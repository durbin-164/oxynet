from oxynet import Tensor 
from oxynet.modules import Module, Parameter
from oxynet.maths import he_initialization
import numpy as np

class Linear(Module):
    def __init__(self,
                in_feature: int ,
                out_feature: int,
                bias: bool = True)-> None:
        
        super().__init__()

        self.in_feature = in_feature
        self.out_feature = out_feature
        self.is_bias = bias

        self.weight = None
        self.bias = None
        self.reset_parameters()

    #TODO : In Linear module
    def reset_parameters(self):
        weight_data = he_initialization(self.in_feature, self.out_feature)
        self.weight = Parameter(data=weight_data)
        if self.is_bias:
            self.bias = Parameter(data=np.zeros((self.out_feature)))
            
  
    def forward(self, input: Tensor) -> Tensor:
        output = input @ self.weight
        if self.is_bias:
            output += self.bias
        
        return output
        

