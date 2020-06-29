from oxynet import Tensor 
from oxynet.modules import Module, Parameter

class Linear(Module):
    def __init__(self,
                in_feature: int ,
                out_feature: int,
                bias: bool = True)-> None:
        
        super().__init__()

        self.in_feature = in_feature
        self.out_feature = out_feature

        self.weight = Parameter(self.in_feature, self.out_feature)

        if bias:
            self.bias = Parameter(self.out_feature)

    #TODO : In Linear module
    def reset_parameters(self):
        raise NotImplementedError
  
    def forward(self, input: Tensor) -> Tensor:
        output = input @ self.weight
        if self.bias is not None:
            output += self.bias
        
        return output
        

