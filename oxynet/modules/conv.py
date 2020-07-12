from typing import Tuple
from oxynet import Tensor 
from oxynet.modules import Module , Parameter
import numpy as np
from .higher_dimention_ops import img2col, get_conv_output_shape
from oxynet.maths import he_initialization

class Conv2d(Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple,
                 stride: int =1,
                 pad: int = 0,
                 )->None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        
        self.weight = None
        self.bias = None
        
        self.initialize_weights()

    def initialize_weights(self)-> None:
        k_h, k_w = self.kernel_size
        weight_data = he_initialization(self.in_channels*k_h*k_w, self.out_channels)
        
        self.weight = Parameter(data=weight_data)
        self.bias = Parameter(data=np.zeros((self.out_channels)))

    def forward(self, input : Tensor)-> Tensor:
        
        N,_,H,W = input.shape
        out_h,out_w = get_conv_output_shape(H,W,self.kernel_size,self.stride, self.pad)
        
        # col_input shape = (N*out_h*out_w, C*k_h*k_w)
        col_input = img2col(input, self.kernel_size, self.stride, self.pad)
        
        '''
        col_input = (N*out_h*out_w, C *k_h * k_w) @ (C*k_h * k_w, out_channel) = self.weight
        shape [N*out_h*out_h, filter_nums]
        '''
        col_output = col_input @ self.weight + self.bias

        output = col_output.reshape(N, out_h, out_w, -1).transpose(0,3,1,2)
       
        return output




class Flatten(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, input: Tensor)-> Tensor:
        N = input.shape[0]
        return input.reshape(N, -1)

