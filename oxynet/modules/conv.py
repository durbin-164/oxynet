from typing import Tuple
from oxynet import Tensor 
from oxynet.modules import Module , Parameter

class Conv2d(Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int =1)->None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.kernels = []

    def initialize_weights(self)-> None:
        for i in range(self.out_channels):
            self.kernels.append( Parameter(self.kernel_size, self.kernel_size,self.in_channels))

    def forward(self, input : Tensor)-> Tensor:
        b, h,w, c = input.shape
        temp = self.kernel_size//2
        output = Parameter(b, h-temp*2,w-temp*2,self.out_channels)
        for b, image in enumerate(input):
            for o , kernel in enumerate(self.kernels):
                for i in range(h-self.kernel_size):
                    for j in range( w-self.kernel_size):
                        output[b,i, j,o] = (image[i:self.kernel_size, j:self.kernel_size] * kernel).sum()


        return output




class Flatten(Module):
    def __init__(self) -> None:
        super().__init__()

    
    def forward(self, input: Tensor):
        b = input.shape[0]
        out_shape = 1
        for i in range(1,len(input.shape)):
            out_shape *= input.shape[i]

        return input.reshape((b, out_shape))

