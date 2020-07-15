from __future__ import annotations
from typing import Tuple
import oxynet as onet
import cupy as cp 

kernel_size_error_message = 'Kernel size must be a list of shape [kernel_h, kernel_w]'

def img2col(inputs: onet.Tensor, ksize, stride=1, pad = 0):
    # data shape = (N*out_h*out_w, C*k_h*k_w)
    data = tensor_to_matrix(inputs.data, ksize, stride, pad)

    requires_grad = inputs.requires_grad
    depends_on = []

    if requires_grad:
        def grad_fn(grad: cp.ndarray) -> cp.ndarray:
            #return grad shape == input shape
            input_shape = inputs.data.shape
            return matrix_to_tensor(grad, input_shape, ksize, stride, pad)

        depends_on = [onet.Dependency(inputs, grad_fn)]

    return onet.Tensor(data,
                       requires_grad,
                       depends_on)


def tensor_to_matrix(inputs, ksize, stride=1, pad=0):
    #https://cs231n.github.io/convolutional-networks/
    #From : https://github.com/qzhao19/Lightweight-Deep-Learning-Framework
    """Reshape tensor of shape [batch_size, channel, height, width] into matrix with 
    shape [batch_size*output_h*output_w, channel*kernel_h*kernel_w] 
   
    Assume you have a image of shape (600, 1, 28, 28), padding=0, stride=2 and a filter with dimensions (3,3). 
    You already know that the output dimension of a convolution operator has to be (13,13) with (28-3)/2 + 1 = 13. 
    tensor_to_matrix creates then a new matrix with the shape of (9 * 1, 600 * 13 * 13) which you then can matrix 
    multiply with your flattend kernel of shape (n,9 * 1). The multiplication will result into a new matrix of shape 
    (n,600*13*13) which you can then reshape into your convolution output (600, n, 13, 13) which is the wanted result. 
    Note that n is the numbers of filters inside your convolution layer.
    Args:
        inputs: 4D inputs tensor of shape [batch_size, channel, height, width]
        ksize: int list, filter shape of [kernel_h, kernel_w] 
        stride: int, he filter convolves around the input volume by shifting one unit at a time
        pad: int, zero padding pads the input volume with zeros around the border
    Returns:
        2D matrix with same type data that tensor of shape [batch_size*output_h*output_w, channel*kernel_h*kernel_w] 
    """
    if len(inputs.shape) != 4:
        raise ValueError('The shape of input tensor must be [inputs_nums, channel, height, width]')
    
    if len(ksize) != 2:
        raise ValueError(kernel_size_error_message)

    if not isinstance(ksize, (tuple, list)):
        ksize = [ksize]

    if not isinstance(stride, int):
        stride = int(stride)
    
    if not isinstance(pad, int):
        pad = int(pad)

    N, C, H, W = inputs.shape
    kernel_h, kernel_w = ksize
    # calculate output shape: output_h and output_w
    out_h , out_w= get_conv_output_shape(H,W, ksize, stride, pad)
    

    # define tensor and matrix
    tensor = cp.pad(inputs, pad_width=[(0, 0), (0, 0), (pad, pad), (pad, pad)], mode='constant')
    matrix = cp.zeros((N, C, kernel_h, kernel_w, out_h, out_w), dtype=inputs.dtype)

    for y in range(kernel_h):
        y_max = y + out_h * stride
        for x in range(kernel_w):
            x_max = x + out_w * stride
            matrix[:, :, y, x, :, :] = tensor[:, :, y:y_max:stride, x:x_max:stride]

    matrix = matrix.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return matrix


def matrix_to_tensor(inputs, desire_shape ,ksize, stride=1, pad=0):
    """Reshape matrix of shape  [batch_size*out_h, batch_size*out_w] into tensor with shape [batch_size, channel, height, width] 
    You can think of this method as the reverse function of tensor_to_matrix. It is adding up the corresponing indices and transforms the given matrix
    back into the initial shape.
    Assuming you have a matrix_to_tensor transformed matrix with shape (9,600*13*13). The original image had a shape of (600,1,28,28) with padding P = 0 
    and stride S = 2. col2im creates out of the im2col matrix and the same hyperparameter a new matrix with a shape of (600, 1, 28, 28).
    Args:
        inputs: 2D matrix of shape [batch_size*output_h*output_w, batch_size*kernel_h*kernel_w] 
        shape: int list, tensor of shape [batch_size, channel, height, width]
        ksize: int list, filter shape of [kernel_h, kernel_w] 
        stride: int, the filter convolves around the input volume by shifting one unit at a time, default value 1
        pad: int, zero padding pads the input volume with zeros around the border, default value 0
    Returns:
        4D tensor of shape [batch_size, channel, height, width]
    """
    if len(ksize) != 2:
        raise ValueError(kernel_size_error_message)
    
    if not isinstance(ksize, (tuple, list)):
        ksize = [ksize]

    if not isinstance(stride, int):
        stride = int(stride)
    
    if not isinstance(pad, int):
        pad = int(pad)

    # get inputs 4d tensor shape
    N, C, H, W = desire_shape
    kernel_h, kernel_w = ksize
    # calculate output tensor shape
    out_h , out_w= get_conv_output_shape(H,W, ksize, stride, pad)

    # here, we difine a matrix with shape [nums, out_height, out_width, in_channel, kernel_height, kernel_width]
    matrix = inputs.reshape(N, out_h, out_w, C, kernel_h, kernel_w).transpose(0, 3, 4, 5, 1, 2)
    tensor = cp.zeros((N, C, H + pad*2 + stride - 1, W + pad*2 + stride - 1), dtype=inputs.dtype)

    for y in range(kernel_h):
        y_max = y + out_h * stride
        for x in range(kernel_w):
            x_max = x + out_w * stride
            tensor[:, :, y:y_max:stride, x:x_max:stride] += matrix[:, :, y, x, :, :]

    return tensor[:, :, pad:H + pad, pad:W + pad]

def get_conv_output_shape(inputs_h, inputs_w, ksize, stride=1, pad=0):
    """compute convolution's shape
    Args:
        inputs_h: int, inputs tensor height
        inputs_w: int, inputs width
        ksize: int list, filter shape of [kernel_h, kernel_w]
        stride: int, the filter convolves around the input volume by shifting one unit at a time, default value 1
        pad: int, zero padding numbers
    Returns:
        outputs height and output width
    """

    if len(ksize) != 2:
        raise ValueError(kernel_size_error_message)

    if not isinstance(inputs_h, int):
        inputs_h = int(inputs_h)
    
    if not isinstance(inputs_w, int):
        inputs_w = int(inputs_w)
    
    if not isinstance(ksize, (tuple, list)):
        ksize = [ksize]
    
    if not isinstance(stride, int):
        stride = int(stride)
    
    if not isinstance(pad, int):
        pad = int(pad)

    kernel_h, kernel_w = ksize
    # calculate output height/width
    output_h = int((inputs_h + pad*2 - kernel_h) / stride) + 1
    output_w = int((inputs_w + pad*2 - kernel_w) / stride) + 1
    return (output_h, output_w)

