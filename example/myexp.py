import cupy as cp 
a = cp.array([1,2,3])
print(a)

import parent_package
import oxynet as onet 

a = onet.Tensor([[1,2,3],[3,2,1]], requires_grad=True)

print(a)
b = a[:,0]
print(b)