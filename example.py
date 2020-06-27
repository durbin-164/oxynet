from oxynet import Tensor 
from oxynet.layer import Parameter

x = Tensor([1,2,3], requires_grad = True)

x = x.sum()

print(x)

x = Parameter(3,4)
print(x)
print(x.sum())