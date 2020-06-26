from oxynet import Tensor 

x = Tensor([1,2,3], requires_grad = True)

x = x.sum()

print(x)