import oxynet as onet 
from oxynet import Tensor 
from oxynet.modules import Module 
from .function import softmax

class CrossEntropyLoss(Module):
    def forward(self, pred: Tensor, actual: Tensor):
        batch_size = Tensor(pred.shape[0], requires_grad =True)

        logits = softmax(pred)

        delta = Tensor(1e-15, requires_grad=True)
        cross_entropy = -onet.sum(actual * onet.log(logits + delta))/batch_size

        return cross_entropy