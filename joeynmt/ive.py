import torch
import scipy.special
import numpy as np
from torch.autograd import Variable

# from https://raw.githubusercontent.com/Sachin19/seq2seq-con/master/onmt/ive.py

# FIXME !!!!! figure out how to read this from config
# m = 30 # switch to this if embed dim == 30
m = 300

class Logcmk(torch.autograd.Function):
    """
    The exponentially scaled modified Bessel function of the first kind
    """
    @staticmethod
    def forward(ctx, k):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(k)
        k = k.double()
        if torch.cuda.is_available():
            answer = (m/2-1)*torch.log(k) - torch.log(scipy.special.ive(m/2-1, k)).cuda() - k - (m/2)*np.log(2*np.pi)
        else:
            answer = (m/2-1)*torch.log(k) - torch.log(scipy.special.ive(m/2-1, k)) - k - (m/2)*np.log(2*np.pi)
        answer = answer.float()
        return answer

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        k, = ctx.saved_tensors
        k = k.double()

        x = -((scipy.special.ive(m/2, k))/(scipy.special.ive(m/2-1,k)))
        if torch.cuda.is_available():
            x = x.cuda()
        x = x.float()

        return grad_output*Variable(x)

# approximation of LogC(m, k)
def factory_approx_logcmk(m):
    def logcmkapprox(z):
        v = m/2-1

        blub = torch.sqrt((v+1)**2+z**2)
        return - (blub - (v-1)*torch.log(v-1 + blub))
    return logcmkapprox
