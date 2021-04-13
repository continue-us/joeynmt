import torch
import scipy.special
import numpy as np
from torch.autograd import Variable

# from https://raw.githubusercontent.com/Sachin19/seq2seq-con/master/onmt/ive.py

# FIXME !!!!! figure out how to read this from config
# m = 30 # switch to this if embed dim == 30
m = 300

class LogCmk(torch.autograd.Function):
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
        k = k.cpu()
        ctx.save_for_backward(k)
        k = k.double()

        answer = (m/2-1)*torch.log(k) - torch.log(scipy.special.ive(m/2-1, k)) - k - (m/2)*np.log(2*np.pi)
        if torch.cuda.is_available():
            answer = answer.cuda()
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

        # see Appendix 8.2 (https://arxiv.org/pdf/1812.04616.pdf)
        x = -((scipy.special.ive(m/2, k))/(scipy.special.ive(m/2-1,k)))
        if torch.cuda.is_available():
            x = x.cuda()
        x = x.float()

        return grad_output*Variable(x)

class LogCmkApprox(torch.autograd.Function):
    """
    The approximation of the exponentially scaled modified Bessel function of the first kind
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

        # see Appendix 8.2 (https://arxiv.org/pdf/1812.04616.pdf)

        v = m/2-1

        blub = torch.sqrt((v+1)**2+k**2)
        return - (blub - (v-1)*torch.log(v-1 + blub))


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        k, = ctx.saved_tensors

        # see Appendix 8.2 (https://arxiv.org/pdf/1812.04616.pdf)

        v = m/2 - 1

        blab = - k / (v-1+torch.sqrt((v+1)**2+k**2))

        return grad_output*Variable(blab)

