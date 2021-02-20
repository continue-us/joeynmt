# coding: utf-8
"""
Module to implement training loss
"""

import math
import torch
from torch import nn, Tensor
from torch.autograd import Variable

class vMF(nn.Module):
    """ Von Mises Fisher Loss """
    def __init__(self, embed_dim, pad_index):
        super().__init__()
        self.m = embed_dim
        self.pad_index = pad_index
        # self._param_init = nn.NLLLoss()

    def forward(self, outputs, targets, target_embeddings, eval=False):

        # approximation of LogC(m, k)
        def logcmkappox(d, z):
            v = d/2-1
            return torch.sqrt((v+1)*(v+1)+z*z) - (v-1)*torch.log(v-1 + torch.sqrt((v+1)*(v+1)+z*z))

        loss = []
        cosine_loss = []
        outputs = Variable(outputs.data, requires_grad=(not eval), volatile=eval)

        batch_size = outputs.size(0)

        for i, (out_t, targ_t) in enumerate(zip(outputs, targets)):
            #print(out_t.shape)
            # out_vec_t = out_t.view(-1, out_t.size(2))

            kappa_times_mean = out_t
            tar_vec_t = target_embeddings(targ_t)
            # tar_vec_t = tar_vec_t.view(-1, tar_vec_t.size(2))

            kappa = out_t.norm(p=2, dim=-1) #*tar_vec_t.norm(p=2,dim=-1)

            tar_vec_norm_t = torch.nn.functional.normalize(tar_vec_t, p=2, dim=-1)
            out_vec_norm_t = torch.nn.functional.normalize(out_t, p=2, dim=-1)

            cosine_loss_t = (1.0-(out_vec_norm_t*tar_vec_norm_t).sum(dim=-1)).masked_select(targ_t.view(-1).ne(self.pad_index)).sum()

            lambda2 = 0.1
            lambda1 = 0.02
            # nll_loss = - logcmk(kappa) + kappa*(lambda2-lambda1*(out_vec_norm_t*tar_vec_norm_t).sum(dim=-1))
            # nll_loss = - logcmk(kappa) + torch.log(1+kappa)*(0.2-(out_vec_norm_t*tar_vec_norm_t).sum(dim=-1))
            nll_loss = logcmkappox(self.m, kappa) + torch.log(1+kappa)*(0.2-(out_vec_norm_t*tar_vec_norm_t).sum(dim=-1))

            loss_t = nll_loss.masked_select(targ_t.view(-1).ne(self.pad_index)).sum()
    
            loss.append(loss_t)
            cosine_loss.append(cosine_loss_t)

            # joey does this in training.py:
            # if not eval:
            #     loss_t.div(batch_size).backward()

        grad_output = None if outputs.grad is None else outputs.grad.data

        rloss =  torch.stack(loss).sum()
        return rloss
        

class XentLoss(nn.Module):
    """
    Cross-Entropy Loss with optional label smoothing
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # standard xent loss
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index,
                                        reduction='sum')
        else:
            # custom label-smoothed loss, computed with KL divergence loss
            self.criterion = nn.KLDivLoss(reduction='sum')

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        Smooth target distribution. All non-reference words get uniform
        probability mass according to "smoothing".

        :param targets: target indices, batch*seq_len
        :param vocab_size: size of the output vocabulary
        :return: smoothed target distributions, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # fill distribution uniformly with smoothing
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # assign true label the probability of 1-smoothing ("confidence")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0-self.smoothing)
        # give padding probability of 0 everywhere
        smooth_dist[:, self.pad_index] = 0
        # masking out padding area (sum of probabilities for padding area = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index,
            as_tuple=False)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        Compute the cross-entropy between logits and targets.

        If label smoothing is used, target distributions are not one-hot, but
        "1-smoothing" for the correct target token and the rest of the
        probability mass is uniformly spread across the other tokens.

        :param log_probs: log probabilities as predicted by model
        :param targets: target indices
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1),
                vocab_size=log_probs.size(-1))
            # targets: distributions with batch*seq_len x vocab_size
            assert log_probs.contiguous().view(-1, log_probs.size(-1)).shape \
                == targets.shape
        else:
            # targets: indices with batch*seq_len
            targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets)
        return loss
