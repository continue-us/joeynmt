# coding: utf-8
"""
Module to implement training loss
"""
import math
import torch
from torch import nn, Tensor
from torch.autograd import Variable

from joeynmt.ive import LogCmk, LogCmkApprox

class vMF(nn.Module):
    """ Von Mises Fisher Loss
    from https://raw.githubusercontent.com/Sachin19/seq2seq-con/master/onmt/ive.py
    """
    def __init__(self, embed_dim, pad_index):
        super().__init__()
        self.m = embed_dim
        self.pad_index = pad_index
        self.warmup = True
        self.logcmk_fun = LogCmkApprox.apply
        # self.logcmk_fun = LogCmk.apply

    def increase_precision(self):
        self.logcmk_fun = LogCmk.apply

    def forward(self, outputs, targets, target_embeddings):

        # permute batch and time dimension to iterate over time
        target_vectors = target_embeddings(targets)

        trg_vec_norm = torch.nn.functional.normalize(target_vectors, p=2, dim=-1)
        out_vec_norm = torch.nn.functional.normalize(outputs, p=2, dim=-1)

        # reg2 = out_vec_norm * trg_vec_norm
        lambda2 = 0.1
        cos = (out_vec_norm * trg_vec_norm).sum(dim=-1)

        # reg1 = kappa 
        lambda1 = 0.02
        kappa = outputs.norm(p=2, dim=-1)

        # vMF LOSS with both regularisations:
        nll_loss = - self.logcmk_fun(kappa) - lambda2 * cos + lambda1 * kappa
        # nll_loss = - lambda2 * cos + lambda1 * kappa

        # discard pad
        mask = targets.ne(self.pad_index)

        loss = nll_loss.masked_select(mask).sum()

        return loss


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
            as_iuple=False)
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
