# coding: utf-8
"""
Module to implement training loss
"""
import math
import torch
from torch import nn, Tensor
from torch.autograd import Variable

from joeynmt.ive import Logcmk, factory_approx_logcmk

class vMF(nn.Module):
    """ Von Mises Fisher Loss
    from https://raw.githubusercontent.com/Sachin19/seq2seq-con/master/onmt/ive.py
    """
    def __init__(self, embed_dim, pad_index):
        super().__init__()
        self.m = embed_dim
        self.pad_index = pad_index
        self.warmup = True
        self.logcmk_fun = factory_approx_logcmk(self.m)

    def increase_precision(self):
        self.logcmk_fun = Logcmk.apply

    def forward(self, outputs, targets, target_embeddings):

        loss = []
        cosine_loss = []

        batch_size = outputs.size(0)

        # permute batch and time dimension to iterate over time
        outputs_timewise = outputs.permute(1,0,2)
        target_vectors = target_embeddings(targets)
        targets_timewise = target_vectors.permute(1,0,2)

        # input(f"batch dim: {batch_size}; outputs: {outputs.shape}, targets: {targets.shape}")

        for t, (out_t, trg_t) in enumerate(zip(outputs_timewise, targets_timewise)):

            trg_vec_norm_t = torch.nn.functional.normalize(trg_t, p=2, dim=-1)
            out_vec_norm_t = torch.nn.functional.normalize(out_t, p=2, dim=-1)

            # print(nll_loss.shape)

            # print("out_t:",out_t.shape,"trg_t:", trg_t.shape)

            # reg2 = l2 *(out_t @ trg_t.T)
            lambda2 = 0.1
            # reg2 = out_vec_norm_t * trg_vec_norm_t
            cos = out_vec_norm_t @ trg_vec_norm_t.T

            # reg1 = l1 * kappa 
            lambda1 = 0.02
            kappa = out_t.norm(p=2, dim=-1)

            # vMF LOSS with both regularisations:
            # nll_loss = -logcmkapprox(self.m, kappa) + lambda2 * reg2 + lambda1 * kappa

            # nll_loss = - Logcmk.apply(self.m,kappa) + torch.log(1+kappa)*(0.2-(out_vec_norm_t*trg_vec_norm_t).sum(dim=-1))
            # this runs:
            # nll_loss = - Logcmk.apply(kappa) + kappa * (lambda2-lambda1*(out_vec_norm_t*trg_vec_norm_t).sum(dim=-1))
            # print("shapes:", Logcmk.apply(kappa).shape, out_vec_norm_t.shape, trg_vec_norm_t.shape,reg2.shape,kappa.shape)
            # input()
            nll_loss = - self.logcmk_fun(kappa) - lambda2 * cos + lambda1 * kappa
            # nll_loss = - Logcmk.apply(kappa) + kappa*(lambda2-lambda1*(out_vec_norm_t*trg_vec_norm_t).sum(dim=-1))
            # nll_loss = logcmkapprox(self.m, kappa) + torch.log(1+kappa)*(0.2-(out_vec_norm_t*normed_trg).sum(dim=-1))

            # print("full nll:",nll_loss.shape)
            # print("reg1:",kappa.shape, "reg2", reg2.shape)
            # input()
            mask = targets[:,t].ne(self.pad_index)
            # input(f"mask: {mask.shape}, nll_loss: {nll_loss.shape}, batch_size: {batch_size}")

            loss_t = nll_loss.masked_select(mask).sum()

            loss.append(loss_t)

        full_loss = torch.stack(loss).sum()
        return full_loss


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
