# General Notes 

- https://arxiv.org/pdf/1812.04616


- working branch is master atm
- dont add large embedding files (make sure .gitignore is correct/up to date before doing git add .)
- ive.py contains three different calculations of Log(C\_{m,k}) (see paper):
-- LogCmk: explicit calculation of exponentially called modified Bessel of first kind; calls into scipy, which is why the authors implemented it as autograd.Function to provide backward explicitly)
-- LogCmkApprox: Approximate logcmk (as explained in paper, for early stages of training, as LogCmk yields NaN for FIXME which values?) with the backward precomputed (should be more efficient than logcmk\_autobackward)), computation according to Appendix 8.2
-- logcmk\_autobackward: the author's implementation of logcmkapprox, with just the forward specified

# TODO


## Marvin notes

* according to 2nd to last commit following things TODO:
- change from bpe to word embeddings (DONE i think? verify!)
- autoregressive training??
- re-understand what that was about the output layer not changing? The loss changed I believe? but grad is always exactly identical? (this is checked in ll 510 of training.py)

## Down the road stuff

* call vMF.increase\_precision() to switch to True LogCmk after some ```n_warmup``` steps during training

