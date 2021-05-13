import math
from tqdm import tqdm
from torch import nn, Tensor
from joeynmt.helpers import freeze_params
import torch
import numpy as np
from joeynmt.constants import PAD_TOKEN
from joeynmt.vocabulary import Vocabulary

import pickle
import os

class Embeddings(nn.Module):

    """
    Simple embeddings class
    """

    # pylint: disable=unused-argument
    def __init__(self,
                 embedding_dim: int = 64,
                 scale: bool = False,
                 vocab_size: int = 0,
                 padding_idx: int = 1,
                 freeze: bool = False,
                 **kwargs):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim,
                                padding_idx=padding_idx)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """
        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)
        return self.lut(x)

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__, self.embedding_dim, self.vocab_size)

class PretrainedEmbeddings(nn.Module):

    """
    Loads pretrained embeddings class
    """
    # pylint: disable=unused-argument
    def __init__(
            self,
            src_vocab: Vocabulary,
            trg_vocab: Vocabulary,
            embedding_dim: int = 300, # or 30
            scale: bool = False,
            vocab_size: int = 0,
            padding_idx: int = 1,
            freeze: bool = False,
            **kwargs
        ):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.
        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()

        self.scale = scale

        # TODO add support for other languages
        # fasttext.util.download_model('de', if_exists='ignore')

        # 30 or 300
        np_embedding = f"ft_deen_{embedding_dim}.np"

        if not os.path.isfile(np_embedding):
            # if else call to avoid importing fasttext if possible
            # because fasttext has unmet dependency on gpu nodes
            import fasttext.util

            src_ft = fasttext.load_model(f'cc.de.{embedding_dim}.bin')
            trg_ft = fasttext.load_model(f'cc.en.{embedding_dim}.bin')

            # Create smaller embeddings, to test on reverse
            # fasttext.util.reduce_model(src_ft, 30)
            # src_ft.save_model('cc.en.30.bin')

            self.embedding_dim = src_ft.get_dimension()

            vectors = []

            for i, word in tqdm(enumerate(src_vocab.itos), desc="adding src vecs"):
                vectors.append(src_ft.get_word_vector(word))
            for i, word in tqdm(enumerate(trg_vocab.itos), desc="adding trg vecs"):
                vectors.append(trg_ft.get_word_vector(word))

            embedding_matrix = np.vstack(vectors)

            with open(np_embedding, "wb") as np_file:
                pickle.dump(embedding_matrix, np_file)

            exit(f"saved joint fasttext embedding matrix as np matrix at {np_embedding}")

        else:

            print("Loading saved embedding ...")
            with open(np_embedding, "rb") as np_file:
                embedding_matrix = pickle.load(np_file)
            print("Loaded saved embedding.")
            self.embedding_dim = embedding_matrix.shape[-1]

        self.lut= nn.Embedding(len(src_vocab)+len(trg_vocab), self.embedding_dim,
            padding_idx=trg_vocab.stoi[PAD_TOKEN])

        self.lut.weight = nn.Parameter(torch.from_numpy(embedding_matrix).float())

        # always freeze pretrained embeddings
        freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.
        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """
        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)
        else:
            return self.lut(x)
