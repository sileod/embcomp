"""
Code for the paper
Composition of Sentence Embeddings: Lessons from Statistical Relational Learning
@inproceedings{sileo-etal-2019-composition,
    title = "Composition of Sentence Embeddings: Lessons from Statistical Relational Learning",
    author = "Sileo, Damien  and
      Van De Cruys, Tim  and
      Pradel, Camille  and
      Muller, Philippe",
    booktitle = "Proceedings of the Eighth Joint Conference on Lexical and Computational Semantics (*{SEM} 2019)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/S19-1004",
    doi = "10.18653/v1/S19-1004",
    pages = "33--43",
}
"""

import torch


def prod(x, y):
    """x*y"""
    return x * y


def complexprod(x, y, mode="full"):
    """Re(x)*Re(y), Im(x)*Re(y), Re(x)*Im(y)
    This composition is an asymmetric yet compact variant of the hadamard product.
    Even dimensions as treated as real numbers and  odd dimensions are treated as complex numbers.
    http://proceedings.mlr.press/v48/trouillon16.pdf
    """
    if mode == "full":
        return torch.cat(
            [x * y, x[:, ::2] * y[:, 1::2], y[:, ::2] * x[:, 1::2]], axis=1
        )
    if mode == "asym":
        return torch.cat(
            [x * y, x[:, ::2] * y[:, 1::2] - y[:, ::2] * x[:, 1::2]], axis=1
        )


def analogy(x, y, t=0):
    """t ∈ ℝ^dim(x)
    This makes the compositions asymmetric and enables analogical reasonning in the latent space.
    This generalizes absolute difference.
    https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html
    """
    return torch.abs(x - y + t)


def absdiff(x, y):
    return torch.abs(x - y)


def left(x, y):
    return x


def right(x, y):
    return y


def cat(*compositions):
    """Helper for concatenation of compositions, e.g "cat([left,right])(x,y)"""

    def concatenated(x, y):
        return torch.cat([c(x, y) for c in compositions], axis=1)

    return concatenated


def heuristic(x, y):
    return cat(left, right, prod, absdiff)

