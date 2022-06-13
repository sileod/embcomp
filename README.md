# Composition of embeddings (e.g. word embeddings, sentence embeddings)

`pip install embcomp`

```python
import embcomp

x=torch.randn(3,10)
y=torch.randn(3,10)

t=torch.randn(3,10)

# combine compositions by concatenating their output 
embcomp.cat([embcomp.complexprod, embcomp.abs])(x,y)
```

This package provide helpers for composition of embeddings. I particularly recommend trying out embcomp.complexprod prod instead of embcomp.prod and embcomp.analogy instead of embcomp.absdiff for asymmetric tasks (e.g. natural language inference.).
`prod` and `absdiff` are used in SentEval. However, they are not expressive enough to solve asymmetrical tasks. 

```bib
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
```
