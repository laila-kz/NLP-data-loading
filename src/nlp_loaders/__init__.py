"""nlp-loaders: reusable PyTorch data-loading utilities for NLP.

Public API:
    - datasets: ``CustomDataset``, ``TextDataset``, ``SummarizationDataset``, ``summarize_texts``
    - collate: ``collate_pad``, ``make_tokenized_collate``, ``make_translation_collate``
    - transforms: ``yield_tokens``, ``tensor_transform``, ``sequential_transforms``
"""

from nlp_loaders.collate import collate_pad, make_tokenized_collate, make_translation_collate
from nlp_loaders.datasets import CustomDataset, SummarizationDataset, TextDataset, summarize_texts
from nlp_loaders.transforms import sequential_transforms, tensor_transform, yield_tokens

__version__ = "0.1.0"

__all__ = [
    "collate_pad",
    "make_tokenized_collate",
    "make_translation_collate",
    "CustomDataset",
    "SummarizationDataset",
    "TextDataset",
    "summarize_texts",
    "sequential_transforms",
    "tensor_transform",
    "yield_tokens",
]
