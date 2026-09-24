"""Collate functions for turning variable-length sequences into padded batches.

The functions in this module are self-contained: every dependency (tokenizer,
vocab, text transforms, languages) is passed explicitly as an argument. The
original versions read these values from module globals, which broke as soon as
they were imported from another module.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


def collate_pad(
    batch: list[torch.Tensor],
    *,
    batch_first: bool = True,
    padding_value: int = 0,
) -> torch.Tensor:
    """Pad already-tokenized tensors to the longest sequence in the batch.

    Args:
        batch: Sequence of variable-length 1D tensors.
        batch_first: ``True`` yields ``(batch_size, max_seq_len)``,
            ``False`` yields ``(max_seq_len, batch_size)``.
        padding_value: Index used to fill the padded positions.

    Returns:
        A single padded tensor.

    """
    return pad_sequence(batch, batch_first=batch_first, padding_value=padding_value)


def make_tokenized_collate(
    tokenizer: Callable[[str], list[str]],
    vocab: Any,
    *,
    batch_first: bool = True,
    padding_value: int = 0,
) -> Callable[[list[str]], torch.Tensor]:
    """Build a collate function that tokenizes raw strings, maps them to vocab ids and pads.

    Args:
        tokenizer: Callable mapping a string to a list of token strings.
        vocab: Mapping of token -> index with a ``__getitem__`` interface
            (e.g. a ``torchtext`` Vocab).
        batch_first: See ``collate_pad``.
        padding_value: Index used to fill the padded positions.

    Returns:
        Collate function for use with ``torch.utils.data.DataLoader``.

    """

    def collate_fn(batch: list[str]) -> torch.Tensor:
        tensor_batch = [
            torch.tensor([vocab[token] for token in tokenizer(sample)], dtype=torch.int64) for sample in batch
        ]
        return pad_sequence(tensor_batch, batch_first=batch_first, padding_value=padding_value)

    return collate_fn


def make_translation_collate(
    text_transform: dict[str, Callable[[str], list[int]]],
    src_language: str,
    tgt_language: str,
    *,
    pad_idx: int = 0,
) -> Callable[[list[tuple[Any, Any]]], tuple[torch.Tensor, torch.Tensor]]:
    """Build a collate function for ``(source, target)`` text pairs.

    Args:
        text_transform: Mapping of language -> full text pipeline
            (tokenize -> vocab -> BOS/EOS tensor), e.g. composed with
            ``nlp_loaders.transforms.sequential_transforms``.
        src_language: Key in ``text_transform`` for the source column.
        tgt_language: Key in ``text_transform`` for the target column.
        pad_idx: Index used to fill the padded positions.

    Returns:
        Collate function returning a padded ``(src_batch, tgt_batch)`` pair of tensors.

    """

    def collate_fn(batch: list[tuple[Any, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
        src_batch, tgt_batch = [], []
        for src_sample, tgt_sample in batch:
            src_sequences = torch.tensor(text_transform[src_language](src_sample.rstrip("\n")), dtype=torch.int64)
            tgt_sequences = torch.tensor(text_transform[tgt_language](tgt_sample.rstrip("\n")), dtype=torch.int64)
            src_batch.append(src_sequences)
            tgt_batch.append(tgt_sequences)

        src_batch = pad_sequence(src_batch, padding_value=pad_idx, batch_first=True)
        tgt_batch = pad_sequence(tgt_batch, padding_value=pad_idx, batch_first=True)
        return src_batch, tgt_batch

    return collate_fn


__all__ = ["collate_pad", "make_tokenized_collate", "make_translation_collate"]
