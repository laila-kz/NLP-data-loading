"""Reusable text transforms for building numericalization pipelines.

Small building blocks, each self-contained. The original code read ``SRC_LANGUAGE``,
``BOS_IDX``, etc. from module globals; everything those functions needed is now an
explicit argument.
"""

from collections.abc import Callable, Iterable, Iterator

import torch


def yield_tokens(
    data_iter: Iterable,
    language: str,
    language_index: dict[str, int],
    token_transform: dict[str, Callable[[str], list[str]]],
) -> Iterator[list[str]]:
    """Yield token lists for ``language`` across an iterator of ``(src, tgt)`` samples.

    Used to feed ``torchtext.vocab.build_vocab_from_iterator``.

    Args:
        data_iter: Iterable of samples, each a ``(source, target)`` tuple.
        language: Either ``src_language`` or ``tgt_language``.
        language_index: Maps each language to its column position (0 or 1).
        token_transform: Maps each language to its tokenizer.

    Yields:
        One token list per data sample for the requested language.

    """
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])


def tensor_transform(
    token_ids: list[int],
    *,
    bos_idx: int,
    eos_idx: int,
    flip: bool = False,
) -> torch.Tensor:
    """Wrap a token-id sequence with BOS/EOS markers, optionally reversing it.

    Args:
        token_ids: List of vocabulary indices.
        bos_idx: Index of the ``<bos>`` token.
        eos_idx: Index of the ``<eos>`` token.
        flip: Reverse the sequence. ``True`` produces the reversed source
            sentence used as RNN encoder input (classic seq2seq setup);
            ``False`` keeps the target sentence in reading order.

    Returns:
        ``[bos] + token_ids(+reversed) + [eos]`` as an int64 tensor.

    """
    seq = torch.tensor(token_ids, dtype=torch.int64)
    if flip:
        seq = torch.flip(seq, dims=(0,))
    return torch.cat(
        (
            torch.tensor([bos_idx], dtype=torch.int64),
            seq,
            torch.tensor([eos_idx], dtype=torch.int64),
        )
    )


def sequential_transforms(*transforms: Callable) -> Callable:
    """Compose multiple transforms into a single callable applied left-to-right.

    Example:
        >>> pipeline = sequential_transforms(tokenize, look_up_vocab, to_tensor)
        >>> pipeline("Hello world")

    """

    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input

    return func


__all__ = ["yield_tokens", "tensor_transform", "sequential_transforms"]
