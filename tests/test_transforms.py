"""Unit tests for nlp_loaders.transforms."""

import torch

from nlp_loaders.transforms import sequential_transforms, tensor_transform, yield_tokens


def _tokenize_de(text: str) -> list[str]:
    return text.split()


def _tokenize_en(text: str) -> list[str]:
    return text.split("-")


def test_yield_tokens_extracts_requested_language():
    token_transform = {"de": _tokenize_de, "en": _tokenize_en}
    language_index = {"de": 0, "en": 1}
    data = [("a b", "x-y"), ("c", "z")]

    de_tokens = list(yield_tokens(data, "de", language_index, token_transform))
    en_tokens = list(yield_tokens(data, "en", language_index, token_transform))

    assert de_tokens == [["a", "b"], ["c"]]
    assert en_tokens == [["x", "y"], ["z"]]


def test_tensor_transform_adds_bos_eos_in_order():
    result = tensor_transform([5, 6], bos_idx=2, eos_idx=3)
    assert result.tolist() == [2, 5, 6, 3]


def test_tensor_transform_flip_reverses_sequence():
    result = tensor_transform([5, 6], bos_idx=2, eos_idx=3, flip=True)
    assert result.tolist() == [2, 6, 5, 3]


def test_tensor_transform_returns_int64():
    result = tensor_transform([1], bos_idx=2, eos_idx=3)
    assert result.dtype == torch.int64


def test_sequential_transforms_applies_left_to_right():
    pipeline = sequential_transforms(lambda s: s.strip(), lambda s: s.upper())
    assert pipeline("  hello  ") == "HELLO"
