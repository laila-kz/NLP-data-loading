"""Unit tests for nlp_loaders.collate."""

import torch

from nlp_loaders.collate import collate_pad, make_tokenized_collate, make_translation_collate


def _split_tokenizer(text: str) -> list[str]:
    return text.split()


def test_collate_pad_batch_first():
    batch = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    padded = collate_pad(batch)
    assert padded.shape == (2, 3)
    assert padded.tolist() == [[1, 2, 3], [4, 5, 0]]


def test_collate_pad_seq_first():
    batch = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    padded = collate_pad(batch, batch_first=False)
    assert padded.shape == (3, 2)
    assert padded.tolist() == [[1, 4], [2, 5], [3, 0]]


def test_collate_pad_custom_padding_value():
    batch = [torch.tensor([1]), torch.tensor([2, 3])]
    padded = collate_pad(batch, padding_value=9)
    assert padded.tolist() == [[1, 9], [2, 3]]


def test_make_tokenized_collate_pads_short_seq():
    vocab = {"<pad>": 0, "a": 1, "bb": 2, "ccc": 3}
    collate_fn = make_tokenized_collate(_split_tokenizer, vocab, padding_value=0)

    padded = collate_fn(["a ccc", "bb"])

    assert padded.shape == (2, 2)
    assert padded.tolist() == [[1, 3], [2, 0]]


def test_make_translation_collate_moves_tensors_to_device():
    text_transform = {
        "de": lambda s: [int(x) for x in s.split()],
        "en": lambda s: [int(x) for x in s.split()],
    }
    collate_fn = make_translation_collate(text_transform, "de", "en", pad_idx=0)

    src, trg = collate_fn([("1 2 3\n", "4 5\n")])

    assert src.shape == (1, 3)
    assert trg.shape == (1, 2)
    assert src.tolist() == [[1, 2, 3]]
    assert trg.tolist() == [[4, 5]]
