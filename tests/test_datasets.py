"""Unit tests for nlp_loaders.datasets."""

import pytest
import torch

from nlp_loaders import CustomDataset, TextDataset
from nlp_loaders.datasets import SummarizationDataset, summarize_texts


def test_custom_dataset_len_and_getitem():
    dataset = CustomDataset(["a", "b", "ccc"])
    assert len(dataset) == 3
    assert dataset[0] == "a"
    assert dataset[2] == "ccc"


def test_text_dataset_returns_raw_text_without_tokenizer():
    dataset = TextDataset(["hello world"])
    assert dataset[0] == "hello world"


def test_text_dataset_tokenizes_when_tokenizer_and_vocab_given():
    tokenizer = lambda s: s.split()  # noqa: E731
    vocab = {"hello": 5, "world": 7}
    dataset = TextDataset(["hello world"], tokenizer=tokenizer, vocab=vocab)

    tensor = dataset[0]

    assert tensor.dtype == torch.int64
    assert tensor.tolist() == [5, 7]


class _FakeTokenizer:
    """Emulates the Hugging Face tokenizer API used by SummarizationDataset."""

    vocab = {"hello": 0, "world": 1, "<pad>": 2}

    def __call__(self, text, truncation=True, padding="max_length", max_length=512, return_tensors="pt"):
        ids = [self.vocab[w] for w in text.split()]
        truncated = ids[:max_length]
        padded = truncated + [self.vocab["<pad>"]] * (max_length - len(truncated))
        return {
            "input_ids": torch.tensor([padded]),
            "attention_mask": torch.tensor([[1] * len(truncated) + [0] * (max_length - len(truncated))]),
        }


def test_summarization_dataset_tokenizes_single_sample():
    tokenizer = _FakeTokenizer()
    dataset = SummarizationDataset(["hello world"], tokenizer=tokenizer, max_length=4)

    sample = dataset[0]

    assert sample["input_ids"].shape == (4,)
    assert sample["attention_mask"].shape == (4,)
    assert sample["input_ids"].tolist() == [0, 1, 2, 2]
    assert sample["attention_mask"].tolist() == [1, 1, 0, 0]


@pytest.mark.network
def test_summarize_texts_smoke():
    texts = [
        "Machine learning is a subset of artificial intelligence that focuses on building "
        "systems that can learn from and make decisions based on data."
    ]
    summaries = summarize_texts(texts, model_name="facebook/bart-large-cnn", summary_length=20)
    assert len(summaries) == 1
    assert isinstance(summaries[0], str)
    assert summaries[0]
