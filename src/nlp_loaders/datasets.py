"""Custom Dataset implementations for NLP tasks."""

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class CustomDataset(Dataset):
    """A minimal dataset that stores and returns raw text samples unchanged."""

    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class TextDataset(Dataset):
    """Dataset for text data with optional on-the-fly tokenization.

    Args:
        texts: List of text samples.
        tokenizer: Callable mapping a string to a list of tokens.
        vocab: Mapping of token -> index used to convert tokens to ids.

    Example:
        >>> dataset = TextDataset(["Hello world", "NLP is fun"])
        >>> len(dataset)
        2

    """

    def __init__(self, texts, tokenizer=None, vocab=None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> torch.Tensor | str:
        """Get sample at index ``idx`` as a token-id tensor, or raw text.

        Args:
            idx: Sample index.

        Returns:
            Raw text if ``tokenizer`` and ``vocab`` are not set,
            otherwise a tokenized int64 tensor.

        """
        text = self.texts[idx]

        if self.tokenizer and self.vocab:
            tokens = self.tokenizer(text)
            return torch.tensor([self.vocab[token] for token in tokens], dtype=torch.int64)

        return text


class SummarizationDataset(Dataset):
    """A Dataset describing a single summarization training/inference sample.

    Each ``__getitem__`` tokenizes one text with a Hugging Face tokenizer,
    truncating to ``max_length`` and padding to a fixed length.

    Args:
        texts: List of texts to be summarized.
        tokenizer: Hugging Face tokenizer.
        max_length: Maximum sequence length for tokenized inputs.

    """

    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Tokenize the input text
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def summarize_texts(
    texts,
    model_name="facebook/bart-large-cnn",
    batch_size=4,
    max_length=512,
    summary_length=50,
):
    """Summarize a list of texts with a pretrained Hugging Face seq2seq model.

    Args:
        texts: List of texts to summarize.
        model_name: Pretrained Hugging Face model name for summarization.
        batch_size: Batch size for the DataLoader.
        max_length: Maximum input sequence length.
        summary_length: Maximum length of generated summaries.

    Returns:
        List of generated summaries.

    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Create DataLoader
    dataset = SummarizationDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Generate summaries
    model.eval()
    summaries = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Summarizing"):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=summary_length,
                num_beams=4,
                early_stopping=True,
            )

            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            summaries.extend(decoded)

    return summaries
