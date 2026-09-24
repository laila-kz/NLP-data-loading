# nlp-loaders

Reusable PyTorch **datasets, collate functions, and text transforms** for NLP data loading, with a set of runnable tutorials that teach how DataLoaders work.

[![CI](https://github.com/<your-org>/nlp-loaders/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-org>/nlp-loaders/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)

## Why this exists

Getting text into a model is surprisingly fiddly: sentences have **variable lengths**, so batches must be **tokenized**, mapped to a **vocabulary**, and **padded** before PyTorch will accept them. Most tutorials bury this logic inside a notebook where it can't be reused — this repo extracts it into a small, importable package and *also* teaches each piece step by step.

Concretely you get:

- **Datasets** — `TextDataset`, `SummarizationDataset`, and a raw `CustomDataset`.
- **Collate functions** — padding raw or tokenized sequences into equal-length batches (`collate_pad`, `make_tokenized_collate`, `make_translation_collate`).
- **Transforms** — `yield_tokens`, `tensor_transform` (BOS/EOS + seq2seq source reversal), `sequential_transforms` — the building blocks of a torchtext-style pipeline.
- **Tutorials** — five runnable scripts that build up the concepts from "what is a DataLoader?" to batching with a Hugging Face summarization model.
- **Examples** — a French batching demo and a full German→English translation pipeline on the Multi30k dataset.

## Project layout

```
.
├── src/nlp_loaders/          # the package
│   ├── datasets.py           # CustomDataset, TextDataset, SummarizationDataset, summarize_texts
│   ├── collate.py            # collate_pad, make_tokenized_collate, make_translation_collate
│   └── transforms.py         # yield_tokens, tensor_transform, sequential_transforms
├── tutorials/                # numbers follow the learning arc; each one imports the package
│   ├── 00_data_loader_basics.py
│   ├── 01_tokenize_to_tensors.py
│   ├── 02_collate_and_padding.py
│   ├── 03_collate_with_tokenizer.py
│   ├── 04_bart_summarization.py
│   └── 05_exercise_padding.py
├── examples/                 # self-contained real-data demos
│   ├── french_batching.py
│   └── multi30k_translation_pipeline.py
└── tests/                    # pytest suite (network tests opt-in)
```

## Install

Requires Python 3.10+.

```bash
git clone <your-repo-url>
cd nlp-loaders
pip install -e ".[dev]"
```

Optional — for `french_batching.py` and the translation pipeline you also need the spacy models:

```bash
python -m spacy download fr_core_news_sm
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Quickstart

```python
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from nlp_loaders import CustomDataset
from nlp_loaders.collate import make_tokenized_collate

sentences = ["Hello world", "This is a longer second sentence"]
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(map(tokenizer, sentences), specials=["<pad>"], special_first=True)

loader = DataLoader(
    CustomDataset(sentences),
    batch_size=2,
    collate_fn=make_tokenized_collate(tokenizer, vocab, padding_value=vocab["<pad>"]),
)
for batch in loader:
    print(batch.shape)  # (2, max_seq_len)
```

## Tutorials & examples

| Script | What it teaches |
|---|---|
| `tutorials/00_data_loader_basics.py` | What a DataLoader is; batching + shuffling |
| `tutorials/01_tokenize_to_tensors.py` | Strings → token ids via `TextDataset` |
| `tutorials/02_collate_and_padding.py` | Custom collate functions, `batch_first` |
| `tutorials/03_collate_with_tokenizer.py` | Tokenizing inside the collate function |
| `tutorials/04_bart_summarization.py` | DataLoaders for generative models (BART, downloads model) |
| `tutorials/05_exercise_padding.py` | Hands-on exercise: control your own padding (solution included) |
| `examples/french_batching.py` | Length-sorted batching on a French corpus |
| `examples/multi30k_translation_pipeline.py` | German→English: tokenize → vocab → BOS/EOS → (src, tgt) batches |

## Testing & development

```bash
pytest                  # runs everything except network-downloading tests
pytest -m network       # opt-in: downloads facebook/bart-large-cnn
ruff check .            # lint
ruff format --check .   # formatting
```

The repo ships with GitHub Actions (`.github/workflows/ci.yml`) that run lint + tests on every push/PR.

## Datasets & prior work

- **Multi30k** (WMT 2016 multimodal) — German/English image captions, loaded via `torchtext.datasets`. Downloaded automatically on first run.
- **facebook/bart-large-cnn** — Hugging Face summarization checkpoint used in `summarize_texts`.
- Tokenization/vocab utilities wrap **spacy** and **torchtext**.
- The teaching structure is inspired by the official PyTorch *Language Translation with Transformer* tutorial and torchtext data-loading docs.

> Note: `torchtext` is in maintenance mode (its `Multi30k` download endpoint is occasionally flaky). The package-core code in `collate.py`/`transforms.py` only depends on PyTorch itself, so most of it keeps working if you swap torchtext for another tokenizer source.

## License

MIT — see [LICENSE](LICENSE). Contributions welcome, see [CONTRIBUTING](CONTRIBUTING.md).