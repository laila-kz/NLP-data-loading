"""German -> English translation data pipeline on Multi30k.

Demonstrates the full torchtext data-loading flow using the ``nlp_loaders``
package: tokenize -> vocab -> BOS/EOS transforms -> padded (src, tgt) batches.

Note: requires spacy models (``python -m spacy download de_core_news_sm
en_core_web_sm``). Multi30k downloads on first run.
"""

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import Multi30k
from torchtext.vocab import build_vocab_from_iterator

from nlp_loaders.collate import make_translation_collate
from nlp_loaders.transforms import sequential_transforms, tensor_transform, yield_tokens

SRC_LANGUAGE = "de"
TGT_LANGUAGE = "en"

# Tokenizers per language (wrapped spacy models)
token_transform = {
    SRC_LANGUAGE: get_tokenizer("spacy", language="de_core_news_sm"),
    TGT_LANGUAGE: get_tokenizer("spacy", language="en_core_web_sm"),
}

# Special symbols, in index order
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ["<unk>", "<pad>", "<bos>", "<eos>"]

# Peek at one training sample
de, en = next(iter(Multi30k(split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))))
print(f"Source ({SRC_LANGUAGE}): {de}\nTarget ({TGT_LANGUAGE}):  {en}")
print(token_transform[SRC_LANGUAGE](de))

# Build one vocab per language. Sorting by source length keeps padding low.
language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
vocab_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    train_iterator = Multi30k(split="train", language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    sorted_dataset = sorted(train_iterator, key=lambda x: len(x[0].split()))
    vocab_transform[ln] = build_vocab_from_iterator(
        yield_tokens(sorted_dataset, ln, language_index, token_transform),
        min_freq=1,
        specials=special_symbols,
        special_first=True,
    )
    vocab_transform[ln].set_default_index(UNK_IDX)

# Full text pipeline per language: tokenize -> vocab -> BOS/EOS tensor.
# The source is reversed (flip=True), the classic seq2seq encoder input.
text_transform = {
    SRC_LANGUAGE: sequential_transforms(
        token_transform[SRC_LANGUAGE],
        vocab_transform[SRC_LANGUAGE],
        lambda ids: tensor_transform(ids, bos_idx=BOS_IDX, eos_idx=EOS_IDX, flip=True),
    ),
    TGT_LANGUAGE: sequential_transforms(
        token_transform[TGT_LANGUAGE],
        vocab_transform[TGT_LANGUAGE],
        lambda ids: tensor_transform(ids, bos_idx=BOS_IDX, eos_idx=EOS_IDX),
    ),
}

BATCH_SIZE = 4
collate_fn = make_translation_collate(text_transform, SRC_LANGUAGE, TGT_LANGUAGE, pad_idx=PAD_IDX)


def sorted_loader(split: str, drop_last: bool = False) -> DataLoader:
    """Return a DataLoader over one Multi30k split, sorted by source length."""
    iterator = Multi30k(split=split, language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    sorted_iterator = sorted(iterator, key=lambda x: len(x[0].split()))
    return DataLoader(
        sorted_iterator,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )


train_dataloader = sorted_loader("train", drop_last=True)
valid_dataloader = sorted_loader("valid", drop_last=True)

src, trg = next(iter(train_dataloader))
print(src.shape, trg.shape)
