"""03 — Tokenizing inside the collate function.

Keeping the dataset "dumb" (raw strings) and doing tokenization inside the
collate function is a common pattern: the dataset stays cheap to build, and
the vocab/tokenizer logic lives in one place.

``nlp_loaders.collate.make_tokenized_collate(tokenizer, vocab)`` wraps that up:
it returns a collate function that tokenizes each sample, maps tokens through
the vocab, and pads the batch.

Run:  python tutorials/03_collate_with_tokenizer.py
"""

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from nlp_loaders import CustomDataset
from nlp_loaders.collate import make_tokenized_collate

SENTENCES = [
    "If you want to know what a man's like.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry.",
    "You are awesome!",
]

tokenizer = get_tokenizer("basic_english")

# Reserve <pad> at the front so its id is 0 (that becomes the padding value).
vocab = build_vocab_from_iterator(
    map(tokenizer, SENTENCES),
    specials=["<pad>"],
    special_first=True,
)
PAD_IDX = vocab["<pad>"]

collate_fn = make_tokenized_collate(tokenizer, vocab, padding_value=PAD_IDX)

# Dataset returns plain strings; the collate function does the rest.
raw_dataset = CustomDataset(SENTENCES)
dataloader = DataLoader(raw_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

for i, batch in enumerate(dataloader):
    words = [[vocab.get_itos()[idx] for idx in row] for row in batch.tolist()]
    print(f"batch {i}: shape={tuple(batch.shape)}")
    for row in words:
        print("   ", row)
