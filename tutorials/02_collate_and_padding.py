"""02 — Padding batches with a collate function.

Once samples are tensors of *different lengths*, the default collate function
(which stacks samples with ``torch.stack``) crashes. A custom collate function
tells the DataLoader how to combine a list of samples into one batch — and for
NLP we normally *pad* them to the longest sequence in the batch.

``nlp_loaders.collate_pad`` is a ready-made one. The only subtlety is the
``batch_first`` flag:

- batch_first=True  -> shape (batch_size, max_seq_len)   [each row = one sentence]
- batch_first=False -> shape (max_seq_len, batch_size)   [each column = one sentence]

Run:  python tutorials/02_collate_and_padding.py
"""

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from nlp_loaders import CustomDataset, TextDataset
from nlp_loaders.collate import collate_pad

SENTENCES = [
    "If you want to know what a man's like.",
    "Fame's a fickle friend, Harry.",
    "You are awesome!",
]

tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(map(tokenizer, SENTENCES))

# From step 01: a dataset that yields variable-length token id tensors.
tensor_dataset = TextDataset(SENTENCES, tokenizer=tokenizer, vocab=vocab)

# batch_first=True
dataloader = DataLoader(tensor_dataset, batch_size=2, collate_fn=collate_pad)
for batch in dataloader:
    print("batch_first=True  ->", tuple(batch.shape), "  padding with 0 at the end")

# batch_first=False
dataloader_t = DataLoader(
    tensor_dataset,
    batch_size=2,
    collate_fn=lambda b: collate_pad(b, batch_first=False),
)
for batch in dataloader_t:
    print("batch_first=False ->", tuple(batch.shape))

# Raw-text dataset: collate_pad only knows tensors, so strings would break.
# Feed it tensors (as above) or use make_tokenized_collate (step 03).
raw = CustomDataset(SENTENCES)
try:
    next(iter(DataLoader(raw, batch_size=2, collate_fn=collate_pad)))
except TypeError as exc:
    print("\nDefault collate would also fail here — tensors only:", exc)
