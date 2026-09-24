"""05 — Exercise: control your own padding.

The notebook this repo grew from advertised an *Exercise* section that was
never written. Here it is — do it before peeking at the solution at the bottom.

Setup:
    1. Build a vocab over the sentences below that reserves BOTH "<unk>" and
       "<pad>" (in that order) so their ids are 0 and 1.
    2. Tokenize with "basic_english".
    3. Build a DataLoader that:
       - uses a CustomDataset of the raw sentences
       - uses make_tokenized_collate with padding_value=<pad> id
       - batches of size 2, no shuffling
    4. Print each batch, then print each batch's shape.

Things to try / questions:
    - What happens if you pad with PAD_IDX vs with 0 when PAD_IDX=1?
    - Why is shuffle=True (as in step 03) run-to-run different here?
    - What would break if a vocab were built WITHOUT "<pad>"?

Run:  python tutorials/05_exercise_padding.py
"""

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from nlp_loaders import CustomDataset
from nlp_loaders.collate import make_tokenized_collate

SENTENCES = [
    "The quick brown fox jumps over the lazy dog.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump!",
    "Sphinx of black quartz, judge my vow.",
]

# YOUR CODE START


# YOUR CODE END


# ---------------------------------------------------------------------------
# SOLUTION (try it yourself first!)
# ---------------------------------------------------------------------------
def solution():
    tokenizer = get_tokenizer("basic_english")
    vocab = build_vocab_from_iterator(
        map(tokenizer, SENTENCES),
        specials=["<unk>", "<pad>"],
        special_first=True,
    )
    UNK_IDX, PAD_IDX = vocab["<unk>"], vocab["<pad>"]
    vocab.set_default_index(UNK_IDX)  # unknown words map to <unk>

    collate_fn = make_tokenized_collate(tokenizer, vocab, padding_value=PAD_IDX)
    dataloader = DataLoader(CustomDataset(SENTENCES), batch_size=2, shuffle=False, collate_fn=collate_fn)

    for i, batch in enumerate(dataloader):
        words = [[vocab.get_itos()[idx] for idx in row] for row in batch.tolist()]
        print(f"batch {i}: shape={tuple(batch.shape)}")
        for row in words:
            print("   ", row)


if __name__ == "__main__":
    print("Padding with a real <pad> token (id != 0) from inside a collate function.")
    solution()
