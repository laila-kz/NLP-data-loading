"""01 — From text to tensors.

A ``Dataset`` is an object holding your data and knowing, for any index ``i``,
how to produce sample ``i``. A naive dataset returns the raw string; a smarter
one pre-processes it on the fly.

``nlp_loaders.TextDataset`` does exactly that: give it a tokenizer and a vocab
and each call to ``dataset[i]`` returns a tensor of token ids instead of the
raw sentence.

The mapping is:  "Hello NLP"  --tokenize-->  ["hello", "nlp"]  --vocab-->  tensor([5, 3])

Run:  python tutorials/01_tokenize_to_tensors.py
"""

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from nlp_loaders import TextDataset

SENTENCES = [
    "If you want to know what a man's like.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry.",
    "You are awesome!",
]

# 1. Tokenizer: splits a string into a list of tokens.
tokenizer = get_tokenizer("basic_english")
print("tokenizer sample:", tokenizer("Hello NLP world!"))

# 2. Vocab: builds the token -> id mapping from the corpus.
vocab = build_vocab_from_iterator(map(tokenizer, SENTENCES))
print("vocab size:", len(vocab))

# 3. Dataset with tokenization enabled: __getitem__ returns a tensor.
dataset = TextDataset(SENTENCES, tokenizer=tokenizer, vocab=vocab)
print("dataset length:", len(dataset))
for i in range(len(dataset)):
    tensor = dataset[i]
    words = [vocab.get_itos()[idx] for idx in tensor.tolist()]
    print(f"  item {i}: ids={tensor.tolist()}\n          words={words}")

# 4. Without tokenizer/vocab the dataset returns the raw text (good for a
#    first pass, or when you want to transform inside a collate function).
raw_dataset = TextDataset(SENTENCES)
print("raw sample:", raw_dataset[0])
