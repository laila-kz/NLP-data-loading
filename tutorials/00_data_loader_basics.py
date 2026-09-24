"""00 — What is a DataLoader?

A DataLoader is the core PyTorch component that efficiently feeds data into a
model during training and evaluation. Instead of loading a whole dataset at
once it:

- splits data into batches
- optionally shuffles the samples
- can load data in parallel (num_workers)
- returns an iterator that yields one batch at a time

This walkthrough uses ``nlp_loaders.CustomDataset`` (a plain ``Dataset`` that
stores a list of sentences) to see the three properties above in action.

Run:  python tutorials/00_data_loader_basics.py
"""

from torch.utils.data import DataLoader

from nlp_loaders import CustomDataset

SENTENCES = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!",
]

# 1. Batching: group samples into mini-batches of `batch_size`.
custom_dataset = CustomDataset(SENTENCES)
dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=True)

# 2. Shuffling: iteration order should be different across runs.
print("Batches (shuffle=True):")
for batch in dataloader:
    print("  ", batch)
print()

# 3. Deterministic order when shuffle=False.
ordered_dataloader = DataLoader(custom_dataset, batch_size=2, shuffle=False)
print("Batches (shuffle=False):")
for batch in ordered_dataloader:
    print("  ", batch)
print()

"""
Why DataLoaders matter in NLP

| Domain          | Data type   | Main challenge    | DataLoader role           |
|-----------------|-------------|-------------------|---------------------------|
| Computer Vision | Images      | Large file sizes  | Load & transform images   |
| NLP             | Text        | Variable length   | Tokenize & pad            |
| Generative AI   | Text/Images | Huge datasets     | Efficient streaming       |
| Tabular         | Structured  | Simplicity        | Basic batching            |

For NLP the variable-length problem is handled with custom Dataset classes
(next step) and custom collate functions (step 02/03).
"""
