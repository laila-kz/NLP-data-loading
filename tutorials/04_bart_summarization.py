"""04 — Using DataLoaders for generative AI (summarization).

Sequence-to-sequence models (BART, T5, ...) also rely on Datasets and
DataLoaders. ``nlp_loaders.SummarizationDataset`` pre-tokenizes each text with
a Hugging Face tokenizer (truncation to a max length, padding to equal
lengths), and ``summarize_texts`` feeds batches through a pretrained BART.

Note: this downloads facebook/bart-large-cnn from the Hugging Face Hub on
first run, so it needs network access (and preferably a HF_TOKEN for the rate
limit).

Run:  python tutorials/04_bart_summarization.py
"""

from nlp_loaders import summarize_texts

SAMPLE_TEXTS = [
    "Machine learning is a subset of artificial intelligence that focuses on building "
    "systems that can learn from and make decisions based on data. These systems improve "
    "their performance over time without being explicitly programmed for every task. "
    "Machine learning is widely used in applications such as recommendation systems, "
    "fraud detection, and image recognition.",
    "Deep learning is a specialized area within machine learning that uses neural networks "
    "with many layers. These deep neural networks are capable of learning complex patterns "
    "in large datasets. They are commonly used in tasks such as speech recognition, "
    "computer vision, and natural language processing.",
]

summaries = summarize_texts(SAMPLE_TEXTS, model_name="facebook/bart-large-cnn", batch_size=2, summary_length=30)
for i, summary in enumerate(summaries):
    print(f"Text {i + 1} summary: {summary}")
