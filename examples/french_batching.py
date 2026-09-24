"""French sentence batching with a tokenizer-based collate function.

Demonstrates ``nlp_loaders.collate.make_tokenized_collate``: tokenize raw
strings, map tokens through a vocab, and pad every batch to equal length.

Note: requires the spacy French model (``python -m spacy download fr_core_news_sm``).
"""

from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from nlp_loaders.collate import make_tokenized_collate

corpus = [
    "Ceci est une phrase.",
    "C'est un autre exemple de phrase.",
    "Voici une troisième phrase.",
    "Il fait beau aujourd'hui.",
    "J'aime beaucoup la cuisine française.",
    "Quel est ton plat préféré ?",
    "Je t'adore.",
    "Bon appétit !",
    "Je suis en train d'apprendre le français.",
    "Nous devons partir tôt demain matin.",
    "Je suis heureux.",
    "Le film était vraiment captivant !",
    "Je suis là.",
    "Je ne sais pas.",
    "Je suis fatigué après une longue journée de travail.",
    "Est-ce que tu as des projets pour le week-end ?",
    "Je vais chez le médecin cet après-midi.",
    "La musique adoucit les mœurs.",
    "Je dois acheter du pain et du lait.",
    "Il y a beaucoup de monde dans cette ville.",
    "Merci beaucoup !",
    "Au revoir !",
    "Je suis ravi de vous rencontrer enfin !",
    "Les vacances sont toujours trop courtes.",
    "Je suis en retard.",
    "Félicitations pour ton nouveau travail !",
    "Je suis désolé, je ne peux pas venir à la réunion.",
    "À quelle heure est le prochain train ?",
    "Bonjour !",
    "C'est génial !",
]

# Tokenizer (torchtext wraps a spacy French model)
tokenizer = get_tokenizer("spacy", language="fr_core_news_sm")

# Build vocabulary from the corpus, reserving <unk> and <pad>
vocab = build_vocab_from_iterator(
    map(tokenizer, corpus),
    specials=["<unk>", "<pad>"],
    special_first=True,
)
UNK_IDX = vocab["<unk>"]
PAD_IDX = vocab["<pad>"]
vocab.set_default_index(UNK_IDX)

# Sort by length so similar-length sentences are batched together (fewer pads)
sorted_data = sorted(corpus, key=lambda x: len(tokenizer(x)))

dataloader = DataLoader(
    sorted_data,
    batch_size=4,
    shuffle=False,
    collate_fn=make_tokenized_collate(tokenizer, vocab, padding_value=PAD_IDX),
)

for batch in dataloader:
    print(batch)
