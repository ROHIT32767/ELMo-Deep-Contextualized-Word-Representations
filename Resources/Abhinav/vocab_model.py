import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict, Counter
from scipy.sparse.linalg import svds
import os

import nltk
nltk.download('brown')

class VocabModel:
    def __init__(self):
        self.vocab = None
        self.word_to_idx = {}
        self.idx_to_word = {}

    def build_vocab(self, sentences, min_freq=3):
        """Build vocabulary from sentences."""
        word_counts = Counter(word for sentence in sentences for word in sentence)
        self.vocab = [word for word, count in word_counts.items() if count >= min_freq]
        self.vocab.append('<unk>')
        self.vocab.append('<pad>')
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

    def save_checkpoint(self, filename):
        """Save embeddings to a .pt file."""
        checkpoint = {
            "vocab": self.vocab,
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
        }
        torch.save(checkpoint, filename)

    def load_from_checkpoint(self, checkpoint):
        # print(checkpoint)
        self.vocab = checkpoint["vocab"]
        self.word_to_idx = checkpoint["word_to_idx"]
        self.idx_to_word = checkpoint["idx_to_word"]
        
    def sentences_w_to_i(self, sentences):
        # Transform from word to indexes.
        return [
            [
                self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in sentence
            ] for sentence in sentences
        ]
        
    def sentences_i_to_w(self, sentences):
        # Transform from indexes to words.
        return [
            [
                self.idx_to_word[idx] for idx in sentence
            ] for sentence in sentences
        ]