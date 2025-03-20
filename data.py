import numpy as np
from collections import Counter
!pip3 install contractions
import contractions
import string
import re
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn

class create_dataset(Dataset):
    def __init__(self, data_path, threshold=3, vocab=None, word2idx=None, idx2word=None):
        self.data_path = data_path
        self.threshold = threshold
        self.sentences = None
        self.vocab = vocab
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.max_length = None
        self.X_forward = None
        self.X_backward = None
        self.y_forward = None
        self.y_backward = None
        self.preprocess_data()
        if vocab is None:
            self.create_vocab()
        self.get_max_length()
        self.padding()
        self.create_training_data()

    def preprocess_data(self):
        data = pd.read_csv(self.data_path)
        sentences = data["Description"].values
        sentences = [contractions.fix(sentence) for sentence in sentences]
        sentences = [sentence.lower() for sentence in sentences]
        sentences = [re.sub(r'http\S+', 'URL', sentence) for sentence in sentences]
        sentences = [re.sub(r'www\S+', 'URL', sentence) for sentence in sentences]
        sentences = [sentence.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))) for sentence in sentences]
        sentences = [(sentence.split()) for sentence in sentences]
        sentences = [['<s>'] + sentence + ['</s>'] for sentence in sentences]
        self.sentences = sentences

    def create_vocab(self):
        words = [word for sentence in self.sentences for word in sentence]
        word_freq = Counter(words)
        vocab = [word for word, freq in word_freq.items() if freq >= self.threshold]
        vocab = ['<pad>', '<unk>'] + vocab
        self.vocab = vocab
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def get_max_length(self):
        self.max_length = int(self.get_n_percentile_sentence_length(95))

    def get_n_percentile_sentence_length(self, percentile):
        sentence_lengths = [len(sentence) for sentence in self.sentences]
        return np.percentile(sentence_lengths, percentile)

    def padding(self):
        padded_sentences = []
        for sentence in self.sentences:
            padded_sentence = [self.word2idx[word] if word in self.word2idx else self.word2idx['<unk>'] for word in sentence]
            if len(padded_sentence) < self.max_length:
                padded_sentence += [self.word2idx['<pad>']] * int(self.max_length - len(padded_sentence))
                padded_sentences.append(padded_sentence)
            else:
                padded_sentences.append(padded_sentence[:self.max_length])

        self.sentences = padded_sentences

    def create_training_data(self):
        X_forward = []
        X_backward = []
        y_forward = []
        y_backward = []
        for sentence in self.sentences:
            X_forward.append(sentence[:-1])
            X_backward.append(sentence[::-1][:-1])
            y_forward.append(sentence[1:])
            y_backward.append(sentence[::-1][1:])

        self.X_forward = torch.tensor(X_forward)
        self.X_backward = torch.tensor(X_backward)
        self.y_forward = torch.tensor(y_forward)
        self.y_backward = torch.tensor(y_backward)

    def __len__(self):
        return len(self.X_forward)
    
    def __getitem__(self, idx):
        return self.X_forward[idx], self.X_backward[idx], self.y_forward[idx], self.y_backward[idx]


from torch.utils.data import DataLoader
import torch

data_path = 'data/train.csv'
threshold = 3
dataset = create_dataset(data_path, threshold)
torch.save(dataset.word2idx, 'word2idx.pt')
torch.save(dataset.idx2word, 'idx2word.pt')