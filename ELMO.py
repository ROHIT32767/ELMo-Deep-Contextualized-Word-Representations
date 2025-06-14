# -*- coding: utf-8 -*-
"""ELMO_result.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1HL9xnvAFwmEza5Vyghy5IbMEnV1p6-oW
"""

import numpy as np
from collections import Counter
# !pip3 install contractions
import contractions
import string
import re
import torch
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
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

class create_dataset(Dataset):
    def __init__(self, data_path, threshold=3, vocab_model=None):
        self.data_path = data_path
        self.threshold = threshold
        self.sentences = None
        self.max_length = None
        self.X_forward = None
        self.X_backward = None
        self.y_forward = None
        self.y_backward = None
        self.preprocess_data()
        if vocab_model is None:
            self.vocab_model = VocabModel()
            self.vocab_model.build_vocab(self.sentences)
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

class Elmo(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Elmo, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_forward1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_forward2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm_backward1 = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.lstm_backward2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.fc_forward = nn.Linear(hidden_dim, vocab_size)
        self.fc_backward = nn.Linear(hidden_dim, vocab_size)

    def forward(self, X_forward, X_backward):
        forward_embedding = self.embedding(X_forward)
        backward_embedding = self.embedding(X_backward)
        forward_lstm1, _ = self.lstm_forward1(forward_embedding)
        backward_lstm1, _ = self.lstm_backward1(backward_embedding)
        forward_lstm2, _ = self.lstm_forward2(forward_lstm1)
        backward_lstm2, _ = self.lstm_backward2(backward_lstm1)
        forward_output = self.fc_forward(forward_lstm2)
        backward_output = self.fc_backward(backward_lstm2)
        return forward_output, backward_output


def train_elmo(model, train_loader, device, vocab_size, epochs=10):
    model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            X_forward, X_backward, y_forward, y_backward = data
            X_forward, X_backward, y_forward, y_backward = X_forward.to(device), X_backward.to(device), y_forward.to(device), y_backward.to(device)
            optimizer.zero_grad()
            forward_output, backward_output = model(X_forward, X_backward)
            y_forward_one_hot = torch.nn.functional.one_hot(y_forward, num_classes=vocab_size).float()
            y_backward_one_hot = torch.nn.functional.one_hot(y_backward, num_classes=vocab_size).float()
            forward_output = forward_output.permute(0, 2, 1)
            backward_output = backward_output.permute(0, 2, 1)
            y_forward_one_hot = y_forward_one_hot.permute(0, 2, 1)
            y_backward_one_hot = y_backward_one_hot.permute(0, 2, 1)
            forward_loss = criterion(forward_output, y_forward_one_hot)
            backward_loss = criterion(backward_output, y_backward_one_hot)
            loss = forward_loss + backward_loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss/len(train_loader))
        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

    return losses, model

# data_path = 'data/train.csv'
# threshold = 3
# dataset = create_dataset(data_path, threshold)
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# test_data_path = 'data/test.csv'
# test_dataset = create_dataset(test_data_path, threshold, vocab=dataset.vocab, word2idx=dataset.word2idx, idx2word=dataset.idx2word)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# vocab_size = len(dataset.word2idx)
# embedding_dim = 150
# hidden_dim = 150
# model = Elmo(vocab_size, embedding_dim, hidden_dim)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# losses, model = train_elmo(model, train_loader, device, vocab_size, epochs=10)

# torch.save(model, 'model.pt')

# torch.save(dataset.word2idx, 'word2idx.pt')
# torch.save(dataset.idx2word, 'idx2word.pt')