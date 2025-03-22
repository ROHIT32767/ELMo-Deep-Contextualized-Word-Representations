import numpy as np
from collections import Counter
import contractions
import string
import re
import torch
import sys
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from ELMO import Elmo
from classification import LSTMClassifier, function

class Create_dataset_classification(Dataset):
    def __init__(self, sentences, word2idx, idx2word):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.sentences = sentences
        self.max_length = None
        self.X = None
        self.preprocess_data()
        self.get_max_length()
        self.padding()
        self.create_training_data()

    def preprocess_data(self):
        text = ' '.join(self.sentences)
        text = contractions.fix(text)
        text = text.lower()
        text = re.sub(r'http\S+', 'URL', text)
        text = re.sub(r'www\S+', 'URL', text)
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        words = text.split()
        self.sentences = [['<s>'] + words + ['</s>']]

    def get_max_length(self):
        self.max_length = len(self.sentences[0])

    def padding(self):
        sentence = self.sentences[0]
        padded_sentence = [self.word2idx.get(word, self.word2idx['<unk>']) for word in sentence]
        if len(padded_sentence) < self.max_length:
            padded_sentence += [self.word2idx['<pad>']] * (self.max_length - len(padded_sentence))
        self.sentences = [padded_sentence]

    def create_training_data(self):
        self.X = torch.tensor(self.sentences)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, device, method, activation='relu'):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.bidirectional = bidirectional
        self.method = method
        if self.method == '1':
            self.lamda1 = nn.Parameter(torch.randn(1), requires_grad=True)
            self.lamda2 = nn.Parameter(torch.randn(1), requires_grad=True)
            self.lamda3 = nn.Parameter(torch.randn(1), requires_grad=True)
        elif self.method == '2':
            self.lamda1 = nn.Parameter(torch.randn(1), requires_grad=False)
            self.lamda2 = nn.Parameter(torch.randn(1), requires_grad=False)
            self.lamda3 = nn.Parameter(torch.randn(1), requires_grad=False)
        else:
            self.func = function(input_dim*3, input_dim)

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.activation = nn.ReLU() if activation == 'relu' else nn.Tanh()

    def forward(self, e_0, h_0, h_1):
        x = self.lamda1 * e_0 + self.lamda2 * h_0 + self.lamda3 * h_1 if self.method in ['1', '2'] else self.func(e_0, h_0, h_1)
        h0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.n_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_dim, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.activation(out[:, -1, :])
        return self.fc(out)

def main():
    if len(sys.argv) != 2:
        print("Usage: python inference.py <saved model path>")
        sys.exit(1)

    model_path = sys.argv[1]
    device = 'cpu'

    word2idx = torch.load('word2idx.pt', map_location=torch.device(device))
    idx2word = torch.load('idx2word.pt', map_location=torch.device(device))
    elmo_model = torch.load('model.pt', weights_only=False, map_location=torch.device(device))
    classifier_model = torch.load(model_path, weights_only=False, map_location=torch.device(device))

    while True:
        description = input("Enter description (or 'exit' to quit): ")
        if description.lower() == 'exit':
            break

        print("Description recieved",description)

        dataset = Create_dataset_classification([description], word2idx, idx2word)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        classifier_model.eval()
        elmo_model.eval()
        with torch.no_grad():
            for X in data_loader:
                X = X.to(device)
                X_flip = torch.flip(X, [1])
                e_f = elmo_model.embedding(X)
                e_b = elmo_model.embedding(X_flip)
                forward_lstm1, _ = elmo_model.lstm_forward1(e_f)
                backward_lstm1, _ = elmo_model.lstm_backward1(e_b)
                forward_lstm2, _ = elmo_model.lstm_forward2(forward_lstm1)
                backward_lstm2, _ = elmo_model.lstm_backward2(backward_lstm1)
                h_0 = torch.cat((forward_lstm1, backward_lstm1), dim=2)
                h_1 = torch.cat((forward_lstm2, backward_lstm2), dim=2)
                e_0 = torch.cat((e_f, e_b), dim=2)
                outputs = classifier_model(e_0, h_0, h_1)
                probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

        for i, prob in enumerate(probabilities):
            print(f'class-{i+1} {prob:.4f}')

if __name__ == "__main__":
    main()
