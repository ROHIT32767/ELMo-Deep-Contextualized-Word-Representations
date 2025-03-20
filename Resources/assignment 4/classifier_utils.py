import numpy as np
import contractions
import string
import re
import torch
import pandas as pd
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

class Create_dataset_classification(Dataset):
    def __init__(self, data_path, word2idx, idx2word):
        self.data_path = data_path
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.sentences = None
        self.labels = None
        self.num_classes = None
        self.max_length = None
        self.X = None
        self.Y = None
        self.preprocess_data()
        self.get_max_length()
        self.padding()
        self.create_training_data()

    def preprocess_data(self):
        data = pd.read_csv(self.data_path)
        sentences = data["Description"].values
        self.labels = data["Class Index"].values
        self.labels = [label - 1 for label in self.labels]
        self.num_classes = len(set(self.labels))
        self.labels = torch.nn.functional.one_hot(torch.tensor(self.labels), num_classes=self.num_classes).float()
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
        X = []
        for sentence in self.sentences:
            X.append(sentence)

        self.X = torch.tensor(X)
        self.Y = self.labels

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    
class function(nn.Module):
    def __init__(self, input_dim,output_dim, activation='relu'):
        super(function, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
    def forward(self, e_0, h_0, h_1):
        x = torch.cat((e_0, h_0, h_1), dim=2)
        x = self.fc1(x)
        x = self.activation(x)
        return x


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, device,method, activation='relu'):
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
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, e_0, h_0, h_1):
        if self.method == '1':
            x = self.lamda1 * e_0 + self.lamda2 * h_0 + self.lamda3 * h_1
        elif self.method == '2':
            x = self.lamda1 * e_0 + self.lamda2 * h_0 + self.lamda3 * h_1
        else:
            x = self.func(e_0, h_0, h_1)
        h0 = torch.zeros(self.n_layers * 2 if self.bidirectional else 1, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers * 2 if self.bidirectional else 1, x.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.activation(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
def train_classifier(model, elmo_model, train_loader,val_loader, device, lr, epochs=10):
    model.to(device)
    elmo_model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    val_losses = []
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            X, y = data
            X, y = X.to(device), y.to(device)
            X_flip = torch.flip(X, [1])
            e_f = elmo_model.embedding(X)
            e_b = elmo_model.embedding(X_flip)
            forward_lstm1,_ = elmo_model.lstm_forward1(e_f)
            backward_lstm1,_ = elmo_model.lstm_backward1(e_b)
            forward_lstm2,_ = elmo_model.lstm_forward2(forward_lstm1)
            backward_lstm2,_ = elmo_model.lstm_backward2(backward_lstm1)
            h_0 = torch.cat((forward_lstm1, backward_lstm1), dim=2)
            h_1 = torch.cat((forward_lstm2, backward_lstm2), dim=2)
            e_0 = torch.cat((e_f, e_b), dim=2)
            y_pred = model(e_0, h_0, h_1)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss/len(train_loader))

        val_running_loss = 0.0
        for i, data in enumerate(val_loader):
            X, y = data
            X, y = X.to(device), y.to(device)
            X_flip = torch.flip(X, [1])
            e_f = elmo_model.embedding(X)
            e_b = elmo_model.embedding(X_flip)
            forward_lstm1,_ = elmo_model.lstm_forward1(e_f)
            backward_lstm1,_ = elmo_model.lstm_backward1(e_b)
            forward_lstm2,_ = elmo_model.lstm_forward2(forward_lstm1)
            backward_lstm2,_ = elmo_model.lstm_backward2(backward_lstm1)
            h_0 = torch.cat((forward_lstm1, backward_lstm1), dim=2)
            h_1 = torch.cat((forward_lstm2, backward_lstm2), dim=2)
            e_0 = torch.cat((e_f, e_b), dim=2)
            y_pred = model(e_0, h_0, h_1)
            loss = criterion(y_pred, y)
            val_running_loss += loss.item()

        val_losses.append(val_running_loss/len(val_loader))

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}, Val Loss: {val_running_loss/len(val_loader)}')

    return losses, val_losses, model

def get_predictions(model, elmomodel, data_loader, device):
    predictions = []
    ground_truth = []
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs_flip = torch.flip(inputs, [1])
        e_f = elmomodel.embedding(inputs)
        e_b = elmomodel.embedding(inputs_flip)
        forward_lstm1,_ = elmomodel.lstm_forward1(e_f)
        backward_lstm1,_ = elmomodel.lstm_backward1(e_b)
        forward_lstm2,_ = elmomodel.lstm_forward2(forward_lstm1)
        backward_lstm2,_ = elmomodel.lstm_backward2(backward_lstm1)
        h_0 = torch.cat((forward_lstm1, backward_lstm1), dim=2)
        h_1 = torch.cat((forward_lstm2, backward_lstm2), dim=2)
        e_0 = torch.cat((e_f, e_b), dim=2)
        outputs = model(e_0, h_0, h_1)
        predictions.extend(outputs.argmax(dim=1).cpu().numpy())
        ground_truth.extend(targets.argmax(dim=1).cpu().numpy())
    return predictions, ground_truth

def get_metrics(predictions, ground_truth):
    accuracy = accuracy_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions, average='weighted')
    precision = precision_score(ground_truth, predictions, average='weighted')
    recall = recall_score(ground_truth, predictions, average='weighted')
    cm = confusion_matrix(ground_truth, predictions)
    return accuracy, f1, precision, recall, cm