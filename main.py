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
import nltk
from nltk.corpus import brown

# Download the Brown Corpus if not already downloaded
nltk.download('brown')

class create_dataset(Dataset):
    def __init__(self, threshold=3, vocab=None, word2idx=None, idx2word=None):
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
        # Fetch sentences from the Brown Corpus
        sentences = brown.sents()
        sentences = [contractions.fix(' '.join(sentence)) for sentence in sentences]
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

from torch.utils.data import DataLoader
import torch

threshold = 3
dataset = create_dataset(threshold)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# For testing, you can use a subset of the Brown Corpus or another corpus
test_dataset = create_dataset(threshold, vocab=dataset.vocab, word2idx=dataset.word2idx, idx2word=dataset.idx2word)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

vocab_size = len(dataset.word2idx)
embedding_dim = 150
hidden_dim = 150
model = Elmo(vocab_size, embedding_dim, hidden_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
losses, model = train_elmo(model, train_loader, device, vocab_size, epochs=10)

torch.save(model.state_dict(), 'model.pt')

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


data_path = 'data/train.csv'
threshold = 3
dataset = create_dataset(data_path, threshold)
torch.save(dataset.word2idx, 'word2idx.pt')
torch.save(dataset.idx2word, 'idx2word.pt')


from torch.utils.data import DataLoader, random_split
import torch

data_path = 'data/train.csv'
test_data_path = 'data/test.csv'

word2idx = torch.load('word2idx.pt')
idx2word = torch.load('idx2word.pt')
dataset = Create_dataset_classification(data_path, word2idx, idx2word)
test_dataset = Create_dataset_classification(test_data_path, word2idx, idx2word)

train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMClassifier(input_dim=300, hidden_dim=128, output_dim=dataset.num_classes,n_layers=2,bidirectional=True,device=device, method='3')

elmo_model = Elmo(vocab_size, embedding_dim, hidden_dim).to(device)
elmo_model.load_state_dict(torch.load('model.pt'))
elmo_model.eval()  # Set the model to evaluation mode
loss, val_loss, model = train_classifier(model, elmomodel, train_loader, val_loader, device,0.001,5)
# print("lamda1, lamda2, lamda3:")
# print(model.lamda1, model.lamda2, model.lamda3)
torch.save(model, 'classification_model_3.pt')

train_pred, train_true = get_predictions(model, elmomodel, train_loader, device)
val_pred, val_true = get_predictions(model, elmomodel, val_loader, device)
test_pred, test_true = get_predictions(model, elmomodel, test_loader, device)

accuracy_train, f1_train, precision_train, recall_train, cm_train = get_metrics(train_pred, train_true)
accuracy_val, f1_val, precision_val, recall_val, cm_val = get_metrics(val_pred, val_true)
accuracy_test, f1_test, precision_test, recall_test, cm_test = get_metrics(test_pred, test_true)

print('Train Accuracy: {:.4f}, F1 Score: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format(accuracy_train, f1_train, precision_train, recall_train))
print('Validation Accuracy: {:.4f}, F1 Score: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format(accuracy_val, f1_val, precision_val, recall_val))
print('Test Accuracy: {:.4f}, F1 Score: {:.4f}, Precision: {:.4f}, Recall: {:.4f}'.format(accuracy_test, f1_test, precision_test, recall_test))

print('Train Confusion Matrix:')
print(cm_train)
print('Validation Confusion Matrix:')
print(cm_val)
print('Test Confusion Matrix:')
print(cm_test)