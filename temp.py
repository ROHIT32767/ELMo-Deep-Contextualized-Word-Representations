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
    

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, bidirectional, device):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
        else:
            self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        h0 = torch.zeros(self.n_layers * 2 if self.bidirectional else 1, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers * 2 if self.bidirectional else 1, x.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.activation(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
def check_for_invalid_data(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaN values found in {name}")
    if torch.isinf(tensor).any():
        print(f"Infinite values found in {name}")

def check_indices(X, vocab_size):
    max_index = X.max().item()
    min_index = X.min().item()
    if max_index >= vocab_size or min_index < 0:
        raise ValueError(f"Input contains indices outside the valid range [0, {vocab_size - 1}]. "
                         f"Min index: {min_index}, Max index: {max_index}")

def train_classifier(model, elmo_model, train_loader, val_loader, device, lr, epochs=10):
    model.to(device)
    elmo_model.to(device)
    elmo_model.eval()  # Freeze ELMO model
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

            # Debugging: Check tensor shapes and devices
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")
            print(f"X_flip shape: {X_flip.shape}")
            print(f"Device of X: {X.device}")
            print(f"Device of y: {y.device}")
            print(f"Device of elmo_model: {next(elmo_model.parameters()).device}")
            print(f"Device of model: {next(model.parameters()).device}")

            # Check for out-of-bounds indices
            vocab_size = elmo_model.embedding.num_embeddings
            check_indices(X, vocab_size)

            # Generate ELMO embeddings
            with torch.no_grad():
                e_f = elmo_model.embedding(X)
                e_b = elmo_model.embedding(X_flip)
                forward_lstm1, _ = elmo_model.lstm_forward1(e_f)
                backward_lstm1, _ = elmo_model.lstm_backward1(e_b)
                forward_lstm2, _ = elmo_model.lstm_forward2(forward_lstm1)
                backward_lstm2, _ = elmo_model.lstm_backward2(backward_lstm1)
                h_0 = torch.cat((forward_lstm1, backward_lstm1), dim=2)
                h_1 = torch.cat((forward_lstm2, backward_lstm2), dim=2)
                e_0 = torch.cat((e_f, e_b), dim=2)
                embeddings = torch.cat((e_0, h_0, h_1), dim=2)

                # Debugging: Check for invalid data
                check_for_invalid_data(e_f, "e_f")
                check_for_invalid_data(e_b, "e_b")
                check_for_invalid_data(forward_lstm1, "forward_lstm1")
                check_for_invalid_data(backward_lstm1, "backward_lstm1")
                check_for_invalid_data(forward_lstm2, "forward_lstm2")
                check_for_invalid_data(backward_lstm2, "backward_lstm2")
                check_for_invalid_data(h_0, "h_0")
                check_for_invalid_data(h_1, "h_1")
                check_for_invalid_data(e_0, "e_0")
                check_for_invalid_data(embeddings, "embeddings")

            # Forward pass through the classifier
            y_pred = model(embeddings)
            loss = criterion(y_pred, y.argmax(dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        losses.append(running_loss / len(train_loader))

        # Validation
        val_running_loss = 0.0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                X, y = data
                X, y = X.to(device), y.to(device)
                X_flip = torch.flip(X, [1])

                # Generate ELMO embeddings
                e_f = elmo_model.embedding(X)
                e_b = elmo_model.embedding(X_flip)
                forward_lstm1, _ = elmo_model.lstm_forward1(e_f)
                backward_lstm1, _ = elmo_model.lstm_backward1(e_b)
                forward_lstm2, _ = elmo_model.lstm_forward2(forward_lstm1)
                backward_lstm2, _ = elmo_model.lstm_backward2(backward_lstm1)
                h_0 = torch.cat((forward_lstm1, backward_lstm1), dim=2)
                h_1 = torch.cat((forward_lstm2, backward_lstm2), dim=2)
                e_0 = torch.cat((e_f, e_b), dim=2)
                embeddings = torch.cat((e_0, h_0, h_1), dim=2)

                # Forward pass through the classifier
                y_pred = model(embeddings)
                loss = criterion(y_pred, y.argmax(dim=1))
                val_running_loss += loss.item()
        val_losses.append(val_running_loss / len(val_loader))

        print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_running_loss / len(val_loader)}')
        model.train()

    return losses, val_losses, model


def get_predictions(model, elmo_model, data_loader, device):
    predictions = []
    ground_truth = []
    model.eval()
    elmo_model.eval()
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs_flip = torch.flip(inputs, [1])

            # Generate ELMO embeddings
            e_f = elmo_model.embedding(inputs)
            e_b = elmo_model.embedding(inputs_flip)
            forward_lstm1, _ = elmo_model.lstm_forward1(e_f)
            backward_lstm1, _ = elmo_model.lstm_backward1(e_b)
            forward_lstm2, _ = elmo_model.lstm_forward2(forward_lstm1)
            backward_lstm2, _ = elmo_model.lstm_backward2(backward_lstm1)
            h_0 = torch.cat((forward_lstm1, backward_lstm1), dim=2)
            h_1 = torch.cat((forward_lstm2, backward_lstm2), dim=2)
            e_0 = torch.cat((e_f, e_b), dim=2)
            embeddings = torch.cat((e_0, h_0, h_1), dim=2)

            # Forward pass through the classifier
            outputs = model(embeddings)
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

# Load pre-trained ELMO model
elmo_model = torch.load('model.pt', map_location=device,weights_only=False)
elmo_model.to(device)
elmo_model.eval()  # Freeze ELMO model

# Initialize classifier
model = LSTMClassifier(input_dim=300, hidden_dim=128, output_dim=dataset.num_classes, n_layers=2, bidirectional=True, device=device)

# Train classifier
loss, val_loss, model = train_classifier(model, elmo_model, train_loader, val_loader, device, 0.001, 5)
torch.save(model, 'classification_model.pt')

# Evaluate
train_pred, train_true = get_predictions(model, elmo_model, train_loader, device)
val_pred, val_true = get_predictions(model, elmo_model, val_loader, device)
test_pred, test_true = get_predictions(model, elmo_model, test_loader, device)

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