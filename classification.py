from classifier_utils import Create_dataset_classification, LSTMClassifier, train_classifier, get_metrics, get_predictions
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

elmomodel = torch.load('model.pt')
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