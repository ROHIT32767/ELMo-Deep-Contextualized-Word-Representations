from utils import create_dataset, Elmo, train_elmo
from torch.utils.data import DataLoader
import torch

data_path = 'data/train.csv'
threshold = 3
dataset = create_dataset(data_path, threshold)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

test_data_path = 'data/test.csv'
test_dataset = create_dataset(test_data_path, threshold, vocab=dataset.vocab, word2idx=dataset.word2idx, idx2word=dataset.idx2word)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

vocab_size = len(dataset.word2idx)
embedding_dim = 150
hidden_dim = 150
model = Elmo(vocab_size, embedding_dim, hidden_dim)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
losses, model = train_elmo(model, train_loader, device, vocab_size, epochs=10)

torch.save(model, 'model.pt')

torch.save(dataset.word2idx, 'word2idx.pt')
torch.save(dataset.idx2word, 'idx2word.pt')
