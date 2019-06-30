from torch import nn
import torch
class PariGRUEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, batch_sz, output_size):
        super(PariGRUEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units =  hidden_units
        self.batch_sz = batch_sz
        self.output_size = output_size
    # layers
        self.embedding = nn.Linear(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_units)
        self.fc = nn.Linear(self.hidden_units, self.output_size)

    def initialize_hidden_state(self, device):
        return torch.zeros((1, self.batch_sz, self.hidden_units)).to(device)

    def forward(self, x, lens, device):
        temp = x    #    (seq_len, batch, input_size)
        temp = self.embedding(temp)
        self.hidden = self.initialize_hidden_state(device)
        output, self.hidden = self.gru(temp, self.hidden)
        print(self.hidden.shape)

class PariGRUDecoder(nn.Module):
    def __init__(self, hidden_dim, emb_dim, vocab_dim):
        super(PariGRUDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim
        self.vocab_dim = vocab_dim
    # layers
        self.embedding = nn.Embedding(vocab_dim, emb_dim)
        self.gru = nn.LSTM(emb_dim, hidden_dim, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, vocab_dim)

    def forward(self, input, hidden):
        temp = input #input = [batch size]
        temp = temp.unsqueeze(0) #temp = [1, batch size]
        print(temp.shape)
        temp = self.embedding(temp)
        print(temp.shape)
        output, hidden = self.gru(temp, hidden)
        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers, batch size, hid dim]
        prediction = self.fc(output.squeeze(0))
        return prediction, hidden




vocab_size = 30
embedding_dim = 10
hidden_units = 200
batch_sz = 64
output_size = 15

seq_len = 40

model = PariGRUEncoder(vocab_size, embedding_dim, hidden_units, batch_sz, output_size)
x = torch.ones([seq_len, batch_sz, vocab_size])
dev = torch.device("cpu")
model.initialize_hidden_state(dev)
temp = model(x, seq_len, dev)

model2 = PariGRUDecoder(hidden_units, embedding_dim, vocab_size)
y = torch.LongTensor(batch_sz).random_(0, vocab_size)
print(y)
print(y.shape)
z = model2(y, temp)[1]
print(z)
