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
