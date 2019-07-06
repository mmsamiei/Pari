from torch import nn
import torch
import random
class PariGRUEncoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units, batch_sz):
        super(PariGRUEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_units =  hidden_units
        self.batch_sz = batch_sz
    # layers
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.dropout = nn.Dropout(p=0.5)
        self.gru = nn.GRU(self.embedding_dim, self.hidden_units)

    def initialize_hidden_state(self, device):
        return torch.zeros((1, self.batch_sz, self.hidden_units)).to(device)

    def forward(self, x, lens, device):
        temp = x    #    (seq_len, batch)
        temp = self.embedding(temp)
        self.hidden = self.initialize_hidden_state(device)
        output, self.hidden = self.gru(temp, self.hidden)
        return self.hidden

class PariGRUDecoder(nn.Module):
    def __init__(self, hidden_dim, emb_dim, vocab_size):
        super(PariGRUDecoder, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

    # layers
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input, hidden):
        #input = [batch_size]
        #hidden = [1, batch_size, hid_dim]

        temp = input.unsqueeze(0) # temp = [1, batch_size]
        temp = self.embedding(temp) # temp = [1, batch_size, emb_dim]
        output, hidden = self.gru(temp, hidden) #output = [1, batch_size, hid_dim]
        prediction = self.fc(output.squeeze(0))

        return prediction, hidden


class PariSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(PariSeq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, trg, teacher_forcing_rate = 0.5):
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_dim = self.decoder.vocab_size
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_dim).to(self.device)
        hidden = self.encoder(src, max_len, self.device)
        input = trg[0, :]
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            teacher_forcing = random.random() < teacher_forcing_rate
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_forcing else top1)
        return outputs

    def generate(self,src):
        batch_size = src.shape[1]
        max_len = src.shape[0]
        trg_vocab_dim = self.decoder.vocab_size
        # tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_dim).to(self.device)
        src = src.to(self.device)
        hidden = self.encoder(src, max_len, self.device)
        input = src[0, :]
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            top1 = output.max(1)[1]
            input = top1
        return outputs


# model = PariGRUEncoder(vocab_size, embedding_dim, hidden_units, batch_sz, output_size)
# x = torch.LongTensor(seq_len, batch_sz).random_(0, vocab_size)
# print(x.shape)
# dev = torch.device("cpu")
# model.initialize_hidden_state(dev)
# temp = model(x, seq_len, dev)

if __name__ == "__main__":
    vocab_size = 30
    embedding_dim = 10
    hidden_units = 200
    batch_sz = 32
    output_size = 15
    seq_len = 45
    dev = torch.device("cuda")
    encoder = PariGRUEncoder(vocab_size, embedding_dim, hidden_units, batch_sz)
    decoder = PariGRUDecoder(hidden_units, embedding_dim, vocab_size)
    seq2seq = PariSeq2Seq(encoder, decoder, dev)
    seq2seq.to(dev)
    x = torch.LongTensor(seq_len, batch_sz).random_(0, vocab_size).to(dev)
    y = torch.LongTensor(seq_len, batch_sz).random_(0, vocab_size).to(dev)
    res = seq2seq(x, y)
