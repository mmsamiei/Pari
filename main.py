from models.PariGRU import *
from src.utils import *
from trainer.trainer import *
batch_sz = 64

ghazals = load_dataset("./data/mesras.json")
pairs = (make_pairs(ghazals))
train, test = split_train_test(pairs)
char_dict, char_list = make_charcabulary(train)

pairs, max_mesra_len = padding_chars_to_max(pairs)

input_tensor, output_tensor = get_input_output_tensor(pairs, max_mesra_len, char_dict)


train_dataset = MyDataSet(input_tensor, output_tensor)
train_dataloader = DataLoader(train_dataset, batch_size=batch_sz, drop_last=True, shuffle=True)

validation_dataset = MyDataSet(input_tensor[:100], output_tensor[:100])
validation_dataloader = DataLoader(train_dataset, batch_size=batch_sz, drop_last=True, shuffle=True)


vocab_size = len(char_list)
embedding_dim = 10
hidden_units = 200
seq_len = max_mesra_len


dev = torch.device("cpu")
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = PariGRUEncoder(vocab_size, embedding_dim, hidden_units, batch_sz)
decoder = PariGRUDecoder(hidden_units, embedding_dim, vocab_size)
seq2seq = PariSeq2Seq(encoder, decoder, dev).to(dev)


trainer = Trainer(seq2seq, train_dataloader, validation_dataloader, char_dict['P'], dev)
trainer.init_weights()
trainer.count_parameters()
trainer.train(1000)


for (x_batch, y_batch, x_len_batch) in train_dataloader:
     x_batch_last = x_batch.permute(1,0)
     res = seq2seq.generate(x_batch_last)
     indices = res.max(2)[1]
     indices = indices.permute(1,0)
     for seq in indices:
         str = ""
         for item in seq:
             str = str + char_list[item]
         print(str)



# for (x_batch, y_batch, x_len_batch) in train_dataloader:
#     x_batch_last = x_batch.permute(1,0)
#     y_batch_last = x_batch.permute(1, 0)
#     res = seq2seq(x_batch_last, y_batch_last)
#     print(res.shape)