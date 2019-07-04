import json
import random
import os.path
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

def load_dataset(file_addr):
    with open(file_addr, "r") as read_file:
        data = json.load(read_file)
        return data

def split_train_test(data, a=0.9):
    temp = data
    random.shuffle(temp)
    train = temp[:int(a*len(temp))]
    test = temp[int(a*len(temp)):]
    return train, test

def make_pairs(ghazals):
    pairs = []
    for ghazal in ghazals:
        ghazlan_len = len(ghazal['ghazal'])
        for i in range(0, int(ghazlan_len/2), 2):
            pair = [None, None]
            pair[0] = ghazal['ghazal'][i]
            pair[1] = ghazal['ghazal'][i+1]
            pairs.append(pair)
    return pairs

def make_vocabulary(pairs):
    vocabs = {""}
    for pair in pairs:
        tokens = pair[0].split() + pair[1].split()
        for token in tokens:
            if token not in vocabs:
                vocabs.add(token)
    vocab_dict = {k: i for i,k in enumerate(vocabs)}
    vocab_list = [None] * len(vocabs)
    for i,k in enumerate(vocab_dict):
        vocab_list[i] = k
    return vocab_dict, vocab_list

def make_charcabulary(pairs):
    chars = {"$","P"}
    for pair in pairs:
        beyt = pair[0] + pair[1]
        for char in beyt:
            if char not in chars:
                chars.add(char)
    char_dict = {k: i for i, k in enumerate(chars)}
    char_list = [None] * len(chars)
    for i,k in enumerate(char_dict):
        char_list[i] = k
    return char_dict, char_list

def padding_chars_to_max(pairs):
    l = []
    for pair in pairs:
        for mesra in pair:
            l.append(mesra)
    max_mesra_len = max([len(item) for item in l])
    temp = pairs
    for i, pair in enumerate(temp):
            pair[0] = pair[0] + "P" * (max_mesra_len-len(pair[0]))
            pair[1] = pair[1] + "P" * (max_mesra_len - len(pair[1]))
    return temp, max_mesra_len

def get_input_output_tensor(pairs, seq_len, char_dict):
    pairs_num = len(pairs)
    #input_tensor = torch.zeros([pairs_num, seq_len], dtype=torch.int32)
    input_tensor = torch.LongTensor(pairs_num, seq_len)
    #output_tensor = torch.zeros([pairs_num, seq_len], dtype=torch.int32)
    output_tensor = torch.LongTensor(pairs_num, seq_len)
    for i in range(len(input_tensor)):
        mesra_avval = pairs[i][0]
        for j,v in enumerate(mesra_avval):
            input_tensor[i][j] = char_dict[v]
        mesra_dovvom = pairs[i][1]
        for j, v in enumerate(mesra_dovvom):
            output_tensor[i][j] = char_dict[v]
    return input_tensor, output_tensor

def get_one_hot_input_tensor(input_tensor, dict_size):
    it_shape = input_tensor.shape
    one_hot_tensor = torch.zeros([it_shape[0], it_shape[1], dict_size])
    for i in range(it_shape[0]):
        for j in range(it_shape[1]):
            active_place = input_tensor[i][j]
            one_hot_tensor[i][j][active_place] = 1
    return one_hot_tensor

def convert_to_pickle(item, directory):
    pickle.dump(item, open(directory,"wb"))

def load_from_pickle(directory):
    return pickle.load(open(directory,"rb"))


class MyDataSet(Dataset):
    def __init__(self, X, y):
        self.data = X
        self.target = y
        self.length = [torch.sum(1 - np.equal(x, 0)) for x in X]


    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]
        return x, y, x_len

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    ghazals = load_dataset("./data/mesras.json")
    pairs = (make_pairs(ghazals))
    train, test = split_train_test(pairs)
    # vocab_dict, vocab_list = make_vocabulary(train)
    # print(vocab_dict['عشق'])
    # print(vocab_list[100])
    char_dict, char_list = make_charcabulary(train)
    print(char_dict['ع'])
    print(len(char_list))
    temp, max_mesra_len = padding_chars_to_max(pairs)
    print(max_mesra_len)
    input_tensor, output_tensor = get_input_output_tensor(pairs, 43, char_dict)
    #input_one_hot_tensor = get_one_hot_input_tensor(input_tensor, len(char_dict))

    train_dataset = MyDataSet(input_tensor, output_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=64, drop_last=True, shuffle=True)
    for (x_batch, y_batch, x_len_batch) in train_dataloader:
        print(x_batch)

