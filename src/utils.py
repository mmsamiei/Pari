import json
import random
import os.path
import torch

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

def get_input_output_tensor(pairs, seq_len):
    pass

def get_one_hot_input_tensor(input_tensor):
    pass



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