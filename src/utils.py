import json
import random
import os.path

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
    return {k: i for i,k in enumerate(vocabs)}


ghazals = load_dataset("./data/mesras.json")
pairs = (make_pairs(ghazals))
train, test = split_train_test(pairs)
vocab_dict = make_vocabulary(train)
print(vocab_dict['عشق'])