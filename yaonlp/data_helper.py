
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

import os


class MyDataset(Dataset):
    def __init__(self, data_path, labels_path, vocab_path, max_length=None):
        self.data_path = data_path
        self.labels_path = labels_path

        self.vocab_path = vocab_path
        self.vocab = self.read_vocab(self.vocab_path)
        self.vocab_size = len(self.vocab)

        self.max_length = max_length

        self.data = self.read_data(self.data_path)
        self.labels = self.read_labels(self.labels_path)

    def read_vocab(self, vocab_path):
        vocab = {}
        with open(vocab_path, "r") as f:
            cnt = 1
            for line in f.readlines():
                word = line.split()[0]
                vocab[word] = cnt

                cnt += 1
        return vocab

    def read_data(self, data_path):
        with open(data_path, "r") as f:
            if self.max_length:
                max_len = self.max_length
            else:
                max_len = max(len(line.split()) for line in f.readlines())
            tokens_lst = []
            for line in f.readlines():
                tokens = np.zeros(self.max_length, dtype=np.int32)
                for i,word in enumerate(line.split()):
                    try:
                        tokens[i] = self.vocab[word]
                    # OOV
                    except KeyError:
                        tokens[i] = 0
                tokens_lst.append(tokens)
        return torch.tensor(tokens_lst)

    def read_labels(self, labels_path):
        labels = []
        with open(labels_path, "r") as f:
            for line in f.readlines():
                labels.append(int(line))
        return torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MyIterableDataset(IterableDataset):
    def __init__(self, data_path, labels_path, vocab_path, max_length):
        self.data_path = data_path
        self.labels_path = labels_path

        self.vocab_path = vocab_path
        self.vocab = self.read_vocab(self.vocab_path)
        self.vocab_size = len(self.vocab)

        self.max_length = max_length
    
    def read_vocab(self, vocab_path):
        vocab = {}
        with open(vocab_path, "r") as f:
            cnt = 0
            for line in f.readlines():
                word = line.split()[0]
                vocab[word] = cnt

                cnt += 1
        return vocab

    # process each line
    def process(self, item):
        doc, label = item
        tokens = np.zeros(self.max_length, dtype=np.int32)
        for i, word in enumerate(doc.split()):
            tokens[i] = self.vocab[word]
        return torch.tensor(tokens), torch.tensor(int(label))
    
    # overriade __iter__() func
    def __iter__(self):

        # open() return an iterator
        data_itr = open(self.data_path, 'r', encoding='utf-8')
        labels_itr = open(self.labels_path, 'r', encoding='utf-8')

        # map to each item in iterator
        mapped_itr = map(self.process, zip(data_itr, labels_itr))
        
        return mapped_itr


if __name__ == "__main__":
    import config_loader

    config = config_loader._load_json("config_example/data.json")

    train_path = config["train_data_path"]
    test_path = config["test_data_path"]

    vocab_path = config["vocab_path"]

    train_data_path = config["train_data_path"]
    test_data_path = config["test_data_path"]

    train_labels_path = config["train_labels_path"]
    test_labels_path = config["test_labels_path"]

    shuffle = config["shuffle"]
    batch_size = config["batch_size"]
    max_len = config["max_len"]

    train_dataset = MyDataset(train_data_path, train_labels_path, vocab_path, max_len)
    test_dataset = MyDataset(test_data_path, test_labels_path, vocab_path, max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    for d in train_dataloader:
        print(d)
        break