
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

import os

class MyDataset(Dataset):
    def __init__(self, config, train=True):
        if train:
            self.data_path = config.train_data_path
            self.labels_path = config.train_labels_path
        else:
            self.data_path = config.test_data_path
            self.labels_path = config.test_labels_path

        self.vocab_path = config.vocab_path
        self.vocab = self.read_vocab(self.vocab_path)
        self.vocab_size = len(self.vocab)

        self.max_length = config.max_len

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
                tokens = np.zeros(self.max_length, dtype=np.int64)
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
        return list(zip(self.data, self.labels))[idx]


class MyIterableDataset(IterableDataset):
    def __init__(self, config, train=True):
        if train:
            self.data_path = config.train_data_path
        else:
            self.data_path = config.test_data_path

        self.labels_path = config.labels_path

        self.vocab_path = config.vocab_path
        self.vocab = self.read_vocab(self.vocab_path)
        self.vocab_size = len(self.vocab)

        self.max_length = config.max_len
    
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


class MyDataLoader(DataLoader):
    def __init__(self, dataset, config):
        super(MyDataLoader, self).__init__(
            dataset, 
            batch_size=config.batch_size, 
            shuffle=config.shuffle,
        )