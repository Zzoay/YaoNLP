
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, random_split
import torch.nn.utils.rnn as rnn_utils

import os
from typing import List, Any, Callable, Optional

from yaonlp.config_loader import Config


class MyDataset(Dataset):
    def __init__(self, config: Config, train=True) -> None:
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

    def read_vocab(self, vocab_file: str) -> dict:
        vocab = {}
        with open(vocab_file, "r") as f:
            cnt = 1
            for line in f.readlines():
                word = line.split()[0]
                vocab[word] = cnt

                cnt += 1
        return vocab

    def read_data(self, data_file: str) -> torch.Tensor:
        with open(data_file, "r") as f:
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

    def read_labels(self, labels_file: str) -> torch.Tensor:
        labels = []
        with open(labels_file, "r") as f:
            for line in f.readlines():
                labels.append(int(line))
        return torch.tensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


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
    def __init__(self, dataset, config, collate_fn) -> None:
        super(MyDataLoader, self).__init__(
            dataset, 
            batch_size=config["batch_size"], 
            shuffle=config["shuffle"],
            collate_fn=collate_fn
        )
        self.data_size = len(self.dataset)


def train_val_split(train_dataset: Dataset, config) -> List: # List[Subset] actually
    size = train_dataset.data_size
    val_size = int(size * config.val_ratio)
    train_size = size - val_size
    return random_split(train_dataset, (train_size, val_size), None)


class SortPadCollator():
    def __init__(self, sort_key: Callable, ignore_index: Optional[int] = None, reverse: bool = True):
        self.sort_key = sort_key
        self.ignore_index = ignore_index
        self.reverse = reverse
    
    def _collate_fn(self, batch):
        if isinstance(batch, list):
            assert self.sort_key, "if batch is a list, sort_key should be provided"  
            """
            param 'key' specifies what sort depends on.
            example: key=lambda x: x[5]; while 5 indicates index of the sequences lenght
                     key=lambda x: len(x); while sequences lenght is not provided
            """
            batch.sort(key=self.sort_key, reverse=self.reverse)  
        elif isinstance(batch, torch.Tensor):
            batch.sort(dim=-1, descending=self.reverse)

        ret = []
        for i, samples in enumerate(zip(*batch)):
            if i == self.ignore_index:
                ret.append(torch.tensor(samples))
                break
            samples = rnn_utils.pad_sequence(samples, batch_first=True, padding_value=0)  # padding
            ret.append(samples)
        return ret

    def __call__(self, batch):
        return self._collate_fn(batch)
