
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data._utils import collate
import torch.nn.utils.rnn as rnn_utils

import os
from typing import List, Callable, Optional, Any, Union, Tuple
from abc import ABCMeta, abstractmethod


class Vocab():
    def __init__(self):
        self.token2idx = dict()
        self.idx2token = list()
    
    def build_from_twolist(self, tokens, idcs):
        self.token2idx = dict(*zip(tokens, idcs))
        self.idx2token = idcs
    
    def build_from_dict(self, token_dct):
        self.token2idx = token_dct
        self.idx2token = list(token_dct.keys())
    
    def get_token_idx(self, token):
        return self.token2idx[token]
    
    def get_token_from_idx(self, idx):
        return self.idx2token[idx]


def train_val_split(train_dataset: Dataset, val_ratio: float, shuffle: bool = True) -> List: # Tuple[Subset] actually
    size = len(train_dataset)  # type: ignore
    val_size = int(size * val_ratio)
    train_size = size - val_size
    if shuffle:
        return random_split(train_dataset, (train_size, val_size), None)
    else:
        return [train_dataset[:train_size], train_dataset[train_size:]]


class Collator():
    def __init__(self):
        __metaclass__ = ABCMeta
    
    @abstractmethod
    def _collate_fn(self, batch):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, batch):
        return self._collate_fn(batch)


class SortPadCollator(Collator):
    def __init__(self, sort_key: Callable, ignore_indics: Union[int, list] = [], reverse: bool = True):
        self.sort_key = sort_key
        self.ignore_indics = ignore_indics
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

        if self.ignore_indics is None:
            self.ignore_indics = []
        elif isinstance(self.ignore_indics, int):
            self.ignore_indics = [self.ignore_indics]

        ret = []
        for i, samples in enumerate(zip(*batch)):
            if i in self.ignore_indics:
                ret.append(torch.tensor(samples))
                continue
            samples = rnn_utils.pad_sequence(samples, batch_first=True, padding_value=0)  # padding
            ret.append(samples)
        return ret

    def __call__(self, batch):
        return self._collate_fn(batch)
