
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data._utils import collate
import torch.nn.utils.rnn as rnn_utils

import os
from typing import List, Callable, Optional, Any


# class MyDataLoader(DataLoader):
#     def __init__(self, dataset, config, collate_fn: Callable[[List[Any]], Any] = collate.default_collate) -> None:
#         super(MyDataLoader, self).__init__(
#             dataset, 
#             batch_size=config["batch_size"], 
#             shuffle=config["shuffle"],
#             collate_fn=collate_fn)
#         # self.data_size = len(self.dataset)


def train_val_split(train_dataset: Dataset, val_ratio: float) -> List: # List[Subset] actually
    size = len(train_dataset)
    val_size = int(size * val_ratio)
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
