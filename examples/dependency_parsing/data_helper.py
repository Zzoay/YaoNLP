
import os
from typing import List, Union, Tuple, Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from dependency import Dependency


class CTBDataset(Dataset):
    def __init__(self, vocab, config: dict):
        self.words, self.tags, self.heads, self.rels, self.masks, self.seq_lens = self.read_data(vocab, config["data_path"])

    def read_data(self, 
                  vocab, 
                  data_path: str,         # word, tag, head, rel, mask
                  max_len: int = None,
                  ) -> Tuple[List[torch.Tensor], ...]:
        seq_len_lst:List = []

        w_tk_lst:List = []
        t_tk_lst:List = []
        h_tk_lst:List = []
        r_tk_lst:List = []
        m_tk_lst:List = []
        for sentence in load_ctb(data_path):   # sentence: [dep, dep, ...]; dep.attrs: id, word, tag, head, rel
            seq_len = len(sentence)

            word_tokens = np.zeros(seq_len, dtype=np.int64)  # <pad> is 0 default
            tag_tokens = np.zeros(seq_len, dtype=np.int64)
            head_tokens = np.zeros(seq_len, dtype=np.int64) 
            rel_tokens = np.zeros(seq_len, dtype=np.int64)
            mask_tokens = np.zeros(seq_len, dtype=np.int64)
            for i,dep in enumerate(sentence):
                if i == seq_len:
                    break
                word = vocab.word2id.get(dep.word)
                word_tokens[i] =  word or 2  # while OOV set <unk> token
                tag_tokens[i] = (vocab.tag2id.get(dep.tag) if word else 0) or 0  # if there is no word or not tag, set 0

                head_idx = (vocab.head2id.get(dep.head) if word else 0) or 0
                if head_idx < seq_len:  # if idx in bounds, set idx into tokens
                    head_tokens[i] = head_idx

                rel_tokens[i] = (vocab.rel2id.get(dep.rel) if word else 0) or 0
                mask_tokens[i] = 1 if word else 0   # if is there a word, mask = 1, else 0 

            seq_len_lst.append(torch.tensor(seq_len))     

            w_tk_lst.append(torch.tensor(word_tokens))
            t_tk_lst.append(torch.tensor(tag_tokens))
            h_tk_lst.append(torch.tensor(head_tokens))
            r_tk_lst.append(torch.tensor(rel_tokens))
            m_tk_lst.append(torch.tensor(mask_tokens))

        return w_tk_lst, t_tk_lst, h_tk_lst, r_tk_lst, m_tk_lst, seq_len_lst
    
    def __getitem__(self, idx):
        return self.words[idx], self.tags[idx], self.heads[idx], self.rels[idx], self.masks[idx], self.seq_lens[idx]
    
    def __len__(self):
        return len(self.words)


def load_ctb(data_path: str):
    file_names:List[str] = os.listdir(data_path)
    ctb_files:List[str] = [data_path+fle for fle in file_names]

    # id, form, tag, head, rel
    sentence:List[Dependency] = []

    for ctb_file in ctb_files:
        with open(ctb_file, 'r', encoding='utf-8') as f:
            # data example: 1	上海	_	NR	NR	_	2	nn	_	_
            for line in f.readlines():
                toks = line.split()
                if len(toks) == 0:
                    yield sentence
                    sentence = []
                elif len(toks) == 10:
                    dep = Dependency(toks[0], toks[1], toks[3], toks[6], toks[7])
                    sentence.append(dep)


# Inspired by https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256
class Prefetcher():
    def __init__(self, loader, cuda=True):
        self.loader = iter(loader)
        if cuda:
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
        self.preload()

    def preload(self):
        try:
            self.samples = self.loader.next()
        except StopIteration:
            self.samples = None
            return
        if self.stream:
            with torch.cuda.stream(self.stream):
                if isinstance(self.samples, list):
                    self.samples = [item.cuda(non_blocking=True) for item in self.samples]
                elif isinstance(self.samples, torch.Tensor):
                    self.samples = self.samples.cuda(non_blocking=True)
            
    def next(self):
        if self.stream:
            torch.cuda.current_stream().wait_stream(self.stream)
        samples = self.samples
        self.preload()
        return samples
    """
    using example:
        data_iter = MyDataLoader(dataset, data_config)
        prefetcher = Prefetcher(data_iter, cuda=trainer_config["cuda"])
        while True:
            batch = prefetcher.next()
            if batch is None:
                break
            # training code
    """
