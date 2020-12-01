
import os
import sys
sys.path.append("C:\\Users\\Admin\\Desktop\\YaoNLP")

from typing import List, Union, Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from yaonlp.config_loader import Config
from dependency import Dependency


class CTBDataset(Dataset):
    def __init__(self, vocab, config: dict):
        # data
        self.words, self.tags, self.heads, self.rels, self.masks = self.read_data(vocab, config["data_path"])

    def read_data(self, 
                 vocab, 
                 data_path: str,         # word, tag, head, rel, mask
                 max_len: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if not max_len:
            max_len = max(len(sentence) for sentence in load_ctb(data_path))
        w_tk_lst:List = []
        t_tk_lst:List = []
        h_tk_lst:List = []
        r_tk_lst:List = []
        m_tk_lst:List = []
        for sentence in load_ctb(data_path):   # sentence: [dep, dep, ...]; dep.attrs: id, word, tag, head, rel
            word_tokens = np.zeros(max_len, dtype=np.int64)  # <pad> is 0 default
            tag_tokens = np.zeros(max_len, dtype=np.int64)
            head_tokens = np.zeros(max_len, dtype=np.int64) 
            rel_tokens = np.zeros(max_len, dtype=np.int64)
            mask_tokens = np.zeros(max_len, dtype=np.int64)
            for i,dep in enumerate(sentence):
                if i == max_len:
                    break
                word = vocab.word2id.get(dep.word)
                word_tokens[i] =  word or 2  # while OOV set <unk> token
                tag_tokens[i] = (vocab.tag2id.get(dep.tag) if word else 0) or 0  # if there is no word or not tag, set 0
                head_tokens[i] = (vocab.head2id.get(dep.head) if word else 0) or 0
                rel_tokens[i] = (vocab.rel2id.get(dep.rel) if word else 0) or 0
                mask_tokens[i] = 1 if word else 0   # if is there a word, mask = 1, else 0 
                
            w_tk_lst.append(word_tokens)
            t_tk_lst.append(tag_tokens)
            h_tk_lst.append(head_tokens)
            r_tk_lst.append(rel_tokens)
            m_tk_lst.append(mask_tokens)
        return torch.tensor(w_tk_lst), torch.tensor(t_tk_lst), torch.tensor(h_tk_lst), torch.tensor(r_tk_lst), torch.tensor(m_tk_lst)
    
    def __getitem__(self, idx):
        return self.words[idx], self.tags[idx], self.heads[idx], self.rels[idx], self.masks[idx]
    
    def __len__(self):
        return len(self.words)


def load_ctb(data_path: str):
    file_names:List[str] = os.listdir(data_path)
    ctb_files:List[str] = [data_path+fle for fle in file_names]

    # id, form, tag, head, rel
    sentence:List[Dependency] = []

    for ctb_file in ctb_files:
        # print(ctb_file)
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


class MyDataLoader(DataLoader):
    def __init__(self, dataset, config) -> None:
        super(MyDataLoader, self).__init__(
            dataset, 
            batch_size=config["batch_size"], 
            shuffle=config["shuffle"],
        )
    
    def __len__(self):
        return len(self.dataset)