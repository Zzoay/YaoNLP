
from typing import Callable, Optional, List, Any

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel


label_to_idx = {'O':1, 'B-T':2, 'I-T':3, 'B-P':4, 'I-P':5}


class MyDataSet(Dataset):
    def __init__(self, args, filename) -> None:
        super(MyDataSet, self).__init__()

        self.data, _, _ = _read_data(filename)
    
    def __getitem__(self, idx):
        tmp = self.data[idx]
        return tmp['sentence'], tmp['label_ids'], tmp['relations'], len(tmp['label_ids'])
    
    def __len__(self):
        return len(self.data)

    
def _read_data(filename):
    datasets = []
    words = []
    labels = []
    label_ids = []
    relations = []

    target_cnt = 0
    opinion_cnt = 0
    relation_cnt = 0
    target_front_cnt = 0
    target_len = 0
    opinion_len = 0
    max_len = 0
    with open(filename, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if line == "#Relations":
                continue
            elif line == "" and len(words)>0:
                tmp_sentence = " ".join(words)
                datasets.append({"sentence":" ".join(words), "words": words, "labels": labels, "label_ids": label_ids, "relations": relations})
                if len(words) > max_len:
                    max_len = len(words)
                words = []
                labels = []
                label_ids = []
                relations = []
            else:
                line_split = line.split("\t")
                line_split_len = len(line_split)
                if line_split_len == 2:  # read words
                    word = line_split[0]
                    label = line_split[1]
                    label_idx = label_to_idx[label]
                    # WORD
                    words.append(word.lower())
                    # LABEL
                    labels.append(label)
                    label_ids.append(label_idx)
                    # count target and opinion
                    if label == "B-T":
                        target_cnt += 1
                        target_len += 1
                    elif label == "B-P":
                        opinion_cnt += 1
                        opinion_len += 1
                    elif label == "I-T":
                        target_len += 1
                    elif label == "I-P":
                        opinion_len += 1

                elif line_split_len == 4:  # read relations
                    target_strat, target_end, opinion_strat, opinion_end = line_split[2], line_split[3], line_split[0], line_split[1]

                    relations.append([target_strat, target_end, opinion_strat, opinion_end])  
    
                    # both true
                    if target_strat != '-1' and opinion_strat != '-1':
                        relation_cnt += 1

                        # count the target in front of opinion
                        if target_strat < opinion_strat:
                            target_front_cnt += 1

    # compute the average of target length and opinion length
    target_len_avg = target_len/target_cnt
    opinion_len_avg = opinion_len/opinion_cnt
    print("max_seq_length: " + str(max_len))

    return datasets, [target_cnt, opinion_cnt, relation_cnt, target_front_cnt], [target_len_avg, opinion_len_avg]


class MyDataLoader(DataLoader):
    def __init__(self, dataset, args, collate_fn: Callable[[List[Any]], Any]) -> None:
        super(MyDataLoader, self).__init__(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=args.shuffle,
            collate_fn=collate_fn)
        self.data_size = len(self.dataset)


class TokenizedCollator():
    def __init__(self, tokenizer, token_idx, label_idx, sort_key):
        self.token_idx = token_idx  # the index of data should be tokenized
        self.label_idx = label_idx  # the index of label 
        self.label2ids = {'O':1, 'B-T':2, 'I-T':3, 'B-O':4, 'I-O':5}

        self.sort_key = sort_key  # sort key

        self.tokenizer = tokenizer
    
    def _collate_fn(self, batch):
        ret = []
        max_len = 0
        batch.sort(key=self.sort_key, reverse=True)  
        for i, samples in enumerate(zip(*batch)):
            if i == self.token_idx:
                # {'input_ids':..., 'token_type_ids':..., 'attention_mask': ...}
                max_len = max(len(sentence.split()) for sentence in samples)
                input_ids, segment_ids, input_masks = self.tokenizer(samples,
                                                                     padding=True,
                                                                     truncation=True,
                                                                     return_tensors="pt").values()
                max_len = input_ids.shape[1]
                ret.append(input_ids)
                ret.append(segment_ids)
                ret.append(input_masks)
            elif i == self.label_idx:
                tmp_samples = []
                for labels in samples:
                    labels.insert(0, 0)  # add <cls> tag of bert model
                    labels.extend([0 for _ in range(max_len - len(labels))])
                    tmp_samples.append(torch.tensor(labels))

                samples = rnn_utils.pad_sequence(tmp_samples, batch_first=True, padding_value=0)  # padding
                ret.append(samples)
            else:
                ret.append(samples)
        # input_ids, segment_ids, input_masks, label_ids, relations
        return ret

    def __call__(self, batch):
        return self._collate_fn(batch)

