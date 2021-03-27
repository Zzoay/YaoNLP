
import torch
from torch.utils.data import Dataset

from collections import Counter
import time

class TTDataset(Dataset):
    def __init__(self, vocab: dict, data_file: str) -> None:
        self.data_file = data_file

        self.summ_lst, self.article_lst, self.article_extend_lst, self.summ_lens, self.article_lens, self.oov_nums = self.read_data(vocab, data_file)

    def read_data(self, 
                  vocab: dict,
                  data_file: str, 
                  max_summ_len: int = 200, 
                  max_article_len: int = 800) -> tuple:
        with open(self.data_file, 'r', encoding='utf-8') as f:
            summ_lst = list()
            article_lst = list()
            article_extend_lst = list()

            oov_nums = list()
            summ_lens = list()
            article_lens = list()
            for line in f.readlines():
                line_dct = eval(line)
                summ = line_dct['summarization']
                article = line_dct['article']

                summ_ids = [vocab.get(x, 0) for x in summ]
                # article_ids = [vocab.get(x, 0) for x in article]
                article_ids, article_extend_vocab, oovs = self.article2ids(vocab, article)  # extend oov

                summ_len = len(summ_ids)
                article_len = len(article_ids)
                oov_nums.append(len(oovs))

                # truncation
                if summ_len > max_summ_len:
                    summ_len = max_summ_len
                    summ_ids = summ_ids[:max_summ_len]
                
                # truncate
                if article_len > max_article_len:
                    article_len = max_article_len
                    article_ids = article_ids[:max_article_len]

                if len(article_extend_vocab) > max_article_len:
                    article_extend_vocab = article_extend_vocab[:max_article_len]

                summ_lens.append(summ_len)
                article_lens.append(article_len)

                summ_lst.append(torch.tensor(summ_ids))
                article_lst.append(torch.tensor(article_ids))
                article_extend_lst.append(torch.tensor(article_extend_vocab).long())
            
            return summ_lst, article_lst, article_extend_lst, summ_lens, article_lens, oov_nums
    
    # extend oov
    def article2ids(self, vocab, article):
        ids = []
        extend_ids = []
        oovs= []
        for w in article:
            try:
                idx = vocab[w]
                ids.append(idx)
                extend_ids.append(idx)
            except KeyError:  # oov
                ids.append(vocab['<UNK>'])  # add <UNK>
                if w not in oovs:
                    oovs.append(w)
                oov_num = oovs.index(w)
                extend_ids.append(len(vocab) + oov_num)  # extend
        return ids, extend_ids, oovs

    def __getitem__(self, idx):
        return self.summ_lst[idx], self.article_lst[idx], self.article_extend_lst[idx], self.summ_lens[idx], self.article_lens[idx], self.oov_nums[idx]
    
    def __len__(self):
        return len(self.article_lst)

class Vocab(object):
    def __init__(self, data_file, vocab_file, min_freq=4):
        self.data_file = data_file
        self.vocab_file = vocab_file
        self.min_freq = min_freq

        self._vocab = self.read_vocab()
    
    def build_vocab(self):
        cnt = None
        with open(self.data_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line_dct = eval(line)
                summ = line_dct['summarization']
                article = line_dct['article']
                if cnt:
                    cnt = cnt + Counter(summ) + Counter(article)
                    continue
                cnt = Counter(summ) + Counter(article)
            print(dict(cnt))
            print(cnt.most_common())
        
        with open("vocab.txt", 'w', encoding='utf-8') as f:
            s = "\n".join([f"{x[0]} {x[1]}" for x in cnt.most_common()])
            f.write(s)
        print('end')

    def read_vocab(self):
        with open(self.vocab_file, 'r', encoding='utf-8') as f:
            char_dct = dict()
            cnt = 1
            for line in f.readlines():
                try:
                    character, freq = line.split(' ')
                except ValueError:
                    continue
                # vocab file sorted already
                if int(freq) < self.min_freq:    
                    break
                char_dct[character] = cnt
                cnt+=1

            # add <PAD>, <UNK> token
            char_dct['<PAD>'] = 0
            char_dct['<UNK>'] = cnt-1
            return char_dct
    
    def get_vocab(self):
        return self._vocab


if __name__ == "__main__":
    s_time = time.time()

    data_file = r"data\TTNewsCorpus_NLPCC2017\toutiao4nlpcc\train_with_summ.txt"
    vocab_file = r"data\TTNewsCorpus_NLPCC2017\vocab.txt"

    vocab = Vocab(data_file, vocab_file).get_vocab()
    dataset = TTDataset(vocab, data_file)

    # pad_test = torch.nn.utils.rnn.pad_sequence(summ_lst[:64], batch_first=True)
    print(f"time cost {time.time() - s_time} s")