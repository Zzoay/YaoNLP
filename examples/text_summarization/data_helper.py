
import torch
from torch.utils.data import Dataset

from collections import Counter
import time

class TTDataset(Dataset):
    def __init__(self, vocab, vocab_file) -> None:
        self.data_file = data_file
        self.vocab_file = vocab_file

        self.summ_lst, self.article_lst, self.summ_lens, self.article_lens = self.read_data(vocab, data_file)

    def read_data(self, vocab, data_file):
        with open(self.data_file, 'r', encoding='utf-8') as f:
            summ_lst = list()
            article_lst = list()
            summ_lens = list()
            article_lens = list()

            for line in f.readlines():
                line_dct = eval(line)
                summ = line_dct['summarization']
                article = line_dct['article']

                summ_tmp = [vocab.get(x, 0) for x in summ]
                article_tmp = [vocab.get(x, 0) for x in article]

                summ_lens.append(len(summ_tmp))
                article_lens.append(len(article_tmp))

                summ_lst.append(torch.tensor(summ_tmp))
                article_lst.append(torch.tensor(article_tmp))
            
            return summ_lst, article_lst, summ_lens, article_lens

        def __getitem__(self, index):
            return self.summ_lst[idx], self.article_lst[idx], self.summ_lens[idx], self.article_lens[idx]
        
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
                if int(freq) < self.min_freq:    
                    break
                char_dct[character] = cnt
                cnt+=1

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