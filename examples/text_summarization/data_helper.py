
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

from collections import Counter
import time

from syntax_enhance.tokenizer import ParserTokenizer
import config


class TTDataset(Dataset):
    def __init__(self, 
                 config,
                 vocab: dict, 
                 data_file: str) -> None:
        self.data_file = data_file
        self.max_summ_len = config.max_summ_len
        self.max_article_len = config.max_article_len

        self.use_syn_enhanced = config.use_syn_enhanced
        if self.use_syn_enhanced:
            self.syn_tokenizer = ParserTokenizer(vocab_file=config.parser_vocab_file)
        
        self.use_bert_enhanced = config.use_bert_enhanced
        if self.use_bert_enhanced:
            self.bert_tokenizer = BertTokenizer.from_pretrained(config.bert_tokenizer_file)

        self.summ_lst, self.summ_tag_lst, self.article_lst, self.article_extend_lst, self.summ_lens, self.article_lens, self.oov_nums = self.read_data(vocab, data_file)

    def read_data(self, 
                  vocab: dict,
                  data_file: str) -> tuple:
        article_bert_tokens, article_syntax_tokens = list(), list()
        with open(self.data_file, 'r', encoding='utf-8') as f:
            summ_lst = list()
            summ_tag_lst = list()
            article_lst = list()
            article_extend_lst = list()

            oov_nums = list()
            summ_lens = list()
            article_lens = list()
            for line in f.readlines():
                line_dct = eval(line)
                summ = line_dct['summarization']
                article = line_dct['article']

                # summ = summ.split()
                # article = article.split()
                # summ_cut = list(self.syn_tokenizer.segment(summ))
                # article_cut = list(self.syn_tokenizer.segment(article))

                summ_ids = [vocab.get(x, vocab['<unk>']) for x in summ]
                # article_ids = [vocab.get(x, 0) for x in article]
                article_ids, article_extend_vocab, oovs = self.article2ids(vocab, article)  # extend oov

                # truncation
                article_ids, article_len = self.truncate(article_ids, self.max_article_len)
                # summ_ids, summ_len = self.truncate(summ_ids, self.max_summ_len)
                # summ_tag_ids = summ_ids
                summ_ids, summ_tag_ids, summ_len = self.get_dec_input_target(summ_ids, self.max_summ_len, vocab['<start>'], vocab['<end>'])
                article_extend_vocab, _ = self.truncate(article_extend_vocab, self.max_article_len)

                # TODO dependency parser


                # TODO bert
                if self.use_bert_enhanced:
                    input_ids, segment_ids, input_masks = self.bert_tokenizer(article,
                                                                                padding=True,
                                                                                truncation=True,
                                                                                return_tensors="pt").values()
                    article_bert_tokens.append((input_ids, segment_ids, input_masks))

                summ_lens.append(summ_len)
                article_lens.append(article_len)

                oov_nums.append(len(oovs))

                summ_lst.append(torch.tensor(summ_ids))
                summ_tag_lst.append(torch.tensor(summ_tag_ids))
                article_lst.append(torch.tensor(article_ids))
                article_extend_lst.append(torch.tensor(article_extend_vocab).long())
            
            
            return summ_lst, summ_tag_lst, article_lst, article_extend_lst, summ_lens, article_lens, oov_nums
    
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
                ids.append(vocab['<unk>'])  # add <unk>
                if w not in oovs:
                    oovs.append(w)
                oov_num = oovs.index(w)
                extend_ids.append(len(vocab) + oov_num)  # extend
        return ids, extend_ids, oovs

    # add <start> and <end> tag, and truncation
    def get_dec_input_target(self, summ_ids, max_length, start_id, end_id):
        ids = [start_id] + summ_ids[:]
        target = summ_ids[:]

        ids, ids_lens = self.truncate(ids, max_length)

        if len(ids) > max_length: # truncate
            target = target[:max_length] # no end_token
        else: # no truncation
            target.append(end_id) # end token
        return ids, target, ids_lens

    def truncate(self, ids_list, max_length):
        length = len(ids_list)
        if length > max_length:
            length = max_length
            ids_list = ids_list[:max_length]
        
        return ids_list, length
        
    def __getitem__(self, idx):
        return self.summ_lst[idx], self.summ_tag_lst[idx], self.article_lst[idx],  self.article_extend_lst[idx], self.summ_lens[idx], self.article_lens[idx], self.oov_nums[idx]
    
    def __len__(self):
        return len(self.article_lst)

class Vocab(object):
    def __init__(self, data_file, vocab_file, min_freq=4):
        self.data_file = data_file
        self.vocab_file = vocab_file
        self.min_freq = 20  # TODO configurable

        self._vocab, self.id2word = self.read_vocab()

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
            char_dct['<pad>'] = 0
            # char_dct['<unk>'] = 1
            char_dct['<start>'] = 1
            char_dct['<end>'] = 2

            cnt = 3
            for line in f.readlines():
                try:
                    character, freq = line.strip().split(' ')
                except ValueError:
                    continue
                # vocab file sorted already
                if config.use_syn_enhanced and (character in ['<pad>', '<root>', '<unk>']):
                    continue
                if int(freq) < self.min_freq:    
                    break
                char_dct[character] = cnt
                cnt+=1
            # add <PAD>, <UNK> token
            # char_dct['<pad>'] = 0
            char_dct['<unk>'] = cnt-1
            return char_dct, list([item[0] for item in char_dct.items()])
    
    def get_vocab(self):
        return self._vocab
    
    def size(self):
        return len(self._vocab)

def ids2words(ids, id2word):
    res = []
    for i in ids:
        if i >= len(id2word): # OOV
            res.append(id2word[-1])
        else:
            res.append(id2word[i])
    return res
    

if __name__ == "__main__":
    s_time = time.time()

    data_file = r"data\TTNewsCorpus_NLPCC2017\toutiao4nlpcc\train_with_summ.txt"
    vocab_file = r"data\TTNewsCorpus_NLPCC2017\vocab.txt"

    vocab = Vocab(data_file, vocab_file).get_vocab()
    dataset = TTDataset(vocab, data_file)

    # pad_test = torch.nn.utils.rnn.pad_sequence(summ_lst[:64], batch_first=True)
    print(f"time cost {time.time() - s_time} s")