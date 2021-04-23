
import sys
sys.path.append(r".")   # add the YAONLP path
sys.path.append(r"/home/jgy/YaoNLP/examples/text_summarization/")

from collections import Counter

import config
from syntax_enhance.tokenizer import ParserTokenizer


# process by syntax tokenizer
def preprocess(syn_tokenizer, data_file, save_file):
    ret = ''
    fw = open(save_file, "w+")
    with open(data_file, 'r') as f:
        for line in f.readlines():
            line_dct = eval(line)
            summ = line_dct['summarization']
            article = line_dct['article']

            summ_cut = " ".join(syn_tokenizer.segment(summ))
            article_cut = " ".join(syn_tokenizer.segment(article))

            dct = {}
            dct["summarization"] = summ_cut
            dct["article"] = article_cut

            # ret += str(dct).strip() + "\n" 
            s = str(dct).strip() + "\n"
            fw.write(s)
    
    fw.close()


def build_vocab(data_file, save_file):
    cnt = None
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_dct = eval(line)
            summ = line_dct['summarization'].split()
            article = line_dct['article'].split()
            if cnt:
                cnt = cnt + Counter(summ) + Counter(article)
                continue
            cnt = Counter(summ) + Counter(article)
        print(dict(cnt))
        print(cnt.most_common())

        with open(save_file, 'w', encoding='utf-8') as f:
            s = "\n".join([f"{x[0]} {x[1]}" for x in cnt.most_common()])
            f.write(s)
        print('end')


if __name__ == "__main__":
    # vocab = Vocab(data_file=config.train_data_file, vocab_file=config.parser_vocab_file).get_vocab()
    syn_tokenizer = ParserTokenizer(vocab_file=config.parser_vocab_file)
    # TODO remember change dir, when process
    # preprocess(syn_tokenizer, config.train_data_file, save_file="/home/jgy/YaoNLP/data/TTNews_Processed/train.txt")
    # build_vocab(data_file="/home/jgy/YaoNLP/data/TTNews_Processed/train.txt", save_file="/home/jgy/YaoNLP/data/TTNews_Processed/vocab.txt")