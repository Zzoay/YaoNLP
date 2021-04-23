
import numpy as np
from textrank4zh import TextRank4Keyword, TextRank4Sentence
from rouge import Rouge

from snownlp import SnowNLP

import re
import time

def load_data(data_file):
    with open(data_file, encoding='utf-8') as f:
        article_lst = list()
        summ_lst = list()

        for line in f.readlines():
            line_dct = eval(line)
            summ = line_dct['summarization']
            article = line_dct['article']

            article_lst.append(article)
            summ_lst.append(summ)
    
        return article_lst, summ_lst


def word_for_rouge(summ, hyps):
    word_dct = {}
    cnt = 0
    for c in summ:
        if c in word_dct:
            continue
        word_dct[c] = str(cnt)
        cnt += 1
    for c in hyps:
        if c in word_dct:
            continue
        word_dct[c] = str(cnt)
        cnt += 1
    summ_ids = [word_dct[c] for c in summ]
    article_ids = [word_dct[c] for c in hyps]
    return summ_ids, article_ids


def clean(content):
    content = content.replace('.', '') # 删除句子分隔符
    content = content.replace(' ', '') # 删除空格
    return content


def cal_rouge_textRank(article_lst, summ_lst):
    rouge = Rouge()
    tr4s = TextRank4Sentence()

    rouges = np.zeros((3,3))
    cnt = 0
    for article, summ in zip(*(article_lst, summ_lst)):    
        tr4s.analyze(text = article, lower = True , source = 'all_filters')
        
        keysentence_list = list()
        for item in tr4s.get_key_sentences(num = 10):
            s = ''.join(item.sentence)
            s = re.sub("\d{4,}",'',s)
            keysentence_list.append(s)
        hyps = ""
        for j , sentence in enumerate(keysentence_list):
            if j == 0 and len(sentence) > 60:
                hyps = sentence[:60]
                break
            if (len(hyps) + len(sentence)) <= 60:
                hyps += sentence
            else:
                break
        
        hyps = clean(hyps)
        summ = summ.strip()
        summ = clean(summ)
        summ_ids, hyps_ids = word_for_rouge(summ, hyps)
        rouge_score = rouge.get_scores(" ".join(hyps_ids)[:len(summ)], " ".join(summ_ids))

        rouge1 = rouge_score[0]["rouge-1"]
        rouge2 = rouge_score[0]["rouge-2"]
        rougel = rouge_score[0]["rouge-l"]

        rouges[0] += np.array(list(rouge1.values()))
        rouges[1] += np.array(list(rouge2.values()))
        rouges[2] += np.array(list(rougel.values()))
        cnt += 1

    rouges = rouges / cnt
    print("Rouge: Rouge-1 : F P R")
    print("Rouge: Rouge-2 : F P R")
    print("Rouge: Rouge-L : F P R")
    print(rouges)


def cal_rouge_snownlp(article_lst, summ_lst):
    rouge = Rouge()

    rouges = np.zeros((3,3))
    cnt = 0
    for article, summ in zip(*(article_lst, summ_lst)):    
        s = SnowNLP(article)
        hyps = s.summary(5)
        hyps = "，".join(hyps)
        hyps = clean(hyps)

        summ = summ.strip()
        summ = clean(summ)
        summ_ids, hyps_ids = word_for_rouge(summ, hyps)
        rouge_score = rouge.get_scores(" ".join(hyps_ids)[:len(summ)], " ".join(summ_ids))

        rouge1 = rouge_score[0]["rouge-1"]
        rouge2 = rouge_score[0]["rouge-2"]
        rougel = rouge_score[0]["rouge-l"]

        rouges[0] += np.array(list(rouge1.values()))
        rouges[1] += np.array(list(rouge2.values()))
        rouges[2] += np.array(list(rougel.values()))
        cnt += 1

    rouges = rouges / cnt
    print("Rouge: Rouge-1 : F P R")
    print("Rouge: Rouge-2 : F P R")
    print("Rouge: Rouge-L : F P R")
    print(rouges)


if __name__ == '__main__':
    s_time = time.time()

    data_file = "/home/jgy/YaoNLP/data/TTNewsCorpus_NLPCC2017/toutiao4nlpcc_eval/evaluation_with_ground_truth.txt"
    article_lst, summ_lst = load_data(data_file)

    cal_rouge_snownlp(article_lst, summ_lst)

    print(f"Time cost: {time.time()-s_time} s")