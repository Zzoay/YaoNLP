
import jieba

class ParserTokenizer():
    def __init__(self, vocab_file)-> None:
        self.min_freq = 2  # TODO configurable

        self.vocab_file = self.read_vocab(vocab_file=vocab_file)
        self.cutter_init(filename=vocab_file)

    def cutter_init(self, filename):
        jieba.load_userdict(filename)

    def tokenize(self, sentence):
        seg_list = jieba.cut(sentence)
        ids = word2ids(seg_list)
        return ids

    def word2ids(self, vocab, word_list):
        ids = []
        for word in word_list:
            try:
                idx = vocab[word]
                ids.append(idx)
            except KeyError:
                ids.append(vocab['<unk>'])

    def read_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            char_dct = dict()
            cnt = 0
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

            return char_dct