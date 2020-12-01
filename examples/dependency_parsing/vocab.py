

class Vocab():
    def __init__(self, config):
        self.min_freq = config["min_freq"]

        self.word2id = self.read_vocab(config["word_vocab_file"], isword=True)
        self.tag2id = self.read_vocab(config["tag_vocab_file"])
        self.head2id = self.read_vocab(config["head_vocab_file"])
        self.rel2id = self.read_vocab(config["rel_vocab_file"])

        self.word_size = len(self.word2id)
        self.tag_size = len(self.tag2id) + 1  # start with 1, 0 represents the <unk> or <pad> or OOV word's tag
        self.rel_size = len(self.rel2id) + 1  # start with 1, 0 represents the <unk> or <pad> or OOV word's relation

    def read_vocab(self, vocab_file: str, isword: bool = False) -> dict:
        vocab = {}
        with open(vocab_file, "r", encoding='utf-8') as f:  # TODO: encoding unify
            if isword:  # 'word' starts with <pad>, <root>, <unk>
                cnt = 0 
            else:
                cnt = 1  
            for line in f.readlines():
                word, freq = line.split()
                if isword and int(freq) < self.min_freq:  # word's freq less than min freq
                    continue
                vocab[word] = cnt
                cnt += 1
        return vocab