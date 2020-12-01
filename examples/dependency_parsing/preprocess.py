
import sys
sys.path.append("C:\\Users\\Admin\\Desktop\\YaoNLP")

from data_helper import load_ctb


def make_vocab(data_path: str, vocab_file: str) -> None: 
    word_vocab:dict = {}  # <pad>, <root>, <unk> = 0, 1, 2
    tag_vocab:dict = {}
    head_vocab:dict = {}
    rel_vocab:dict = {}
    for sentence in load_ctb(data_path):
        for dep in sentence:
            word_vocab = make_vocab_dct(dep.word, word_vocab)
            tag_vocab = make_vocab_dct(dep.tag, tag_vocab)
            head_vocab = make_vocab_dct(dep.head, head_vocab)
            rel_vocab= make_vocab_dct(dep.rel, rel_vocab)

    save_vocab(word_vocab, vocab_file, vocab_name="word_vocab.txt", isword=True)
    save_vocab(tag_vocab, vocab_file, vocab_name="tag_vocab.txt")
    save_vocab(head_vocab, vocab_file, vocab_name="head_vocab.txt")
    save_vocab(rel_vocab, vocab_file, vocab_name="rel_vocab.txt")


def make_vocab_dct(tok, vocab): 
    try:
        vocab[tok] += 1
    except KeyError:
        vocab[tok] = 1
    return vocab


def save_vocab(vocab: dict, vocab_path: str, vocab_name: str, isword: bool = False) -> None:
    with open(vocab_path+vocab_name, "w+", encoding='utf-8') as f:
        word_freq = []
        if isword:
            word_freq.extend([("<pad>", 10000), ("<root>", 10000), ("<unk>", 10000)])
        word_freq.extend(sorted(vocab.items(), key=lambda item:item[1], reverse=True))
        for word, freq in word_freq:
            f.write("{} {} \n".format(word, freq))
    
if __name__ == "__main__":
    data_path = "data/ctb8.0/dep/"
    vocab_path = "data/ctb8.0/vocab/"
    make_vocab(data_path, vocab_path)