
import jieba

# cut word for dependency parser
def cut_word(vocab_file):
    jieba.load_userdict(vocab_file)
    jieba.cut
    return