
import jieba

# cut word for dependency parser
def cut_word(vocab_file):
    jieba.load_userdict(vocab_file)
    jieba.cut
    return


# TODO move
def drop_bi_input_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.new_full((batch_size, seq_length), 1-dropout_emb)
    word_masks = torch.bernoulli(word_masks)
    tag_masks = tag_embeddings.new_full((batch_size, seq_length), 1-dropout_emb)
    tag_masks = torch.bernoulli(tag_masks)
    scale = 2.0 / (word_masks + tag_masks + 1e-12)
    word_masks *= scale
    tag_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks

    return word_embeddings, tag_embeddings