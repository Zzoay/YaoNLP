
import torch
from torch import nn
from torch.nn import functional as F

from yaonlp.layers import NonLinear, Biaffine, BiLSTM


# Biaffine: https://arxiv.org/abs/1611.01734s
class DependencyParser(nn.Module):
    def __init__(self, vocab_size, tag_size, rel_size, config):
        super(DependencyParser, self).__init__()

        word_embed_dim:int = config["word_embed_dim"]
        tag_embed_dim:int = config["tag_embed_dim"]

        lstm_hiddens:int = config["lstm_hiddens"]
        lstm_num_layers:int = config["lstm_num_layers"]

        mlp_arc_size:int = config["mlp_arc_size"]
        mlp_rel_size:int = config["mlp_rel_size"]

        dropout = config["dropout"]

        self.word_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_embed_dim)
        self.tag_emb = nn.Embedding(num_embeddings=tag_size, embedding_dim=tag_embed_dim)

        self.bilstm = BiLSTM(input_size=word_embed_dim+tag_embed_dim, 
                             hidden_size=lstm_hiddens, 
                             num_layers=lstm_num_layers,
                             batch_first=True, 
                             dropout=0)

        self.mlp_arc_dep = NonLinear(in_features=2*lstm_hiddens, 
                                     out_features=mlp_arc_size+mlp_rel_size, 
                                     activation=nn.LeakyReLU(0.1))

        self.mlp_arc_head = NonLinear(in_features=2*lstm_hiddens, 
                                      out_features=mlp_arc_size+mlp_rel_size, 
                                      activation=nn.LeakyReLU(0.1))

        self.total_num = int((mlp_arc_size+mlp_rel_size) / 100)
        self.arc_num = int(mlp_arc_size / 100)
        self.rel_num = int(mlp_rel_size / 100)
        
        self.arc_biaffine = Biaffine(mlp_arc_size, mlp_arc_size, 1)
        self.rel_biaffine = Biaffine(mlp_rel_size, mlp_rel_size, rel_size)

        self.dropout = nn.Dropout(dropout)

    def encode(self, words, tags, seq_lens, eval=False):
        embed_word = self.word_emb(words)   # batch_size, seq_len, embed_dim
        embed_tag = self.tag_emb(tags) # batch_size, seq_len, embed_dim

        embed_x = torch.cat([embed_word, embed_tag], dim=2)
        embed_x = self.dropout(embed_x)

        lstm_output, _ = self.bilstm(embed_x, seq_lens)
        lstm_output = self.dropout(lstm_output)

        all_dep = self.mlp_arc_dep(lstm_output)  
        all_head = self.mlp_arc_head(lstm_output)

        all_dep = self.dropout(all_dep)
        all_head = self.dropout(all_head)

        return (embed_x, lstm_output, all_dep, all_head)    

    def forward(self, words, tags, heads, seq_lens, eval=False):  # x: batch_size, seq_len
        raise RuntimeError("This parser only use for encoding!")

        embed_word = self.word_emb(words)   # batch_size, seq_len, embed_dim
        embed_tag = self.tag_emb(tags) # batch_size, seq_len, embed_dim

        embed_x = torch.cat([embed_word, embed_tag], dim=2)
        embed_x = self.dropout(embed_x)

        lstm_output, _ = self.bilstm(embed_x, seq_lens)
        lstm_output = self.dropout(lstm_output)

        all_dep = self.mlp_arc_dep(lstm_output)  
        all_head = self.mlp_arc_head(lstm_output)

        all_dep = self.dropout(all_dep)
        all_head = self.dropout(all_head)

        all_dep_splits = torch.split(all_dep, split_size_or_sections=100, dim=2)
        all_head_splits = torch.split(all_head, split_size_or_sections=100, dim=2)

        arc_dep = torch.cat(all_dep_splits[:self.arc_num], dim=2)
        arc_head = torch.cat(all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(arc_dep, arc_head)   # batch_size, seq_len, seq_len
        arc_logit = arc_logit.squeeze(3)

        rel_dep = torch.cat(all_dep_splits[self.arc_num:], dim=2)
        rel_head = torch.cat(all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(rel_dep, rel_head)  # batch_size, seq_len, seq_len, rel_nums
        
        if eval:
            # change heads from golden to predicted
            _, heads = arc_logit.max(2)
        # expand: -1 means not changing the size of that dimension
        index = heads.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, rel_logit_cond.shape[-1]) 
        rel_logit = torch.gather(rel_logit_cond, dim=2, index=index).squeeze(2)
        return arc_logit, rel_logit