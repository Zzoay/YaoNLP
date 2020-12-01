
import torch
from torch import nn
from torch.nn import functional as F

from yaonlp.layers import NonLinear


class DependencyParser(nn.Module):
    def __init__(self, vocab_size, tag_size, rel_size, config):
        super(DependencyParser, self).__init__()
        # vocab_size:int = 0
        word_embed_dim:int = config["word_embed_dim"]

        # tag_size:int = 0
        tag_embed_dim:int = config["tag_embed_dim"]

        lstm_hiddens:int = config["lstm_hiddens"]
        lstm_dropout:float = config["lstm_dropout"]

        mlp_arc_size:int = config["mlp_arc_size"]
        mlp_rel_size:int = config["mlp_rec_size"]

        self.word_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=word_embed_dim)
        self.tag_emb = nn.Embedding(num_embeddings=tag_size, embedding_dim=tag_embed_dim)

        self.lstm = nn.LSTM(input_size=word_embed_dim+tag_embed_dim, 
                            hidden_size=lstm_hiddens, 
                            batch_first=True, 
                            bidirectional=True, 
                            dropout=0)

        self.mlp_arc_dep = NonLinear(in_features=2*lstm_hiddens, 
                                     out_features=mlp_arc_size+mlp_rel_size, 
                                     activation=nn.LeakyReLU(0.1))

        self.mlp_arc_head = NonLinear(in_features=2*lstm_hiddens, 
                                      out_features=mlp_arc_size+mlp_rel_size, 
                                      activation=nn.LeakyReLU(0.1))

        self.total_num = int((config["mlp_arc_size"]+config["mlp_rec_size"]) / 100)
        self.arc_num = int(config["mlp_arc_size"] / 100)
        self.rel_num = int(config["mlp_rec_size"] / 100)
        
        self.arc_biaffine = Biaffine(config["mlp_arc_size"], config["mlp_arc_size"], 1)
        self.rel_biaffine = Biaffine(config["mlp_rec_size"], config["mlp_rec_size"], rel_size)

        self.dropout = nn.Dropout(config["dropout"])
        
    def forward(self, words, tags, heads):  # x: batch_size, seq_len
        embed_word = self.word_emb(words)   # batch_size, seq_len, embed_dim
        embed_tag = self.tag_emb(tags) # batch_size, seq_len, embed_dim

        embed_x = torch.cat([embed_word, embed_tag], dim=2)
        embed_x = self.dropout(embed_x)

        # TODO consider the mask
        lsmt_output, _ = self.lstm(embed_x)  # lsmt_output: seq_len, batch, hidden_size * num_directions
        embed_x = self.dropout(lsmt_output)

        """
        UserWarning: The .grad attribute of a Tensor that is not a leaf Tensor is being accessed. Its .grad attribute won't be populated during autograd.backward(). 
        If you indeed want the gradient for a non-leaf Tensor, use .retain_grad() on the non-leaf Tensor. 
        If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead. See github.com/pytorch/pytorch/pull/30531 for more informations.
        """
        all_dep = self.mlp_arc_dep(lsmt_output)  
        all_head = self.mlp_arc_head(lsmt_output)

        all_dep = self.dropout(all_dep)
        all_head = self.dropout(all_head)

        all_dep_splits = torch.split(all_dep, split_size_or_sections=100, dim=2)
        all_head_splits = torch.split(all_head, split_size_or_sections=100, dim=2)

        arc_dep = torch.cat(all_dep_splits[:self.arc_num], dim=2)
        arc_head = torch.cat(all_head_splits[:self.arc_num], dim=2)

        arc_logit = self.arc_biaffine(arc_dep, arc_head)   # batch_size, seq_len, seq_len
        arc_logit = torch.squeeze(arc_logit, dim=3)

        rel_dep = torch.cat(all_dep_splits[self.arc_num:], dim=2)
        rel_head = torch.cat(all_head_splits[self.arc_num:], dim=2)

        rel_logit_cond = self.rel_biaffine(rel_dep, rel_head)  # batch_size, seq_len, seq_len, rel_nums
        rel_pred = torch.gather(rel_logit_cond, 2, heads.unsqueeze(2).unsqueeze(3).expand(-1, -1, -1, rel_logit_cond.shape[-1])).squeeze(2)
        return arc_logit, rel_pred


class Biaffine(nn.Module):
    def __init__(self, in1_features: int, in2_features: int, out_features: int):
        super(Biaffine, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features

        self.linear_in_features = in1_features 
        self.linear_out_features = out_features * in2_features

        # with bias default
        self.linear = nn.Linear(in_features=self.linear_in_features,
                                out_features=self.linear_out_features)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()

        affine = self.linear(input1)

        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine