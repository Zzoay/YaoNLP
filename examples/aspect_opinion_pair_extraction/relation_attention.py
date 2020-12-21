
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class RelationAttention(nn.Module):
    def __init__(self, args):
        super(RelationAttention, self).__init__()
        self.relation_hidden_dim = args.relation_hidden_dim
        self.relation_attention_dim = args.relation_attention_dim

        self.w_ta = nn.Parameter(torch.Tensor(self.relation_attention_dim, self.relation_hidden_dim))
        self.w_ja = nn.Parameter(torch.Tensor(self.relation_attention_dim, self.relation_hidden_dim))
        self.b = nn.Parameter(torch.Tensor(1, 1, 1, self.relation_attention_dim))
        self.v = nn.Parameter(torch.Tensor(1, self.relation_attention_dim))

        init.xavier_uniform_(self.w_ta)
        init.xavier_uniform_(self.w_ja)
        init.xavier_uniform_(self.b)
        init.xavier_uniform_(self.v)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, relation_hidden):
        batchSize = relation_hidden.size(0)
        seqLen = relation_hidden.size(1)

        ta_result = F.linear(relation_hidden, self.w_ta, None).view(batchSize, seqLen, 1, self.relation_attention_dim).repeat(1, 1, seqLen,
                                                                                                          1)
        ja_result = F.linear(relation_hidden, self.w_ja, None).view(batchSize, 1, seqLen, self.relation_attention_dim).repeat(1, seqLen, 1,
                                                                                                          1)

        attention_alpha = torch.tanh(ta_result + ja_result + self.b)  # b,s,s,h
        attention_alpha = F.linear(attention_alpha, self.v, None)  # b,s,s,1

        attention_alpha = self.softmax(attention_alpha.view(batchSize, seqLen, seqLen))  # b,s,s
        return attention_alpha
