

import torch
from torch import nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, params):
        super(TextCNN, self).__init__()
        
        embed_dim = params.embed_dim
        kernel_num = params.kernel_num
        kernel_sizes = params.kernel_sizes
        dropout = params.dropout
        class_nums = params.class_num
        
        self.emb = nn.Embedding(params.vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(kernel_num*len(kernel_sizes), class_nums)

    def forward(self, x):
        x = self.emb(x)  # [batch_size, sentence_len, embed_dim]
        
        x = x.unsqueeze(1)  # [batch_size, 1, sentence_len, embed_dim]
        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        
        x = self.dropout(x)
        logit = self.fc(x)

        return logit
