
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy


class TransformerReprs(nn.Module):
    def __init__(self):
        super(TransformerReprs, self).__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        self.postion_embedding = PositionalEncoding(config.embed_dim, config.seq_len, config.dropout, config.device)
        self.encoder = TransformerEncoder(config.attn_dim, config.head_nums, config.hidden_dim, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])

        self.projection = nn.Linear(config.seq_len * config.attn_dim, config.project_dim)

    def forward(self, x):
        out = self.embedding(x[0])
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)  # (batch_size, seq_len, attn_dim)
        out = out.view(out.size(0), -1) 

        out = self.projection(out)  # (batch_size, seq_len, project_dim)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, head_nums, hidden, dropout):
        super(TransformerEncoder, self).__init__()
        self.attention = MultiHeadAttention(model_dim, head_nums, dropout)
        self.feed_forward = PositionWiseFeedForward(model_dim, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)  # (batch_size, seq_len, attn_dim)
        out = self.feed_forward(out)  # (batch_size, seq_len, attn_dim)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, embed, pad_size, dropout, device):
        super(PositionalEncoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed)) for i in range(embed)] for pos in range(pad_size)])
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, head_nums, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.head_nums = head_nums
        assert model_dim % head_nums == 0
        self.head_dim = model_dim // self.head_nums
        self.fc_Q = nn.Linear(model_dim, head_nums * self.head_dim)
        self.fc_K = nn.Linear(model_dim, head_nums * self.head_dim)
        self.fc_V = nn.Linear(model_dim, head_nums * self.head_dim)

        self.fc = nn.Linear(head_nums * self.head_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_Q(x)
        K = self.fc_K(x)
        V = self.fc_V(x)
        Q = Q.view(batch_size * self.head_nums, -1, self.head_dim)
        K = K.view(batch_size * self.head_nums, -1, self.head_dim)
        V = V.view(batch_size * self.head_nums, -1, self.head_dim)

        scale = K.size(-1) ** -0.5  # scale factor

        # compute self-attention
        attention = torch.matmul(Q, K.permute(0, 2, 1))  # (batch_size * head_nums, seq_len, seq_len)
        attention = attention * scale
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)  # (batch_size * head_nums, seq_len, head_dim)

        context = context.view(batch_size, -1, self.head_dim * self.head_nums)  # (batch_size, seq_len, head_dim * head_nums)
        out = self.fc(context)  # (batch_size, seq_len, model_dim)
        out = self.dropout(out)
        out = out + x  # residual
        out = self.layer_norm(out)
        return out


class PositionWiseFeedForward(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)  # (batch_size, seq_len, model_dim)
        out = self.dropout(out)
        out = out + x  # residual
        out = self.layer_norm(out)
        return out