
import torch
from torch import nn
from torch.nn import functional as F

from .yaonlp.layers import BiLSTM


class PointerGenerator(nn.Module):
    def __init__(self,
                 input_size, 
                 hidden_size, 
                 num_layers, 
                 dropout) -> None:

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.encoder = None
        self.decoder = None


class Encoder(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 emb_dim,
                 hidden_size, 
                 dropout) -> None:
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=emb_dim)

        self.bilstm = BiLSTM(input_size=emb_dim, 
                             hidden_size=hidden_size,
                             num_layers=1, 
                             batch_first=True, 
                             bidirectional=True, 
                             dropout=dropout)

    def forward(self, inputs, seq_lens):
        embedding = self.embedding(inputs)

        encoder_outputs, _ = self.bilstm(embedding)
        encoder_outputs = encoder_outputs.contiguous()

        return encoder_outputs


class Attention(nn.Module):
    def __init__(self, hidden_size) -> None:
        self.W_h = nn.Linear(hidden_size*2, hidden_size*2)
        self.W_s = nn.Linear(hidden_size*2, hidden_size*2)
        self.W_c = nn.Linear(hidden_size*2, hidden_size)

        self.v = nn.Linear(hidden_size*2, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, coverage_vector):
        # calculate e, equation 11
        e = F.tanh(self.W_h(encoder_outputs) + self.W_s(decoder_state) + self.W_c(coverage_vector))
        e = self.v(e)

        # calculate attention distribution 'a', eqution 2
        attn_dist = F.softmax(e)

        # eqution 3
        h_star_t = torch.bmm(attn_dist, encoder_outputs)

        # c_t, eqution 10
        coverage_vector = coverage_vector + attn

        return h_star_t, attn_dist, coverage_vector


class Decoder(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 emb_dim,
                 hidden_size,
                 dropout) -> None:

        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=emb_dim)
    
        self.lstm = nn.LSTM(input_size=emb_dim, 
                            hidden_size=hidden_size,
                            num_layers=1, 
                            batch_first=True, 
                            bidirectional=False, 
                            dropout=dropout)
        
        self.attention = Attention(hidden_size)

        self.p_vocab1 = nn.Linear(hidden_size * 3, hidden_size)
        self.p_vocab2 = nn.Linear(hidden_size, vocab_size)

        self.p_gen = nn.Linear(hidden_size * 4 + emb_dim, 1)
    
    def forward(self, golden_pre, decoder_state, encoder_outputs, coverage_vector):
        # decoder_state 's_t'
        x = self.embedding(golden_pre)
        lstm_out, s_t = self.lstm(x, s_t)
        dec_h, dec_c = s_t
        s_t = torch.cat((dec_h.view(-1, hidden_size),
                         dec_c.view(-1, hidden_size)), dim=1)  

        # attention distribution
        h_star_t, attn_dist, coverage_vector = self.attention(decoder_state, encoder_outputs, coverage_vector)

        # calculate vocab distribution P_vocab, eqution 4
        combine_pv = torch.cat((decoder_state, h_star_t), dim=1)
        vocab_dist = F.softmax(self.p_vocab2(self.p_vocab1(combine)))

        # calculate P_gen, eqution 8
        combine_pg = torch.cat((decoder_state, h_star_t, x), dim=1)
        p_gen = F.sigmoid(self.p_gen(combine_pg))

        # calculate final distribution P_w, eqution 9
        vocab_dist_ = p_gen * vocab_dist
        attn_dist_ = (1 - p_gen) * attn_dist
        final_dist = vocab_dist_ + attn_dist_

        return final_dist, s_t, h_star_t, attn_dist, p_gen, coverage_vector
