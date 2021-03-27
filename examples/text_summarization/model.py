
import torch
from torch import nn
from torch.nn import functional as F

from yaonlp.layers import BiLSTM


class PointerGenerator(nn.Module):
    def __init__(self,
                 input_size, 
                 hidden_size,
                 vocab_size,
                 emb_dim,
                 dropout) -> None:
        super(PointerGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.dropout = dropout

        # TODO configurable
        self.coverage_loss_weight = 1.0  # lambda, equation 13

        self.use_coverage =  True
        self.use_pgen = True
        
        self.encoder = Encoder(vocab_size, emb_dim, hidden_size, dropout)
        self.reduce_state = ReduceState(hidden_size)
        self.decoder = Decoder(vocab_size, emb_dim, hidden_size, dropout)

    def forward(self,     
                enc_inputs,
                enc_lens, 
                enc_inputs_extend,
                oov_nums,
                dec_inputs, 
                dec_lens,
                max_len=150):
        batch_size, seq_len = enc_inputs.size()

        # enc_states: (batch_size, seq_len, hidden_size * 2)
        enc_states, enc_hidden = self.encoder(inputs=enc_inputs, seq_lens=enc_lens)

        # it is needed, because encoder's lstm is bidirectional, but decoder isn't
        dec_state = self.reduce_state(enc_hidden)

        # initial
        context_vector = torch.zeros(batch_size, self.hidden_size * 2)
        coverage_vector = torch.zeros(enc_states.size()[:2])   # (batch_size, seq_len)

        step_losses = []
        for step in range(min(max(dec_lens), max_len)):
            dec_prev = dec_inputs[:, step]

            final_dist, dec_state, h_star_t, attn_dist, p_gen, next_coverage =  \
                self.decoder(prev_target=dec_prev, 
                             prev_dec_state=dec_state, 
                             enc_states=enc_states, 
                             enc_input_extend=enc_inputs_extend,
                             oov_nums=oov_nums,
                             prev_context_vector=context_vector,
                             prev_coverage=coverage_vector)
            
            loss = final_dist.gather(1, dec_prev.unsqueeze(1)).squeeze()  # (batch_size, )
            step_loss = -torch.log(loss)

            if self.use_coverage:
                step_coveraged_loss = torch.sum(torch.min(attn_dist, coverage_vector))  # equation 12
                step_loss = step_loss + self.coverage_loss_weight * step_coveraged_loss  # equation 13

                coverage_vector = next_coverage
            
            step_losses.append(step_loss)

        sum_losses = torch.sum(torch.stack(step_losses, 1), 1)
        batch_avg_loss = sum_losses / dec_lens
        loss = torch.mean(batch_avg_loss)
 
        return loss


class Encoder(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 emb_dim,
                 hidden_size, 
                 dropout) -> None:
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=emb_dim)

        self.bilstm = BiLSTM(input_size=emb_dim, 
                             hidden_size=hidden_size,
                             num_layers=1, 
                             batch_first=True, 
                             dropout=dropout)
        
        self.proj = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)

    def forward(self, inputs, seq_lens):
        embedding = self.embedding(inputs)

        enc_states, hidden = self.bilstm(inputs=embedding, seq_lens=seq_lens)
        enc_states = enc_states.contiguous()

        return enc_states, hidden


# Add to the graph a linear layer to reduce the encoder's final hiddent and cell state into a single initial state for the decoder
class ReduceState(nn.Module):
    def __init__(self, hidden_size) -> None:
        super(ReduceState, self).__init__()
        self.hidden_size = hidden_size

        self.reduce_h = nn.Linear(hidden_size * 2, hidden_size)
        self.reduce_c = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, enc_hidden):
        h, c = enc_hidden  # from bilstm, both with shape (2, batch_size, hidden_size)

        # (batch_size, hidden_size * 2)
        h_in = h.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2)
        hidden_reduced_h = F.relu(self.reduce_h(h_in))

        c_in = c.transpose(0, 1).contiguous().view(-1, self.hidden_size * 2)
        hidden_reduced_c = F.relu(self.reduce_c(c_in))

        # h, c dim = (1, b, hidden_size)
        return (hidden_reduced_h.unsqueeze(0), hidden_reduced_c.unsqueeze(0)) 


class Attention(nn.Module):
    def __init__(self, hidden_size) -> None:
        super(Attention, self).__init__()
        self.use_coverage = True

        self.W_h = nn.Linear(hidden_size * 2, hidden_size * 2)  # encoder projection
        self.W_s = nn.Linear(hidden_size * 2, hidden_size * 2)  # decoder projection
        self.W_c = nn.Linear(1, hidden_size * 2)  # coverage projection

        self.v = nn.Linear(hidden_size*2, 1, bias=False)

    def forward(self, dec_in_state, enc_states, coverage_vector):
        batch_size, seq_len, _ = enc_states.size() 

        enc_feats = self.W_h(enc_states)  # (batch_size, seq_len, hidden_size * 2)
        dec_feats = self.W_s(dec_in_state)  # (batch_size, hidden_size * 2)

        # expand decoder features, (batch_size, seq_len, hidden_size * 2)
        dec_feats_expanded = \
            dec_feats.unsqueeze(1).expand(enc_states.size()).contiguous() 

        attn_feats = enc_feats + dec_feats_expanded # (batch_size, seq_len, hidden_size * 2)

        # calculate e, equation 11
        if self.use_coverage:
            coverage_vector = coverage_vector.unsqueeze(2)
            coverage_feats = self.W_c(coverage_vector)
            attn_feats += coverage_feats
        
        e = F.tanh(attn_feats)
        e = self.v(e)

        # calculate attention distribution 'a', equation 2
        attn_dist = F.softmax(e, dim=1).transpose(1, 2)  # (batch_size, 1, seq_len)

        # equation 3, calculate context vectors 'h_star_t
        context_vector = torch.bmm(attn_dist, enc_states)  
        context_vector = context_vector.squeeze(1) # (batch_size, hidden_size * 2)

        # squeeze, (batch_size, seq_len)
        attn_dist = attn_dist.squeeze(1) 

        # c_t, equation 10
        if self.use_coverage:
            coverage_vector = coverage_vector.squeeze(2)  # (batch_size, seq_len)
            coverage_vector = coverage_vector + attn_dist

        return context_vector, attn_dist, coverage_vector


class Decoder(nn.Module):
    def __init__(self, 
                 vocab_size, 
                 emb_dim,
                 hidden_size,
                 dropout) -> None:
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.use_coverage = True
        self.use_pgen = True
        
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=emb_dim)

        self.x_context = nn.Linear(hidden_size * 2 + emb_dim, emb_dim)

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
    
    def forward(self, 
                prev_target,
                prev_dec_state, 
                enc_states, 
                enc_input_extend, 
                oov_nums, 
                prev_context_vector, 
                prev_coverage):
        batch_size, seq_len, _ = enc_states.size()
        # decoder state 's_t'  
        target_emb = self.embedding(prev_target)
        # project target embedding and context vectors into a new embedding space
        x_context = self.x_context(torch.cat((target_emb, prev_context_vector), dim=1))  # (batch, emb_dim)

        lstm_out, dec_state = self.lstm(x_context.unsqueeze(1), prev_dec_state) # lstm_out: (batch_size, 1, hidden_size) 
        dec_h, dec_c = dec_state
        # concatenate lstm hidden state and cell state, (batch_size, hidden_size * 2)
        dec_state_hat = torch.cat((dec_h.view(-1, self.hidden_size),
                                   dec_c.view(-1, self.hidden_size)), dim=1)  

        # attention distribution
        # (batch_size, hidden_size * 2); (batch_size, seq_len); (batch_size, seq_len);
        h_star_t, attn_dist, coverage_vector = self.attention(dec_state_hat, enc_states, prev_coverage)

        # calculate vocab distribution P_vocab, equation 4
        combine_pv = torch.cat((lstm_out.squeeze(1), h_star_t), dim=1)
        vocab_dist = F.softmax(self.p_vocab2(self.p_vocab1(combine_pv)), dim=1)  # (batch_size, vocab_size)

        if self.use_pgen:
            # calculate P_gen, equation 8
            combine_pg = torch.cat((dec_state_hat, h_star_t, x_context), dim=1)
            p_gen = F.sigmoid(self.p_gen(combine_pg))

            # calculate final distribution P_w, equation 9
            vocab_dist_ = p_gen * vocab_dist
            attn_dist_ = (1 - p_gen) * attn_dist

            # extend vocab
            max_oov_nums = torch.max(oov_nums)
            extra_zeros = torch.zeros(batch_size, max_oov_nums)
            vocab_dist_ = torch.cat((vocab_dist_, extra_zeros), dim=1)
            # final_dist = vocab_dist_ + attn_dist_
            final_dist = vocab_dist_.scatter_add(dim=1, index=enc_input_extend, src=attn_dist_)
        else:
            final_dist = vocab_dist

        return final_dist, dec_state, h_star_t, attn_dist, p_gen, coverage_vector
