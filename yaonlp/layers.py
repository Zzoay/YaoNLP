
from typing import Callable, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

 
class NonLinear(nn.Module):
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 activation: Optional[Callable] = None, 
                 initial: Optional[Callable] = None) -> None: 
        super(NonLinear, self).__init__()
        self._linear = nn.Linear(in_features, out_features)
        self._activation = activation

        if initial:
            initial(self._linear.weight)
    
    def forward(self, x):
        if self._activation:
            return self._activation(self._linear(x))
        return self._linear(x)


class BiLSTM(nn.Module):
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int = 1,
                 batch_first: bool = True, 
                 dropout: float = 0) -> None: 
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            batch_first=True, 
                            bidirectional=True, 
                            dropout=dropout)
        self.batch_first = batch_first

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor):
        seq_packed = torch.nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=self.batch_first)
  
        lsmt_output, _ = self.lstm(seq_packed)   # lsmt_output: *, hidden_size * num_directions
        # seq_unpacked: batch_size, seq_max_len in batch, hidden_size * num_directions
        seq_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lsmt_output, batch_first=self.batch_first)  
        return seq_unpacked
    

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

    def forward(self, input1: torch.Tensor, input2: torch.Tensor):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()

        affine = self.linear(input1)

        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine


class CRF(nn.Module):
    def __init__(self, 
                 tag_size: int,
                 init_func: Callable = init.normal_,
                 to_cuda: bool = False) -> None:
        super(CRF, self).__init__()
        self.tag_size = tag_size
        self.START_IDX = tag_size - 2
        self.END_IDX = tag_size - 1

        self.to_cuda = to_cuda

        # Matrix of transition parameters.  
        # Entry i,j is the score of transitioning to i from j.
        if to_cuda:
           self.transitions = nn.Parameter(torch.empty(self.tag_size, self.tag_size)).cuda() 
        else:
            self.transitions = nn.Parameter(torch.empty(self.tag_size, self.tag_size))
        self.reset_parameters(init_func=init_func)

    def reset_parameters(self, init_func: Optional[Callable] = None) -> None:
        if init_func:
            init_func(self.transitions)
        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.START_IDX, :] = -10000
        self.transitions.data[:, self.END_IDX] = -10000

    def _forward_alg(self, feats: torch.Tensor):
        batch_size, seq_len, tag_size = feats.size()
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((batch_size, self.tag_size), -10000.0)  # batch_size, tag_size
        # START_TAG has all of the score with log(1) = 0, while others're log(0) ~= -10000 (a small number)
        init_alphas[:, self.START_IDX] = 0.0

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas
        if self.to_cuda:
            forward_var = forward_var.cuda()

        # # Iterate through the sentence
        for i in range(seq_len):
            feat = feats[:, i, :]   # (batch_size, tag_size)
            forward_broadcast = forward_var.unsqueeze(1)  # (batch_size, 1, tag_size) 

            # emitting score and transfering score
            emit_score = feat.unsqueeze(2)   # (batch_size, tag_size, 1)
            trans_score = self.transitions.unsqueeze(0)  # (1, tag_size, tag_size)

            score = forward_broadcast + emit_score + trans_score  # (batch_size, tag_size, tag_sizes)
            
            forward_var = torch.logsumexp(score, dim=2)
        # add END_TAG   
        terminal_var = forward_var + self.transitions[self.END_IDX]  # (batch_size, tag_size)
        alpha = torch.logsumexp(terminal_var, dim=1)  # (batch_size, )
        return alpha

    def _score_sentence(self, feats: torch.Tensor, tags: torch.Tensor):
        batch_size, seq_len, tag_size = feats.size()
        # Gives the score of a provided tag sequence
        score = torch.zeros(batch_size)
        # concatenate START_TAG ix
        start_tag_ix = torch.full((batch_size, 1), self.START_IDX, dtype=torch.long)
        if self.to_cuda:
            score = score.cuda()
            start_tag_ix = start_tag_ix.cuda()

        tags = torch.cat([start_tag_ix, tags], dim=1)  # (batch_size, seq_len + 1)

        for i in range(seq_len):
            feat = feats[:, i, :]  # (batch_size, tag_size)
            # Accumulate the transition and emission score for each frame
            emission_score = feat[range(batch_size), tags[:, i + 1]]  # (batch_size, )
            transfer_score = self.transitions[tags[:, i + 1], tags[:, i]].flatten()  # (batch_size, )

            score = score + transfer_score + emission_score  # (batch_size, )
        # add END_TAG 
        score = score + self.transitions[self.END_IDX, tags[:,-1]]   # (batch_size, )
        return score

    def neg_log_likelihood(self, feats, tags):
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.sum(forward_score - gold_score)
    
    def _viterbi_decode(self, feats):
        batch_size, seq_len, tag_size = feats.size()

        # Initialize the viterbi variables in log space
        init_alphas = torch.full((batch_size, self.tag_size), -10000.)
        init_alphas[:, self.START_IDX] = 0.0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_alphas
        if self.to_cuda:
            forward_var = forward_var.cuda()

        backpointers = []
        for feat_index in range(seq_len):
            feat = feats[:, feat_index, :]

            forward_broadcast = forward_var.unsqueeze(1)  # (batch_size, 1, tag_size)
            trans_score = self.transitions.unsqueeze(0)  # (1, tag_size, tag_size)

            next_tag_var = forward_broadcast + trans_score  # (batch_size, tag_size, tag_size)

            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=2)  # (batch_size, tag_size)

            forward_var = viterbivars_t + feat  # (batch_size, tag_size)
            backpointers.append(bptrs_t)

        # add transition of END_TAG
        terminal_var = forward_var + self.transitions[self.END_IDX]
    
        path_score, best_tag_id = torch.max(terminal_var, dim=1)
        best_tag_id = best_tag_id.tolist()

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[range(bptrs_t.shape[0]), best_tag_id].tolist()
            best_path.append(best_tag_id)

        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start[0] == self.START_IDX  # Sanity check

        best_path.reverse()
        # transpose to (batch_size, seq_len)
        best_path = torch.tensor(best_path).T
        if self.to_cuda:
            best_path = best_path.cuda()

        return path_score, best_path
    
    # don't confuse this with _forward_alg above.
    def forward(self, feats):  
        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(feats)
        return score, tag_seq
