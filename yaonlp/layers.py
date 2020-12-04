
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

 
class NonLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: Callable): 
        super(NonLinear, self).__init__()
        self._linear = nn.Linear(in_features, out_features)
        self._activation = activation
    
    def forward(self, x):
        return self._activation(self._linear(x))


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

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor, batch_first: bool = True):
        seq_packed = torch.nn.utils.rnn.pack_padded_sequence(inputs, seq_lens, batch_first=batch_first)
  
        lsmt_output, _ = self.lstm(seq_packed)   # lsmt_output: *, hidden_size * num_directions
        # seq_unpacked: batch_size, seq_max_len in batch, hidden_size * num_directions
        seq_unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(lsmt_output, batch_first=batch_first)  
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

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()

        affine = self.linear(input1)

        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)

        biaffine = torch.transpose(torch.bmm(affine, input2), 1, 2)

        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine