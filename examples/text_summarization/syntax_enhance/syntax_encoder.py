
import sys
sys.path.append(r".") 

import torch
from torch import nn

from yaonlp.layers import NonLinear


# TODO configurable
word_dims = 100
lstm_layers = 3
lstm_hiddens = 400
mlp_arc_size = 500
mlp_rel_size = 100


class SyntaxReprs(nn.Module):
    def __init__(self, parser_file):
        super(SyntaxReprs, self).__init__()
        self.hidden_dims = 2 * lstm_hiddens

        self.parser = self.load_parser(parser_file)

        self.transformer_emb = NonLinear(word_dims, self.hidden_dims, activation=nn.ReLU)

        parser_dim = 2 * lstm_hiddens
        # self.transformer_lstm = nn.ModuleList([NonLinear(parser_dim, self.hidden_dims, activation=nn.ReLU)
        #                                         for i in range(lstm_layers)])
        self.transformer_lstm = NonLinear(parser_dim, self.hidden_dims, activation=nn.ReLU)

        parser_mlp_dim = mlp_arc_size + mlp_rel_size
        self.transformer_dep = NonLinear(parser_mlp_dim, self.hidden_dims, activation=nn.ReLU)
        self.transformer_head = NonLinear(parser_mlp_dim, self.hidden_dims, activation=nn.ReLU)

    def load_parser(self, parser_file):
        parser = torch.load(parser_file)

        return parser

    def forward(self, inputs, seq_lens):
        embed_word = parser.word_emb(inputs)   # batch_size, seq_len, embed_dim

        lstm_output, _ = parser.bilstm(embed_x, seq_lens)

        all_dep = parser.mlp_arc_dep(lstm_output)  
        all_head = parser.mlp_arc_head(lstm_output)

        x_syns = []
        x_syn_emb = self.transformer_emb(embed_word)
        x_syns.append(x_syn_emb)

        # for layer in range(self.parser_lstm_layers):
        #     x_syn_lstm = self.transformer_lstm[layer].forward(synxs[syn_idx])  
        #     x_syns.append(x_syn_lstm)
        # TODO DepSAWR use the every layers of LSTM, here only use the last layer output
        x_syn_lstm = self.transformer_lstm(lstm_output)  
        x_syns.append(x_syn_lstm)

        x_syn_dep = self.transformer_dep(all_dep)
        x_syns.append(x_syn_dep)

        x_syn_head = self.transformer_head(all_head)
        x_syns.append(x_syn_head)

        x_syn = self.synscale(x_syns)

        return x_syn


if __name__ == '__main__':
    parser_file = "/home/jgy/YaoNLP/pretrained_model/dependency_parser/parser.pt"
    syntax_encoder = SyntaxReprs(parser_file)