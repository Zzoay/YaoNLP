
from torch import nn

from yaonlp.layers import NonLinear

from syntax_enhace.scale_mix import ScaleMix


class EnhancedEncoder(nn.Module):
    def __init__(self, config, parser_config, input_dims, bert_layers):
        super(EnhancedEncoder, self).__init__()
        self.config = config
        self.input_dims = input_dims
        self.input_depth = bert_layers if config.bert_tune == 0 else 1
        self.hidden_dims = 2 * config.lstm_hiddens
        self.projections = nn.ModuleList(
            [NonLinear(self.input_dims, self.hidden_dims, activation=GELU())
             for i in range(self.input_depth)])

        self.rescale = ScalarMix(mixture_size=self.input_depth)

        self.transformer_emb = NonLinear(parser_config.word_dims, self.hidden_dims, activation=GELU())

        parser_dim = 2 * parser_config.lstm_hiddens
        self.transformer_lstm = nn.ModuleList(
            [NonLinear(parser_dim, self.hidden_dims, activation=GELU())
             for i in range(parser_config.lstm_layers)])

        parser_mlp_dim = parser_config.mlp_arc_size + parser_config.mlp_rel_size
        self.transformer_dep = NonLinear(parser_mlp_dim, self.hidden_dims, activation=GELU())
        self.transformer_head = NonLinear(parser_mlp_dim, self.hidden_dims, activation=GELU())

        self.parser_lstm_layers = parser_config.lstm_layers
        self.synscale = ScalarMix(mixture_size=3+parser_config.lstm_layers)

    def forward(self, inputs, synxs):
        proj_hiddens = []
        for idx, input in enumerate(inputs):
            cur_hidden = self.projections[idx](input)
            proj_hiddens.append(cur_hidden)

        word_represents = self.rescale(proj_hiddens)

        syn_idx = 0
        x_syns = []
        x_syn_emb = self.transformer_emb(synxs[syn_idx])
        x_syns.append(x_syn_emb)
        syn_idx += 1

        for layer in range(self.parser_lstm_layers):
            x_syn_lstm = self.transformer_lstm[layer].forward(synxs[syn_idx])
            syn_idx += 1
            x_syns.append(x_syn_lstm)

        x_syn_dep = self.transformer_dep(synxs[syn_idx])
        x_syns.append(x_syn_dep)
        syn_idx += 1

        x_syn_head = self.transformer_head(synxs[syn_idx])
        x_syns.append(x_syn_head)
        syn_idx += 1

        x_syn = self.synscale(x_syns)
        if self.training:
            word_represents, x_syn = drop_bi_input_independent(word_represents, x_syn, self.config.dropout_emb)

        x_lexical = torch.cat((word_represents, x_syn), dim=2)

        return x_lexical


def drop_bi_input_independent(word_embeddings, tag_embeddings, dropout_emb):
    batch_size, seq_length, _ = word_embeddings.size()
    word_masks = word_embeddings.new_full((batch_size, seq_length), 1-dropout_emb)
    word_masks = torch.bernoulli(word_masks)
    tag_masks = tag_embeddings.new_full((batch_size, seq_length), 1-dropout_emb)
    tag_masks = torch.bernoulli(tag_masks)
    scale = 2.0 / (word_masks + tag_masks + 1e-12)
    word_masks *= scale
    tag_masks *= scale
    word_masks = word_masks.unsqueeze(dim=2)
    tag_masks = tag_masks.unsqueeze(dim=2)
    word_embeddings = word_embeddings * word_masks
    tag_embeddings = tag_embeddings * tag_masks

    return word_embeddings, tag_embeddings