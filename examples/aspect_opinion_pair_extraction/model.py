
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel

from yaonlp.layers import NonLinear, Biaffine, BiLSTM

from relation_attention import RelationAttention


class APOE(nn.Module):
    def __init__(self, args, config, label_alphabet):
        super(APOE, self).__init__()
        print("build network...")
        self.gpu = args.ifgpu
        self.label_size = label_alphabet.size()
        self.bert_encoder_dim = config.hidden_size
        self.target_hidden_dim = args.target_hidden_dim
        self.relation_hidden_dim = args.relation_hidden_dim
        self.relation_threds = args.relation_threds
        self.drop = args.dropout
        self.step = args.step

        self.hidden_mask, self.score_mask = self.check_trick(args.trick)

        # encoder
        self.bert = BertModel.from_pretrained('bert-base-uncased') # the bert model path, including pytorch.bin and config.json
        
        self.target_mlp = NonLinear(in_features=self.bert_encoder_dim, 
                                    out_features=self.target_hidden_dim, 
                                    activation=torch.tanh,
                                    initial=init.xavier_uniform_)

        self.rel_mlp = NonLinear(in_features=self.bert_encoder_dim, 
                                 out_features=self.relation_hidden_dim, 
                                 activation=torch.tanh,
                                 initial=init.xavier_uniform_)

        self.hidden2tag = NonLinear(in_features=self.target_hidden_dim,
                                    out_features=self.label_size + 2,  # add <start> and <end> tag
                                    activation=None,     
                                    initial=init.xavier_uniform_)
        
        self.rel_att = RelationAttention(args)

    def check_trick(self, trick):
        if trick == "All":
            return True, True
        elif trick == "HiddenMask":
            return True, False
        elif trick == "ScoreMask":
            return False, True
        return False, False
    
    def neg_log_likelihood_loss(self, all_input_ids, all_segment_ids, all_labels, all_relations, all_input_mask):
        batch_size = all_input_ids.size(0)
        seq_len = all_input_ids.size(1)
        mask_tmp1 = all_input_mask.view(batch_size, 1, seq_len).repeat(1, seq_len, 1)
        mask_tmp2 = all_input_mask.view(batch_size, seq_len, 1).repeat(1, 1, seq_len)
        maskMatrix = mask_tmp1 * mask_tmp2

        target_pred, r_pred, tag_seq = self.main_structure(maskMatrix, all_input_ids, all_segment_ids, self.step,
                                                          all_input_mask)
        # target Loss
        target_loss = self.crf.neg_log_likelihood_loss(target_pred, all_input_mask.byte(), all_labels)
        # scores, tag_seq = self.crf._viterbi_decode(target_pred, all_input_mask.byte())
        target_loss = target_loss / batch_size

        # relation Loss
        weight = torch.FloatTensor([0.01, 1.0]).cuda()
        relation_loss_function = nn.CrossEntropyLoss(weight=weight)
        relationScoreLoss = r_pred.view(-1, 1)
        relationlabelLoss = all_relations.view(batch_size * seq_len * seq_len)
        relationScoreLoss = torch.cat([1 - relationScoreLoss, relationScoreLoss], 1)
        relation_loss = relation_loss_function(relationScoreLoss, relationlabelLoss)

        return target_loss, relation_loss, tag_seq, r_pred

    def forward(self, all_input_ids, all_segment_ids, all_input_mask):
        batch_size, seq_len = all_input_ids.shape
        mask_tmp1 = all_input_mask.view(batch_size, 1, seq_len).repeat(1, seq_len, 1)
        mask_tmp2 = all_input_mask.view(batch_size, seq_len, 1).repeat(1, 1, seq_len)
        maskMatrix = mask_tmp1 * mask_tmp2

        target_pred, r_pred, tag_seq = self.main_structure(maskMatrix, all_input_ids, all_segment_ids, self.step,
                                                          all_input_mask)
        return tag_seq, r_pred

    def main_structure(self, maskMatrix, all_input_ids, all_segment_ids, steps, all_input_mask):
        batch_size, seq_len = all_input_ids.shape
        # bert
        all_outputs = self.bert(all_input_ids, all_segment_ids, all_input_mask)
        sequence_output = all_outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        target_hidden = self.target_mlp(sequence_output)

        target_pred = self.hidden2tag(target_hidden)
        tag_score, tag_seq = self.crf._viterbi_decode(target_pred, all_input_mask.byte())

        # while useful tags (like 'B-T',...) start from 2 to 5
        # {"O":1, "B-T":2, "I-T":3,"B-P":4,"I-P":5}
        tag_mask = tag_seq.gt(1)

        # use tag_score to calculate masks
        # hidden_entity_score = tag_score.view(batch_size, seq_len, 1).repeat(1, 1, self.relation_hidden_dim)

        relation_hidden = self.rel_mlp(sequence_output)  # b,s,relation_hidden_dim
        if self.hidden_mask:
            hidden_entity_masks = tag_mask.view(batch_size, seq_len, 1).repeat(1, 1, self.relation_hidden_dim)
            relation_hidden = relation_hidden * hidden_entity_masks
 
        relation_score = self.rel_att(relation_hidden) # b,s,s 
        if self.score_mask:
            mask_tmp1 = tag_mask.view(batch_size, 1, seq_len).repeat(1, seq_len, 1).float()
            mask_tmp2 = tag_mask.view(batch_size, seq_len, 1).repeat(1, 1, seq_len).float()
            score_entity_masks = mask_tmp1 * mask_tmp2

            relation_score = relation_score * score_entity_masks

        rel_pred = relation_score * (maskMatrix.float())  # b,s,s

        return target_pred, rel_pred, tag_seq