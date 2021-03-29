#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/

import os
import time

import torch
from torch.autograd import Variable

from data_helper import Vocab
import config
from model import PointerGenerator
from data_util.utils import write_for_rouge, rouge_eval, rouge_log


use_cuda = config.use_cuda and torch.cuda.is_available()


class Beam(object):
  def __init__(self, tokens, log_probs, state, context, coverage):
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.context = context
    self.coverage = coverage

  def extend(self, token, log_prob, state, context, coverage):
    return Beam(tokens = self.tokens + [token],
                log_probs = self.log_probs + [log_prob],
                state = state,
                context = context,
                coverage = coverage)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def avg_log_prob(self):
    return sum(self.log_probs) / len(self.tokens)


class BeamSearch(object):
    def __init__(self, model_file_path):
        model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
                               batch_size=config.beam_size, single_pass=True)
        time.sleep(15)

        self.model = Model(model_file_path, is_eval=True)

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self, data_iter):
        start = time.time()
        counter = 0
        for batch in train_iter:
            summ, article, article_extend, summ_lens, article_lens, oov_nums = batch
            if use_cuda and torch.cuda.is_available():
                summ, article, article_extend = to_cuda(data=(summ, article, article_extend))
            
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(batch)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            decoded_words = data.outputids2words(output_ids, self.vocab,
                                                 (batch.art_oovs[0] if config.pointer_gen else None))

            # Remove the [STOP] token from decoded_words, if necessary
            # try:
            #     fst_stop_idx = decoded_words.index(data.STOP_DECODING)
            #     decoded_words = decoded_words[:fst_stop_idx]
            # except ValueError:
            #     decoded_words = decoded_words

            original_abstract_sents = summ

            write_for_rouge(original_abstract_sents, decoded_words, counter,
                            self._rouge_ref_dir, self._rouge_dec_dir)
            counter += 1
            if counter % 1000 == 0:
                print('%d example in %d sec'%(counter, time.time() - start))
                start = time.time()

        print("Decoder has finished reading dataset for single_pass.")
        print("Now starting ROUGE eval...")
        results_dict = rouge_eval(self._rouge_ref_dir, self._rouge_dec_dir)
        rouge_log(results_dict, self._decode_dir)

    def beam_search(self,
                    enc_inputs,
                    enc_lens, 
                    enc_inputs_extend,
                    oov_nums,
                    dec_inputs, 
                    dec_lens,
                    max_len=150):
        #batch should have only one example

        enc_states, enc_hidden = self.model.encoder(inputs=enc_inputs, seq_lens=enc_lens)
        # encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        dec_state = self.model.reduce_state(enc_hidden)

        dec_h, dec_c = dec_state # 1 x 2*hidden_size
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # initial
        context_vector = torch.zeros(batch_size, self.hidden_size * 2)
        coverage_vector = torch.zeros(enc_states.size()[:2])   # (batch_size, seq_len)

        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab.word2id(data.START_DECODING)],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = context_vector[0],
                      coverage=(coverage_vector[0] if config.is_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < self.vocab.size() else self.vocab.word2id(data.UNKNOWN_TOKEN) \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))
            if use_cuda:
                y_t_1 = y_t_1.cuda()
            all_state_h =[]
            all_state_c = []

            all_context = []

            for h in beams:
                state_h, state_c = h.state
                all_state_h.append(state_h)
                all_state_c.append(state_c)

                all_context.append(h.context)

            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)

            coverage_t_1 = None
            if config.is_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

            # final_dist, s_t, c_t, attn_dist, p_gen, coverage_t = self.model.decoder(y_t_1, s_t_1,
            #                                             encoder_outputs, encoder_feature, enc_padding_mask, c_t_1,
            #                                             extra_zeros, enc_batch_extend_vocab, coverage_t_1, steps)

            final_dist, dec_state, h_star_t, attn_dist, p_gen, next_coverage =  \
                self.model.decoder(prev_target=dec_prev, 
                             prev_dec_state=dec_state, 
                             enc_states=enc_states, 
                             enc_input_extend=enc_inputs_extend,
                             oov_nums=oov_nums,
                             prev_context_vector=c_t_1,
                             prev_coverage=coverage_t_1)

            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)

            dec_h, dec_c = dec_state
            dec_h = dec_h.squeeze()
            dec_c = dec_c.squeeze()

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = h_star_t[i]
                coverage_i = (next_coverage[i] if config.is_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab.word2id(data.STOP_DECODING):
                    if steps >= config.min_dec_steps:
                        results.append(h)
                else:
                    beams.append(h)
                if len(beams) == config.beam_size or len(results) == config.beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = beams

        beams_sorted = self.sort_beams(results)

        return beams_sorted[0]

if __name__ == '__main__':
    model_filename = ""
    beam_Search_processor = BeamSearch(model_filename)
    beam_Search_processor.decode()

