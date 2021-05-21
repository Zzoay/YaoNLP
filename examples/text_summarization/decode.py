#Except for the pytorch part content of this file is copied from https://github.com/abisee/pointer-generator/blob/master/
import sys
sys.path.append(r".")   # add the YAONLP path

import os
import time

from rouge import Rouge
import torch
from torch.autograd import Variable

from data_helper import Vocab, TTDataset, ids2words
import config
from model import PointerGenerator
from utils import write_for_rouge, rouge_eval, rouge_log

from yaonlp.data import train_val_split, DataLoader, SortPadCollator
from yaonlp.utils import to_cuda, set_seed

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
    def __init__(self, model_file_path, vocab, id2word):
        model_name = os.path.basename(model_file_path)
        self._decode_dir = os.path.join(config.log_root, 'decode_%s' % (model_name))
        self._rouge_ref_dir = os.path.join(self._decode_dir, 'rouge_ref')
        self._rouge_dec_dir = os.path.join(self._decode_dir, 'rouge_dec_dir')
        for p in [self._decode_dir, self._rouge_ref_dir, self._rouge_dec_dir]:
            if not os.path.exists(p):
                os.mkdir(p)

        self.vocab = vocab
        self.id2word = id2word
        # self.batcher = Batcher(config.decode_data_path, self.vocab, mode='decode',
        #                        batch_size=config.beam_size, single_pass=True)
        # time.sleep(15)

        self.model = PointerGenerator(config=config, vocab_size=len(vocab), model_file=model_file_path)
        if use_cuda:
            self.model = self.model.cuda()

    def sort_beams(self, beams):
        return sorted(beams, key=lambda h: h.avg_log_prob, reverse=True)

    def decode(self, data_iter):
        start = time.time()
        counter = 0
        for batch in data_iter:
            summ, summ_tag, article, article_extend, summ_lens, article_lens, oov_nums = batch
            summ, summ_tag, article, article_extend, summ_lens, article_lens, oov_nums = [data.repeat(config.beam_size, 1) for data in batch]
            summ_lens, article_lens, oov_nums = summ_lens.squeeze(), article_lens.squeeze(), oov_nums.squeeze()
            if use_cuda and torch.cuda.is_available():
                summ, article, article_extend = to_cuda(data=(summ, article, article_extend))
            
            airticle_words = ids2words(article[0].squeeze()[1:], self.id2word) 
            # Run beam search to get best Hypothesis
            best_summary = self.beam_search(enc_inputs=article, 
                                            enc_lens=article_lens,  
                                            enc_inputs_extend=article_extend,
                                            oov_nums=oov_nums,
                                            dec_lens=summ_lens)

            # Extract the output ids from the hypothesis and convert back to words
            output_ids = [int(t) for t in best_summary.tokens[1:]]
            # decoded_words = data.outputids2words(output_ids, self.vocab,
            #                                      (batch.art_oovs[0] if config.use_pgen else None))

            decoded_words = ids2words(output_ids, self.id2word)

            # Remove the [STOP] token from decoded_words, if necessary
            try:
                fst_stop_idx = decoded_words.index('<end>')
                decoded_words = decoded_words[:fst_stop_idx]
            except ValueError:
                decoded_words = decoded_words

            original_abstract_sents = ids2words(summ[0].squeeze()[1:], self.id2word)  # summary repeat beam_size time

            decoded_words = "".join(decoded_words)
            original_abstract_sents = "".join(original_abstract_sents)

            rouge = Rouge()
            rouge_score= rouge.get_scores(decoded_words[:len(original_abstract_sents)], original_abstract_sents)
            rouge_score = rouge.get_scores(str(output_ids), str(summ[0].squeeze()[1:]))

            rouge_1 = rouge_score[0]["rouge-1"]
            rouge_2 = rouge_score[0]["rouge-2"]
            rouge_l = rouge_score[0]["rouge-l"]

            # pyrouge don't support chinese, need to use token
            # decoded_words = [str(ids) for ids in output_ids]
            # original_abstract_sents = [str(ids) for ids in summ[0].squeeze()[1:].tolist()]

            # write_for_rouge(original_abstract_sents, decoded_words, counter,
            #                 self._rouge_ref_dir, self._rouge_dec_dir)
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
                    dec_lens,
                    max_len=150):
        #batch should have only one example

        batch_size, seq_len = enc_inputs.size()

        enc_states, enc_hidden = self.model.encoder(inputs=enc_inputs, seq_lens=enc_lens)
        # encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)
        dec_state = self.model.reduce_state(enc_hidden)

        # (beam_size, hidden_size), (beam_size, hidden_size)
        dec_h, dec_c = dec_state 
        dec_h = dec_h.squeeze()
        dec_c = dec_c.squeeze()

        # initial
        context_vector = torch.zeros(batch_size, config.hidden_size * 2)  # (beam_size, hidden_size * 2)
        coverage_vector = torch.zeros(enc_states.size()[:2])   # (batch_size, seq_len)

        #decoder batch preparation, it has beam_size example initially everything is repeated
        beams = [Beam(tokens=[self.vocab['<start>']],
                      log_probs=[0.0],
                      state=(dec_h[0], dec_c[0]),
                      context = context_vector[0],
                      coverage=(coverage_vector[0] if config.use_coverage else None))
                 for _ in range(config.beam_size)]
        results = []
        steps = 0
        while steps < config.max_dec_steps and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in beams]
            latest_tokens = [t if t < len(self.vocab) else self.vocab['<unk>'] \
                             for t in latest_tokens]
            y_t_1 = Variable(torch.LongTensor(latest_tokens))  # (beam_size, )
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

            # ((1, beam_size, hidden_size), (1, beam_size, hidden_size))
            s_t_1 = (torch.stack(all_state_h, 0).unsqueeze(0), torch.stack(all_state_c, 0).unsqueeze(0))
            c_t_1 = torch.stack(all_context, 0)  # (beam_size, hidden_size * 2)
            if use_cuda:
                c_t_1 = c_t_1.cuda()

            coverage_t_1 = None
            if config.use_coverage:
                all_coverage = []
                for h in beams:
                    all_coverage.append(h.coverage)
                coverage_t_1 = torch.stack(all_coverage, 0)

                if use_cuda:
                    coverage_t_1 = coverage_t_1.cuda()  # (beam_size, seq_len)

            final_dist, dec_state, h_star_t, attn_dist, p_gen, next_coverage =  \
                self.model.decoder(prev_target=y_t_1, 
                                   prev_dec_state=s_t_1, 
                                   enc_states=enc_states, 
                                   enc_input_extend=enc_inputs_extend,
                                   oov_nums=oov_nums,
                                   prev_context_vector=c_t_1,
                                   coverage=coverage_t_1,
                                   dec_lens=dec_lens,
                                   enc_lens=enc_lens)

            log_probs = torch.log(final_dist)  # (beam_size, vocab_size)
            topk_log_probs, topk_ids = torch.topk(log_probs, config.beam_size * 2)  # (beam_size, beam_size * 2), (beam_size, beam_size * 2)

            dec_h, dec_c = dec_state  # (beam_size, hidden_size), (beam_size, hidden_size)
            dec_h = dec_h.squeeze(0)
            dec_c = dec_c.squeeze(0)

            all_beams = []
            num_orig_beams = 1 if steps == 0 else len(beams)
            for i in range(num_orig_beams):
                h = beams[i]
                state_i = (dec_h[i], dec_c[i])
                context_i = h_star_t[i]
                coverage_i = (next_coverage[i] if config.use_coverage else None)

                for j in range(config.beam_size * 2):  # for each of the top 2*beam_size hyps:
                    new_beam = h.extend(token=topk_ids[i, j].item(),
                                   log_prob=topk_log_probs[i, j].item(),
                                   state=state_i,
                                   context=context_i,
                                   coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for h in self.sort_beams(all_beams):
                if h.latest_token == self.vocab['<end>']:
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
    set_seed(config.random_seed)

    sp_collator = SortPadCollator(sort_key=lambda x:x[5], ignore_indics=[4, 5, 6]) 

    vocab, id2word = Vocab(data_file=config.train_data_file, vocab_file=config.vocab_file).read_vocab()
    val_dataset = TTDataset(config=config, vocab=vocab, data_file=config.val_data_file)
    val_iter = DataLoader(dataset=val_dataset,  
                         batch_size=1, 
                        shuffle=config.shuffle, 
                        collate_fn=sp_collator)

    beam_Search_processor = BeamSearch(model_file_path="examples/text_summarization/models/model.pt", vocab=vocab, id2word=id2word)
    beam_Search_processor.decode(val_iter)

