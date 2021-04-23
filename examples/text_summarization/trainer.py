
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

from yaonlp.trainer import Trainer
from yaonlp.utils import to_cuda


class MyTrainer(Trainer):
    def __init__(self, config)-> None:
        self.epochs = config.epochs
        self.max_grad_norm = config.max_grad_norm
        self.prt_erery_step = config.prt_erery_step 
        self.eval_every_step = config.eval_every_step
        self.use_cuda = config.use_cuda

    def train(self, model: nn.Module, train_iter: DataLoader, val_iter: DataLoader) -> None:
        model.train()
        if self.use_cuda and torch.cuda.is_available():
            model = model.cuda()

        self.optim = Adam(model.parameters())

        step = 0
        for epoch in range(1, self.epochs + 1):
            for batch in train_iter:
                summ, summ_tag, article, article_extend, summ_lens, article_lens, oov_nums = batch
                if self.use_cuda and torch.cuda.is_available():
                    summ, article, article_extend = to_cuda(data=(summ, article, article_extend))

                self.optim.zero_grad()

                loss = model(enc_inputs=article, 
                             enc_lens=article_lens,  
                             enc_inputs_extend=article_extend,
                             oov_nums=oov_nums,
                             dec_inputs=summ, 
                             dec_tags=summ_tag,
                             dec_lens=summ_lens)

                if step % self.prt_erery_step == 0: 
                    print(f"Epoch {epoch}, steps {step}, loss: {loss.item()}")
                
                if (step+1) % self.eval_every_step == 0:
                    self.eval(model, val_iter)

                    model.train()

                loss.backward()
                clip_grad_norm_(model.parameters(), self.max_grad_norm)
                self.optim.step()

                step += 1
            self.save_model(model, save_file=f"examples/text_summarization/models/model_epoch{epoch}.pt")
        self.eval(model, val_iter)


    def eval(self, model: nn.Module, test_iter: DataLoader) -> None:
        model.eval()

        if self.use_cuda and torch.cuda.is_available():
            model = model.cuda()

        loss_sum = 0
        cnt_batch = 0  # count num of batch
        for batch in test_iter:
            with torch.no_grad():
                summ, summ_tag, article, article_extend, summ_lens, article_lens, oov_nums = batch
                if self.use_cuda and torch.cuda.is_available():
                    summ, article, article_extend = to_cuda(data=(summ, article, article_extend))


                loss = model(enc_inputs=article, 
                                enc_lens=article_lens,  
                                enc_inputs_extend=article_extend,
                                oov_nums=oov_nums,
                                dec_inputs=summ, 
                                dec_tags=summ_tag,
                                dec_lens=summ_lens)

                loss_sum += loss
            
            cnt_batch += 1
        loss_avg = loss_sum / cnt_batch
        self.loss_avg = loss_avg
        print(f"--Evaluation, loss: {loss_avg} \n")
    
    def save_model(self, model: nn.Module, save_file: str):
        state = {
            'iter': iter,
            'encoder_state_dict': model.encoder.state_dict(),
            'decoder_state_dict': model.decoder.state_dict(),
            'reduce_state_dict': model.reduce_state.state_dict(),
            'optimizer': self.optim.state_dict(),
            'current_loss': self.loss_avg
        }
        torch.save(state, save_file)