
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from yaonlp.trainer import Trainer
from yaonlp.utils import to_cuda

class MyTrainer(Trainer):
    def __init__(self, config)-> None:
        self.epochs = config.epochs
        self.prt_erery_step = config.prt_erery_step 
        self.eval_every_step = config.eval_every_step
        self.use_cuda = config.use_cuda

    def train(self, model: nn.Module, train_iter: DataLoader, val_iter: DataLoader) -> None:
        model.train()
        if self.use_cuda and torch.cuda.is_available():
            model = model.cuda()

        optim = Adam(model.parameters())

        step = 0
        for epoch in range(1, self.epochs + 1):
            for batch in train_iter:
                summ, article, article_extend, summ_lens, article_lens, oov_nums = batch
                if self.use_cuda and torch.cuda.is_available():
                    summ, article, article_extend = to_cuda(data=(summ, article, article_extend))

                optim.zero_grad()

                loss = model(enc_inputs=article, 
                             enc_lens=article_lens,  
                             enc_inputs_extend=article_extend,
                             oov_nums=oov_nums,
                             dec_inputs=summ, 
                             dec_lens=summ_lens)

                if step % self.prt_erery_step == 0: 
                    print(f"Steps {step}, loss: {loss.item()}")
                
                if (step+1) % self.eval_every_step == 0:
                    self.eval(model, val_iter)

                    model.train()

                loss.backward()
                
                optim.step()

                step += 1
        self.eval(model, val_iter)
        
    def eval(self, model: nn.Module, test_iter: DataLoader) -> None:
        model.eval()

        if self.use_cuda and torch.cuda.is_available():
            model = model.cuda()

        loss_sum = 0
        for epoch in range(1, self.epochs + 1):
            cnt_batch = 0  # count num of batch
            for batch in test_iter:
                with torch.no_grad():
                    summ, article, article_extend, summ_lens, article_lens, oov_nums = batch
                    if self.use_cuda and torch.cuda.is_available():
                        summ, article, article_extend = to_cuda(data=(summ, article, article_extend))


                    loss = model(enc_inputs=article, 
                                enc_lens=article_lens,  
                                enc_inputs_extend=article_extend,
                                oov_nums=oov_nums,
                                dec_inputs=summ, 
                                dec_lens=summ_lens)

                    loss_sum += loss
                
                cnt_batch += 1

        loss_avg = loss_sum / cnt_batch
        print(f"--Evaluation, loss: {loss_avg} \n")
    
    def save_model(self, model: nn.Module, save_file: str):
        # TODO save model with experiment result
        torch.save(model.state_dict(), save_file)