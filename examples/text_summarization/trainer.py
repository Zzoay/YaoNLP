
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

from yaonlp.trainer import Trainer


class MyTrainer(Trainer):
    def __init__(self, epochs=1) -> None:
        self.epochs = epochs
        return

    def train(self, model: nn.Module, train_iter: DataLoader, val_iter: DataLoader) -> None:
        model.train()

        optim = Adam(model.parameters())

        for epoch in range(1, self.epochs + 1):
            for batch in train_iter:
                summ, article, article_extend, summ_lens, article_lens, oov_nums = batch

                optim.zero_grad()

                loss = model(enc_inputs=article, 
                             enc_lens=article_lens,  
                             enc_inputs_extend=article_extend,
                             oov_nums=oov_nums,
                             dec_inputs=summ, 
                             dec_lens=summ_lens)

                print(loss)
                loss.backward()
                
                optim.step()
        return
    
    def eval(self, model: nn.Module, test_iter: DataLoader) -> None:

        return