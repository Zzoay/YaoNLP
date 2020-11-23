
from collections import namedtuple
from typing import NamedTuple
# import foo

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from yaonlp import config_loader
from yaonlp.data_helper import MyDataset

from torch.optim import *


class Trainer():
    def __init__(self, model: nn.Module, config: config_loader.Config) -> None:
        self.model = model.cuda()

        self.num_epochs = config.num_epochs
        
        # choose optimizer automatically 
        self.optim = globals()[config.optimizer["type"]](self.model.parameters(), **config.optimizer["parameters"])

    def train(self, train_iter: DataLoader):
        step = 0
        for epoch in range(1, self.num_epochs+1):
            for batch in train_iter:
                batch_x, batch_y = batch
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                self.optim.zero_grad()
                logit = self.model(batch_x)

                loss = F.cross_entropy(logit, batch_y)
                loss.backward()
                self.optim.step()

                if step % 10 == 0:
                    print(f"step: {step}, loss: {loss}")
                step += 1
