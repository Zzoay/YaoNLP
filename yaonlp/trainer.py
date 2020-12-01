
from collections import namedtuple
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

from yaonlp import config_loader
from yaonlp.data_helper import MyDataset

from torch.optim import *


class Trainer():
    def __init__(self, config: config_loader.Config) -> None:
        self.config = config
        self.num_epochs = config.num_epochs

        self.optim_type = config.optimizer["type"]
        self.optim_params = config.optimizer["parameters"]

    def train(self, model: nn.Module, train_iter: DataLoader, val_iter: DataLoader) -> None:
        if torch.cuda.is_available():
            model.cuda()
        # choose optimizer automatically 
        optim = globals()[self.optim_type](model.parameters(), **self.optim_params)

        step = 0
        for epoch in range(1, self.num_epochs+1):
            for batch in train_iter:
                batch_x, batch_y = batch
                if torch.cuda.is_available():
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                optim.zero_grad()
                logit = model(batch_x)

                loss = F.cross_entropy(logit, batch_y)
                loss.backward()
                optim.step()

                if step % 100 == 0:
                    predicts = torch.max(logit, 1)[1]
                    corrects = (predicts.view(batch_y.size()).data == batch_y.data).float().sum()
                    accuracy = 100.0 * float(corrects/len(batch_y))
                    print('--Step {} , training accuracy : {:.2f} %'.format(step, accuracy))

                    self.test(model, val_iter)
                step += 1

    # test and val func
    def test(self, model: nn.Module, test_iter: DataLoader) -> None:
        corrects, avg_loss = 0, 0
        for batch in test_iter:
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            logit = model(batch_x)
            loss = F.cross_entropy(logit, batch_y)
            
            avg_loss += loss.data
            predicts = torch.max(logit, 1)[1]

            corrects += (predicts.view(batch_y.size()).data == batch_y.data).sum()
        
        size = len(test_iter)
        avg_loss /= size
        corrects = float(corrects)
        accuracy = 100.0 * corrects/size
        print("--Evaluation:")
        print("-loss: {:.6f}  acc: {:.2f}%({}/{})\n".format(avg_loss, accuracy, int(corrects), size))
