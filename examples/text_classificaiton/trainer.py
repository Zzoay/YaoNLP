
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from yaonlp.data import DataLoader
from yaonlp.optim import OptimChooser
from yaonlp.trainer import Trainer


class MyTrainer(Trainer):
    def __init__(self, loss_fn: Callable, metrics_fn: Callable, config: dict):
        super(MyTrainer, self).__init__(loss_fn, metrics_fn, config)

        self.num_epochs = config["num_epochs"]

        self.optim_type = config["optimizer"]["type"]
        self.optim_params = config["optimizer"]["parameters"]

    def train(self, model: nn.Module, train_iter: DataLoader, val_iter: DataLoader) -> None:
        model.train()
        if torch.cuda.is_available():
            model.cuda()

        optim_chooser = OptimChooser(model.parameters(), 
                                     optim_type=self.optim_type, 
                                     optim_params=self.optim_params)
        optim = optim_chooser.optim()

        step = 0
        for epoch in range(1, self.num_epochs+1):
            for batch in train_iter:
                batch_x, batch_y = batch
                if torch.cuda.is_available():
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

                optim.zero_grad()
                logit = model(batch_x)

                loss = self.loss_fn(logit, batch_y)
                loss.backward()
                optim.step()

                if step % 100 == 0:
                    accuracy = self.metrics_fn(logit, batch_y)
                    print('--Step {} , training accuracy : {:.2f} %'.format(step, accuracy))

                    self.eval(model, val_iter)
                    model.train()
                step += 1

    # eval func
    def eval(self, model: nn.Module, test_iter: DataLoader) -> None:
        model.eval()
        corrects, avg_loss = 0, 0.0
        for batch in test_iter:
            batch_x, batch_y = batch
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            logit = model(batch_x)
            loss = F.cross_entropy(logit, batch_y)
            
            avg_loss += float(loss.data)
            predicts = torch.max(logit, 1)[1]

            corrects += (predicts.view(batch_y.size()).data == batch_y.data).sum()
        
        size = len(test_iter.dataset)
        avg_loss = avg_loss / size
        accuracy = 100.0 * corrects/size
        print("--Evaluation:")
        print("-loss: {:.6f}  acc: {:.2f}%({}/{})\n".format(avg_loss, accuracy, int(corrects), size))