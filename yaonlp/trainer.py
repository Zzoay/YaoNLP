
from abc import ABCMeta, abstractmethod
from typing import Callable, Optional

from torch.nn import Module

from yaonlp.data import DataLoader


class Trainer():
    def __init__(self, 
                 loss_fn: Callable, 
                 metrics_fn: Callable, 
                 config: dict) -> None:
        __metaclass__ = ABCMeta
        
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn

        self.config = config

    @abstractmethod
    def train(self, 
              model: Module,
              train_iter: DataLoader, 
              val_iter: DataLoader) -> None:
        raise NotImplementedError

    @abstractmethod
    def eval(self, 
              model: Module,
              test_iter: DataLoader) -> None:
        raise NotImplementedError
