
from abc import ABCMeta, abstractmethod
from typing import Callable, Optional

from torch.nn import Module

from yaonlp.data import MyDataLoader


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
              train_iter: MyDataLoader, 
              val_iter:Optional[MyDataLoader] = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def eval(self, 
              model: Module,
              val_iter: MyDataLoader) -> None:
        raise NotImplementedError
