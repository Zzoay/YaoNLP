
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from yaonlp.trainer import Trainer


class MyTrainer(Trainer):
    def __init__(self) -> None:

        return

    def train(self, model: nn.Module, train_iter: DataLoader, val_iter: DataLoader) -> None:

        return
    
    def eval(self, model: nn.Module, test_iter: DataLoader) -> None:

        return