
import sys
sys.path.append("C:\\Users\\Admin\\Desktop\\YaoNLP")

import torch
from torch.utils.data import DataLoader

from yaonlp import config_loader
from yaonlp.data_helper import MyDataset, MyDataLoader, train_val_split
from yaonlp.trainer import Trainer
from yaonlp.config_loader import Config

from model import TextCNN


if __name__ == "__main__":
    data_config = config_loader.load_config("examples/text_classificaiton/config/data.json")
    model_config = config_loader.load_config("examples/text_classificaiton/config/model.json")
    trainer_config = config_loader.load_config("examples/text_classificaiton/config/trainer.json")

    train_dataset = MyDataset(data_config, train=True)
    test_dataset = MyDataset(data_config, train=False)

    train_dataset, val_dataset = train_val_split(train_dataset, data_config)

    train_iter = MyDataLoader(train_dataset, config=data_config)
    val_iter = MyDataLoader(val_dataset, config=data_config)
    test_iter = MyDataLoader(test_dataset, config=data_config)

    model = TextCNN(model_config)
    trainer = Trainer(trainer_config)

    trainer.train(model, train_iter, val_iter)
    trainer.test(model, test_iter)
