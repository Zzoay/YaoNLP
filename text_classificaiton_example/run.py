
import sys
sys.path.append("C:\\Users\\Admin\\Desktop\\YaoNLP")

import torch
from torch.utils.data import DataLoader

from yaonlp import config_loader
from yaonlp.data_helper import MyDataset, MyDataLoader
from yaonlp.trainer import Trainer
from yaonlp.config_loader import Config

from model import TextCNN


torch.cuda.set_device(0)


if __name__ == "__main__":
    data_config = config_loader.load_config("text_classificaiton_example/config/data.json")
    model_config = config_loader.load_config("text_classificaiton_example/config/model.json")
    trainer_config = config_loader.load_config("text_classificaiton_example/config/trainer.json")

    train_dataset = MyDataset(data_config, train=True)
    test_dataset = MyDataset(data_config, train=False)

    train_iter = MyDataLoader(train_dataset, config=data_config)
    test_iter = MyDataLoader(test_dataset, config=data_config)
    
    model = TextCNN(model_config)
    trainer = Trainer(model, trainer_config)

    trainer.train(train_iter)