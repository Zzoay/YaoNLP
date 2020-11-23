
import sys
sys.path.append("C:\\Users\\Admin\\Desktop\\YaoNLP")

import torch
from torch.utils.data import DataLoader

from yaonlp import config_loader
from yaonlp.data_helper import MyDataset
from yaonlp.trainer import Trainer
from yaonlp.config_loader import Config

from model import TextCNN


torch.cuda.set_device(0)


if __name__ == "__main__":
    data_config = config_loader.load_config("text_classificaiton_example/config/data.json")

    train_data_path = data_config.train_data_path
    test_data_path = data_config.test_data_path

    train_labels_path = data_config.train_labels_path
    test_labels_path = data_config.test_labels_path

    vocab_path = data_config.vocab_path

    shuffle = data_config.shuffle
    batch_size = data_config.batch_size
    max_len = data_config.max_len

    train_dataset = MyDataset(train_data_path, train_labels_path, vocab_path, max_len)
    test_dataset = MyDataset(test_data_path, test_labels_path, vocab_path, max_len)

    train_iter = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_iter = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    
    model_config = config_loader.load_config("text_classificaiton_example/config/model.json")
    trainer_config = config_loader.load_config("text_classificaiton_example/config/trainer.json")

    model = TextCNN(model_config)
    trainer = Trainer(model, trainer_config)

    trainer.train(train_iter)