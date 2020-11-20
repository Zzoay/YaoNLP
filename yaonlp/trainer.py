
from torch.utils.data import DataLoader

import config_loader
from data_helper import MyDataset


if __name__ == "__main__":
    config = config_loader._load_json("config_example/data.json")

    train_path = config["train_data_path"]
    test_path = config["test_data_path"]

    vocab_path = config["vocab_path"]

    train_data_path = config["train_data_path"]
    test_data_path = config["test_data_path"]

    train_labels_path = config["train_labels_path"]
    test_labels_path = config["test_labels_path"]

    shuffle = config["shuffle"]
    batch_size = config["batch_size"]
    max_len = config["max_len"]

    train_dataset = MyDataset(train_data_path, train_labels_path, vocab_path, max_len)
    test_dataset = MyDataset(test_data_path, test_labels_path, vocab_path, max_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    print()