
import sys
sys.path.append(r".")   # add the YAONLP path

from model import TextCNN
from data_helper import MyDataset
from trainer import MyTrainer
from functional import cross_entropy, compute_acc

from yaonlp.config_loader import load_config
from yaonlp.data import MyDataLoader, train_val_split, DataLoader


if __name__ == "__main__":
    data_config = load_config("examples/text_classificaiton/config/data.json")
    model_config = load_config("examples/text_classificaiton/config/model.json")
    trainer_config = load_config("examples/text_classificaiton/config/trainer.json")

    train_dataset = MyDataset(data_config, train=True)
    test_dataset = MyDataset(data_config, train=False)

    train_dataset, val_dataset = train_val_split(train_dataset, data_config["val_ratio"])

    # train_iter = MyDataLoader(train_dataset, config=data_config)
    # val_iter = MyDataLoader(val_dataset, config=data_config)
    # test_iter = MyDataLoader(test_dataset, config=data_config)
    train_iter = DataLoader(dataset=train_dataset, 
                            batch_size=data_config["batch_size"], 
                            shuffle=data_config["shuffle"])
    val_iter = DataLoader(dataset=val_dataset,
                          batch_size=data_config["batch_size"], 
                          shuffle=data_config["shuffle"])
    test_iter = DataLoader(dataset=test_dataset, 
                           batch_size=data_config["batch_size"], 
                           shuffle=data_config["shuffle"])

    model = TextCNN(model_config)
    
    trainer = MyTrainer(loss_fn=cross_entropy, metrics_fn=compute_acc, config=trainer_config)

    trainer.train(model, train_iter, val_iter)
    trainer.eval(model, test_iter)
