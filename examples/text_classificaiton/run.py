
import sys
sys.path.append(r"C:\Users\Admin\Desktop\YaoNLP")

from model import TextCNN
from data_helper import MyDataset
from trainer import MyTrainer
from functional import cross_entropy, compute_acc

from yaonlp.config_loader import load_config
from yaonlp.data_helper import MyDataLoader, train_val_split


if __name__ == "__main__":
    data_config = load_config("examples/text_classificaiton/config/data.json")
    model_config = load_config("examples/text_classificaiton/config/model.json")
    trainer_config = load_config("examples/text_classificaiton/config/trainer.json")

    train_dataset = MyDataset(data_config, train=True)
    test_dataset = MyDataset(data_config, train=False)

    train_dataset, val_dataset = train_val_split(train_dataset, data_config)

    train_iter = MyDataLoader(train_dataset, config=data_config)
    val_iter = MyDataLoader(val_dataset, config=data_config)
    test_iter = MyDataLoader(test_dataset, config=data_config)

    model = TextCNN(model_config)
    
    trainer = MyTrainer(loss_fn=cross_entropy, metrics_fn=compute_acc, config=trainer_config)

    trainer.train(model, train_iter, val_iter)
    trainer.eval(model, test_iter)
