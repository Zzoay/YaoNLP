
from data_helper import CTBDataset
from trainer import Trainer
from vocab import Vocab
from model import DependencyParser

from yaonlp.config_loader import load_config
from yaonlp.data_helper import MyDataLoader, SortPadCollator
from yaonlp.optim import OptimChooser
from yaonlp.utils import set_seed


if __name__ == "__main__":
    data_path = "data/ctb8.0/dep/"
    vocab_file = "data/ctb8.0/processed/vocab.txt"

    data_config = load_config("examples/dependency_parsing/config/data.json", mode="dict")
    model_config = load_config("examples/dependency_parsing/config/model.json", mode="dict")
    trainer_config = load_config("examples/dependency_parsing/config/trainer.json", mode="dict")

    set_seed(trainer_config["seed"])

    vocab = Vocab(data_config)
    dataset = CTBDataset(vocab, data_config)

    sp_collator = SortPadCollator(sort_key=lambda x:x[5], ignore_index=5)   # while idx '5' indicates the lenght of sentence
    data_iter = MyDataLoader(dataset, data_config, collate_fn=sp_collator)

    model  = DependencyParser(vocab_size=vocab.word_size, tag_size=vocab.tag_size, rel_size=vocab.rel_size, config=model_config)

    optim_chooser = OptimChooser(model.parameters(), 
                                 optim_type=trainer_config["optimizer"]["type"], 
                                 optim_params=trainer_config["optimizer"]["parameters"])
    optim = optim_chooser.optim()

    trainer = Trainer(trainer_config)
    trainer.train(model=model, optim=optim, train_iter=data_iter)

    print("finished.")
