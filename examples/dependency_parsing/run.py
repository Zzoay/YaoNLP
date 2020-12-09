
import sys
sys.path.append(r"C:\Users\Admin\Desktop\YaoNLP")

from data_helper import CTBDataset
from trainer import MyTrainer
from vocab import Vocab
from model import DependencyParser
from functional import arc_rel_loss, uas_las

from yaonlp.config_loader import load_config
from yaonlp.data import MyDataLoader, SortPadCollator, train_val_split
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

    train_dataset, val_dataset = train_val_split(dataset, data_config)

    sp_collator = SortPadCollator(sort_key=lambda x:x[5], ignore_index=5)   # while idx '5' indicates the lenght of sentence

    train_iter = MyDataLoader(train_dataset, data_config, collate_fn=sp_collator)
    val_iter = MyDataLoader(val_dataset, data_config, collate_fn=sp_collator)

    model  = DependencyParser(vocab_size=vocab.word_size, tag_size=vocab.tag_size, rel_size=vocab.rel_size, config=model_config)
    
    trainer = MyTrainer(loss_fn=arc_rel_loss, metrics_fn=uas_las, config=trainer_config)

    trainer.train(model=model, train_iter=train_iter, val_iter=val_iter)
    trainer.eval(model=model, eval_iter=val_iter)
    print("finished.")
