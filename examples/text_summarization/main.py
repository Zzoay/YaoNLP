
import sys
sys.path.append(r".")   # add the YAONLP path

from data_helper import TTDataset, Vocab
from model import PointerGenerator
from trainer import MyTrainer
import config

from yaonlp.data import train_val_split, DataLoader, SortPadCollator


if __name__ == "__main__":
    vocab = Vocab(data_file=config.train_data_file, vocab_file=config.vocab_file).get_vocab()
    train_dataset = TTDataset(vocab=vocab, data_file=config.train_data_file)
    val_dataset = TTDataset(vocab=vocab, data_file=config.val_data_file)

    sp_collator = SortPadCollator(sort_key=lambda x:x[4], ignore_indics=[3, 4, 5]) 

    train_iter = DataLoader(dataset=train_dataset,  
                            batch_size=config.batch_size, 
                            shuffle=config.shuffle, 
                            collate_fn=sp_collator)

    val_iter = DataLoader(dataset=val_dataset,  
                          batch_size=config.batch_size, 
                          shuffle=config.shuffle, 
                          collate_fn=sp_collator)

    model = PointerGenerator(config=config, vocab_size=len(vocab), mode="baseline")

    trainer = MyTrainer(config=config)

    trainer.train(model=model, train_iter=train_iter, val_iter=val_iter)
    trainer.save_model(model=model, save_file=config.model_file)
