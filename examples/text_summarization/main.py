
import sys
sys.path.append(r".")   # add the YAONLP path

from data_helper import TTDataset, Vocab
from model import PointerGenerator
from trainer import MyTrainer

from yaonlp.data import train_val_split, DataLoader, SortPadCollator


if __name__ == "__main__":
    data_file = r"data\TTNewsCorpus_NLPCC2017\toutiao4nlpcc\train_with_summ.txt"
    vocab_file = r"data\TTNewsCorpus_NLPCC2017\vocab.txt"

    val_ratio = 0.2
    batch_size = 64
    shuffle = True

    input_size = 0 
    emb_dim = 300
    hidden_size = 100
    dropout = 0

    vocab = Vocab(data_file=data_file, vocab_file=vocab_file).get_vocab()
    dataset = TTDataset(vocab=vocab, data_file=data_file)

    train_dataset, val_dataset = train_val_split(train_dataset=dataset, val_ratio=val_ratio)

    sp_collator = SortPadCollator(sort_key=lambda x:x[4], ignore_indics=[3, 4, 5]) 

    train_iter = DataLoader(dataset=train_dataset,  
                            batch_size=batch_size, 
                            shuffle=shuffle, 
                            collate_fn=sp_collator)

    val_iter = DataLoader(dataset=val_dataset,  
                          batch_size=batch_size, 
                          shuffle=shuffle, 
                          collate_fn=sp_collator)

    model = PointerGenerator(input_size=input_size,
                             vocab_size=len(vocab), 
                             emb_dim=emb_dim,
                             hidden_size=hidden_size, 
                             dropout=dropout)

    trainer = MyTrainer()

    trainer.train(model=model, train_iter=train_iter, val_iter=val_iter)

    print()