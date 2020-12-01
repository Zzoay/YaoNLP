
def clean_data(data_path, name_path, train_path, test_path):
    f1 = open(data_path)
    f2 = open(name_path)

    path = train_path
    for doc, name in zip(f1.readlines(), f2.readlines()):
        name = name.split()
        if name[1] == "test":
            path = test_path
        
        with open(path+"data.txt", "a+") as f:
            f.write(doc) 

        with open(path+"labels.txt", "a+") as f:
            f.write(name[2]+"\n")
    
    f1.close()
    f2.close()


def doc_to_vocab(path, vocab_path):
    vocab = {}
    with open(path) as f:
        for line in f.readlines():
            for word in line.split():
                try:
                    vocab[word] += 1
                except KeyError:
                    vocab[word] = 1
    with open(vocab_path, "w+") as f:
        for word, freq in sorted(vocab.items(), key=lambda item:item[1], reverse=True):
            f.write("{} {} \n".format(word, freq))


if __name__ == "__main__":
    data_path = "data/mr/data.clean.txt"
    name_path = "data/mr/doc_names.txt"

    train_path = "data/mr/train/"
    test_path = "data/mr/test/"

    vocab_path = "data/mr/vocab.txt"

    # clean_data(data_path, name_path, train_path, test_path)
    doc_to_vocab(train_path+"data.txt", vocab_path)