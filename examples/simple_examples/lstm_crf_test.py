
import sys
sys.path.append(r".")   # add the YAONLP path

from typing import Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim

from yaonlp.layers import CRF

# inspired by: 
# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
# https://github.com/mali19064/LSTM-CRF-pytorch-faster/blob/master/LSTM_CRF_faster_parallel.py

def prepare_sequence(seq, to_ix):
    ixs = [to_ix[w] for w in seq]
    # return torch.tensor(ixs, dtype=torch.long)
    return ixs


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, to_cuda):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tag_size = len(tag_to_ix)
        self.to_cuda = to_cuda

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim // 2,
                            num_layers=1, 
                            bidirectional=True, 
                            batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tag_size)

        self.crf = CRF(tag_size=self.tag_size, to_cuda=self.to_cuda)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _get_lstm_features(self, sentence):
        # self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)  # batch_size, seq_len, embedding_dim
        lstm_out, self.hidden = self.lstm(embeds)  # batch_size, seq_len, lstm_hidden
        # lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)  # batch_size, seq_len, tag_size
        return lstm_feats

    def compute_loss(self, sentences, tags):
        lstm_feats = self._get_lstm_features(sentences)
        return self.crf.neg_log_likelihood(lstm_feats, tags)

    def forward(self, sentence):  
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        return self.crf.forward(lstm_feats)
    

if __name__ == "__main__":
    START_TAG = "<START>"
    END_TAG = "<END>"
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4
    use_cuda = True

    # Make up some training data, padded already
    training_data = [(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia <pad> <pad> <pad> <pad>".split(),
        "B I O O O O B O O O O".split()
    ),(
        "the wall street journal reported today that apple corporation made money".split(),
        "B I I I O O O B I O O".split()
    ), (
        "georgia tech is a university in georgia <pad> <pad> <pad> <pad>".split(),
        "B I O O O O B O O O O".split()
    )]

    word_to_ix: dict = {}
    for sentence, _ in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"B": 0, "I": 1, "O": 2, START_TAG: 3, END_TAG: 4}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, to_cuda=True)
    if use_cuda:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    batch_size = 2

    # Check predictions before training
    with torch.no_grad():
        sentence_in = torch.tensor([prepare_sequence(sentence, word_to_ix) for sentence,_ in training_data[0:batch_size]], dtype=torch.long)
        if use_cuda:
            sentence_in = sentence_in.cuda()
        score, path = model(sentence_in)
        print(score)
        print(path)

    step = 0
    for epoch in range(300):  
        idx = 0
        for i in range(0, len(training_data), batch_size):
            sentences: Tuple[str, ...] = tuple()  # type annotated
            tags: Tuple[str, ...] = tuple()
            sentences, tags = zip(*training_data[idx:idx + batch_size])
            idx += batch_size
            model.zero_grad()

            sentence_in = torch.tensor([prepare_sequence(sentence, word_to_ix) for sentence in sentences], dtype=torch.long)
            targets = torch.tensor([prepare_sequence(tag, tag_to_ix) for tag in tags], dtype=torch.long)
            if use_cuda:
                sentence_in = sentence_in.cuda()
                targets = targets.cuda()

            loss = model.compute_loss(sentence_in, targets)
            if step % 20 == 0:
                print(f"epoch: {epoch}, step: {step}, loss: {loss.item()}")

            loss.backward()
            optimizer.step()

            step += 1

    # Check predictions after training
    with torch.no_grad():
        precheck_sent = torch.tensor([prepare_sequence(sentence, word_to_ix) for sentence in sentences], dtype=torch.long)
        if use_cuda:
            precheck_sent = precheck_sent.cuda()
        path_score, best_path = model(precheck_sent)
        print(path_score)
        print(best_path)
