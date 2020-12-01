
import sys
sys.path.append("C:\\Users\\Admin\\Desktop\\YaoNLP")

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn.functional import cross_entropy

from data_helper import CTBDataset, load_ctb, MyDataLoader
from model import Biaffine
from trainer import Trainer
from vocab import Vocab
from model import DependencyParser

from yaonlp.config_loader import load_config


def compute_loss(arc_logits: torch.Tensor, 
                 rel_logits: torch.Tensor, 
                 arc_gt: torch.Tensor,
                 rel_gt: torch.Tensor, 
                 mask: torch.Tensor) -> torch.Tensor:
    flip_mask = mask.eq(0)

    def one_loss(logits, gt):
        tmp1 = logits.view(-1, logits.size(-1))
        tmp2 = gt.masked_fill(flip_mask, -1).reshape(-1)
        return cross_entropy(tmp1, tmp2, ignore_index=-1)

    arc_loss = one_loss(arc_logits, arc_gt)
    rel_loss = one_loss(rel_logits, rel_gt)

    return arc_loss + rel_loss


def compute_metrics(head_pred: torch.Tensor,
                    rel_pred: torch.Tensor,
                    head_gt: torch.Tensor,
                    rel_gt: torch.Tensor,
                    mask: torch.Tensor):
    """
    CoNLL:
    LAS(labeled attachment score): the proportion of “scoring” tokens that are assigned both the correct head and the correct dependency relation label.
    Punctuation tokens are non-scoring. In very exceptional cases, and depending on the original treebank annotation, some additional types of tokens might also be non-scoring.
    The overall score of a system is its labeled attachment score on all test sets taken together.

    UAS(Unlabeled attachment score): the proportion of “scoring” tokens that are assigned the correct head (regardless of the dependency relation label).
    """
    if len(head_pred.shape) > len(head_gt.shape):
        pred_dim, indices_dim = 2, 1
        head_pred = head_pred.max(pred_dim)[indices_dim]
    
    if len(rel_pred.shape) > len(rel_gt.shape):
        pred_dim, indices_dim = 2, 1
        rel_pred = rel_pred.max(pred_dim)[indices_dim]

    head_pred_correct = (head_pred == head_gt).long() * mask
    rel_pred_correct = (rel_pred == rel_gt).long() * head_pred_correct
    arc = head_pred_correct.sum().item()
    rel = rel_pred_correct.sum().item()
    num = mask.sum().item()

    return {'UAS': arc / num, 'LAS': rel / num}


if __name__ == "__main__":
    data_path = "data/ctb8.0/dep/"
    vocab_file = "data/ctb8.0/processed/vocab.txt"

    data_config = load_config("examples/dependency_parsing/config/data.json", mode="dict")
    model_config = load_config("examples/dependency_parsing/config/model.json", mode="dict")
    trainer_config = load_config("examples/dependency_parsing/config/trainer.json", mode="dict")

    vocab = Vocab(data_config)
    dataset = CTBDataset(vocab, data_config)

    data_iter = MyDataLoader(dataset, data_config)

    model  = DependencyParser(vocab_size=vocab.word_size, tag_size=vocab.tag_size, rel_size=vocab.rel_size, config=model_config)
    if torch.cuda.is_available():
        model.cuda()

    epochs = 10
    optim = Adam(model.parameters(), lr=0.001)

    step = 0
    for epoch in range(1, epochs+1):
        for batch in data_iter:
            words, tags, heads, rels, masks = batch
            if torch.cuda.is_available():
                words, tags, heads, rels, masks = words.cuda(), tags.cuda(), heads.cuda(), rels.cuda(), masks.cuda()

            optim.zero_grad()
            arc_pred, rel_pred = model(words, tags, heads)

            loss = compute_loss(arc_pred, rel_pred, heads, rels, masks)
            loss.backward()

            metrics = compute_metrics(arc_pred, rel_pred, heads, rels, masks)

            optim.step()

            if step % 100 == 0:
                print(f"--step {step}, loss {loss}")
                print(f"  {metrics}")
            step += 1

    print("finished.")