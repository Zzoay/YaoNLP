
import torch
import torch.nn as nn
import torch.nn.functional as F

from yaonlp.utils import to_cuda 

class Trainer():
    def __init__(self, config):
        self.config = config

    def train(self, 
              model: nn.Module, 
              optim: torch.optim.Optimizer, 
              train_iter: torch.utils.data.DataLoader) -> None:
        if self.config["cuda"] and torch.cuda.is_available():
            model.cuda()

        step = 0
        for epoch in range(1, self.config["epochs"]+1):
            for batch in train_iter:
                words, tags, heads, rels, masks, seq_lens = batch

                if self.config["cuda"] and torch.cuda.is_available():
                    words, tags, heads, rels, masks, seq_lens = to_cuda(data=(words, tags, heads, rels, masks, seq_lens))

                arc_logits, rel_logits = model(words, tags, heads, seq_lens)

                loss = compute_loss(arc_logits, rel_logits, heads, rels, masks)
                loss = loss / self.config['update_every']
                loss.backward()

                metrics = compute_metrics(arc_logits, rel_logits, heads, rels, masks)

                if step % self.config['update_every'] == 0:
                    nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, model.parameters()), max_norm=self.config['clip'])
                    optim.step()
                    model.zero_grad() 

                if step % self.config['print_every'] == 0:
                    print(f"--epoch {epoch}, step {step}, loss {loss}")
                    print(f"  {metrics}")
                step += 1
        print("--training finished.")


def compute_loss(arc_logits: torch.Tensor, 
                 rel_logits: torch.Tensor, 
                 arc_gt: torch.Tensor,  # ground truth
                 rel_gt: torch.Tensor, 
                 mask: torch.Tensor) -> torch.Tensor:
    flip_mask = mask.eq(0)  # where equals 0 is True

    def one_loss(logits, gt):
        tmp1 = logits.view(-1, logits.size(-1))
        tmp2 = gt.masked_fill(flip_mask, -1).view(-1)
        return F.cross_entropy(tmp1, tmp2, ignore_index=-1)

    arc_loss = one_loss(arc_logits, arc_gt)
    rel_loss = one_loss(rel_logits, rel_gt)

    return arc_loss + rel_loss


def compute_metrics(arc_logits: torch.Tensor,
                    rel_logits: torch.Tensor,
                    arc_gt: torch.Tensor,  # ground truth
                    rel_gt: torch.Tensor,
                    mask: torch.Tensor):
    """
    CoNLL:
    LAS(labeled attachment score): the proportion of “scoring” tokens that are assigned both the correct head and the correct dependency relation label.
    Punctuation tokens are non-scoring. In very exceptional cases, and depending on the original treebank annotation, some additional types of tokens might also be non-scoring.
    The overall score of a system is its labeled attachment score on all test sets taken together.

    UAS(Unlabeled attachment score): the proportion of “scoring” tokens that are assigned the correct head (regardless of the dependency relation label).
    """
    if len(arc_logits.shape) > len(arc_gt.shape):
        pred_dim, indices_dim = 2, 1
        arc_logits = arc_logits.max(pred_dim)[indices_dim]
    
    if len(rel_logits.shape) > len(rel_gt.shape):
        pred_dim, indices_dim = 2, 1
        rel_logits = rel_logits.max(pred_dim)[indices_dim]

    arc_logits_correct = (arc_logits == arc_gt).long() * mask
    rel_logits_correct = (rel_logits == rel_gt).long() * arc_logits_correct
    arc = arc_logits_correct.sum().item()
    rel = rel_logits_correct.sum().item()
    num = mask.sum().item()

    return {'UAS': arc / num, 'LAS': rel / num}
