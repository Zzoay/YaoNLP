
from typing import Dict

import torch
import torch.nn.functional as F


def arc_rel_loss(arc_logits: torch.Tensor, 
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


def uas_las(arc_logits: torch.Tensor,
            rel_logits: torch.Tensor,
            arc_gt: torch.Tensor,  # ground truth
            rel_gt: torch.Tensor,
            mask: torch.Tensor) -> Dict[str:float, str:float]:
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
