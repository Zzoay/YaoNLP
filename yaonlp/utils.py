
from numpy import random as np_random
import torch
import random


def set_seed(seed):
    random.seed(seed)
    np_random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def to_cuda(data):
    if isinstance(data, tuple):
        return [d.cuda() for d in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    raise RuntimeError


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask