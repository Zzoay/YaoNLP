
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
