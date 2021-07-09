
import random

import numpy as np
import torch


def set_seed(seed):
    """
    set random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def to_cuda(data):
    """
    put input data into cuda device
    """
    if isinstance(data, tuple):
        return [d.cuda() for d in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    raise RuntimeError


def seq_mask_by_lens(lengths:torch.Tensor, 
                     maxlen=None, 
                     dtype=torch.bool):
    """
    giving sequence lengths, return mask of the sequence.
    example:
        input: 
        lengths = torch.tensor([4, 5, 1, 3])

        output:
        tensor([[ True,  True,  True,  True, False],
                [ True,  True,  True,  True,  True],
                [ True, False, False, False, False],
                [ True,  True,  True, False, False]])
    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(start=0, end=maxlen.item(), step=1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


def trace_back_index(idcs):
    """
    giving some idcs which will be used to change data order, return a trace index to back to the original order.
    example: 
        a = [1, 2, 3, 4]
        idcs = [3, 2, 0, 1]
        b = a[idcs] = [4, 3, 1, 2]

        trace_idcs = trace_back_index(idcs) = [2, 3, 1, 0]
        b[trace_idcs] = [1, 2, 3, 4] = a
    """
    trace_idcs = np.zeros(len(idcs), dtype=int)
    trace_idcs[idcs] = np.arange(len(idcs))
    return list(trace_idcs)
