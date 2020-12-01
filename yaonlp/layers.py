
from typing import Callable

from torch import nn
from torch.nn import functional as F

 
class NonLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation: Callable):  # TODO activation func
        super(NonLinear, self).__init__()
        self._linear = nn.Linear(in_features, out_features)
        self._activation = activation
    
    def forward(self, x):
        return self._activation(self._linear(x))   # TODO leaky_relu now