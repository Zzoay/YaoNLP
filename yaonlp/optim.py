
from torch.optim import *

class OptimChooser():
    def __init__(self, params, optim_type, optim_params) -> None:
        self.optim_type = optim_type
        self.optim_params = optim_params
        self._optim = globals()[optim_type](params, **optim_params)

    def optim(self):
        return self._optim
