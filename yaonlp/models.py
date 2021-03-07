from abc import ABCMeta, abstractmethod

from torch.nn import Module


class Model(Module):
    def __init__(self, )->None:
        super(Model, self).__init__()
    __metaclass__ = ABCMeta

    @abstractmethod
    def initial(self):
        raise NotImplementedError