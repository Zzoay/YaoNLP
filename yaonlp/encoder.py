
from abc import abstractmethod

from transformers import BertModel
from torch import Tensor
from torch.nn import Module


# pretrained models representation
class PretReps(Module):
    def __init__(self, model_path: str) -> None:
        super(PretReps, self).__init__()

        self.model_path = model_path

    # @abstractmethod
    # def forward(self, **inputs) -> Tensor:
    #     raise NotImplementedError


class BertReps(PretReps):
    def __init__(self, model_path) -> None:
        super(BertReps, self).__init__(model_path)
        self.encoder = BertModel.from_pretrained(self.model_path)
    
    def forward(self, 
                all_input_ids: Tensor, 
                all_segment_ids: Tensor, 
                all_input_mask: Tensor) -> Tensor:
        return self.encoder(all_input_ids, all_segment_ids, all_input_mask)


# TODO GloVe
class GloveReps(PretReps):
    def __init__(self, model_path) -> None:
        super(GloveReps, self).__init__(model_path)

    def forward(self) -> Tensor:
        raise NotImplementedError