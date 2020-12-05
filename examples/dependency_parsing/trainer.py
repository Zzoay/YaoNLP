
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from yaonlp.utils import to_cuda 
from yaonlp.optim import OptimChooser
from yaonlp.trainer import Trainer

class MyTrainer(Trainer):
    def __init__(self, 
                 loss_fn: Callable, 
                 metrics_fn: Callable, 
                 config: dict) -> None:
        super(MyTrainer, self).__init__(loss_fn, metrics_fn, config)

        self.optim_type = config["optimizer"]["type"]
        self.optim_params = config["optimizer"]["parameters"]

    def train(self, 
              model: nn.Module, 
              train_iter: torch.utils.data.DataLoader) -> None:
        if self.config["cuda"] and torch.cuda.is_available():
            model.cuda()

        optim_chooser = OptimChooser(model.parameters(), 
                                     optim_type=self.optim_type, 
                                     optim_params=self.optim_params)
        optim = optim_chooser.optim()

        step = 0
        for epoch in range(1, self.config["epochs"]+1):
            for batch in train_iter:
                words, tags, heads, rels, masks, seq_lens = batch

                if self.config["cuda"] and torch.cuda.is_available():
                    words, tags, heads, rels, masks, seq_lens = to_cuda(data=(words, tags, heads, rels, masks, seq_lens))

                arc_logits, rel_logits = model(words, tags, heads, seq_lens)

                loss = self.loss_fn(arc_logits, rel_logits, heads, rels, masks)
                loss = loss / self.config['update_every']
                loss.backward()

                metrics = self.metrics_fn(arc_logits, rel_logits, heads, rels, masks)

                if step % self.config['update_every'] == 0:
                    nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, model.parameters()), max_norm=self.config['clip'])
                    optim.step()
                    model.zero_grad() 

                if step % self.config['print_every'] == 0:
                    print(f"--epoch {epoch}, step {step}, loss {loss}")
                    print(f"  {metrics}")
                step += 1
        print("--training finished.")
