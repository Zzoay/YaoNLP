
from typing import Callable, Optional

import torch
import torch.nn as nn

from yaonlp.utils import to_cuda 
from yaonlp.optim import OptimChooser
from yaonlp.trainer import Trainer
from yaonlp.data import MyDataLoader


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
              train_iter: MyDataLoader, 
              val_iter: Optional[MyDataLoader] = None) -> None:
        model.train()
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
                    words, tags, heads, rels, masks = to_cuda(data=(words, tags, heads, rels, masks))

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

                if val_iter and step % self.config['eval_every'] == 0:
                    self.eval(model, val_iter)
                    # back to train mode
                    model.train()

                step += 1
        print("--training finished.")

    # eval func
    def eval(self, model: nn.Module, eval_iter: MyDataLoader) -> None:
        model.eval()

        avg_loss, avg_uas, avg_las, step = 0.0, 0.0, 0.0, 0
        for step, batch in enumerate(eval_iter):
            words, tags, heads, rels, masks, seq_lens = batch

            if self.config["cuda"] and torch.cuda.is_available():
                words, tags, heads, rels, masks = to_cuda(data=(words, tags, heads, rels, masks))

            with torch.no_grad():
                arc_logits, rel_logits = model(words, tags, heads, seq_lens, eval=True)

            loss = self.loss_fn(arc_logits, rel_logits, heads, rels, masks)
            avg_loss += loss

            metrics = self.metrics_fn(arc_logits, rel_logits, heads, rels, masks)

            avg_uas += metrics['UAS']
            avg_las += metrics['LAS']

        # size = eval_iter.data_size 
        avg_loss /= step
        avg_uas /= step
        avg_las /= step 
        print("--Evaluation:")
        print("-loss: {}  UAS: {}  LAS: {} \n".format(avg_loss, avg_uas, avg_las))