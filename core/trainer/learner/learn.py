import dgl
import torch
from core.trainer.base import BaseTrainer
from core.trainer.utils import get_optim, out_dict

#Typing
from typing import List, Dict, Tuple
from torch.nn import Module

class NormalTrainer(BaseTrainer):
    def __init__(self,
                 args,
                 model: Module,
                 dataloaders: Dict
                 ):
        self.model = model

        for label, loader in dataloaders.items():
            if label == "train":
                self.train_loaders = (dataloaders[label],)
            if label == "test":
                self.test_loaders = (dataloaders[label],)
            if label == "valid":
                self.valid_loaders = (dataloaders[label],)
        
        if not hasattr(self, "test_loaders"):
            self.test_loaders = tuple()

        self.device = args.device
        self.model.to(self.device)
        self.optim = get_optim(args)(self.model.parameters())
        self.Loss = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, args.epochs)
        
    def fit(self, epoch:int)->Dict:

        _out= out_dict()
        self.model.train()

        for idx, (in_nodes, out_nodes, blocks) in enumerate(self.train_loaders[0]):
            
            self.optim.zero_grad()
            blocks = [b.to(self.device) for b in blocks]
            data = blocks[0].srcdata["feat"].to(self.device)
            labels = blocks[-1].dstdata["label"].to(self.device)
            _pred, feat = self.model(blocks, data)
            _loss = self.Loss(_pred, labels)
            _loss.backward()
            self.optim.step()
            
            _, pred_labels = torch.max(_pred, 1)
            acc = torch.eq(pred_labels, labels).sum().item() / len(out_nodes)

            # Logging
            _out["train_loss"].append(_loss.item())
            _out["train_acc"].append(acc)

        self.scheduler.step()
        
        for k in _out.keys():
            _out[k] = sum(_out[k])/len(_out[k])

        return _out
