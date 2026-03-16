import dgl
import copy
import torch
from tqdm import tqdm
import torch.nn.functional as F
from core.trainer.base import BaseTrainer
from core.trainer.utils import get_optim, to_device, out_dict, clone_blocks
import logging
# Typing
from typing import List, Dict, Tuple
from torch.nn import Module

class RetrainTrainer(BaseTrainer):
    def __init__(self,
                 args,
                 model: Module,
                 dataloaders: Dict,
                 **kwargs):
        super().__init__()

        self.model = model
        self.loader_rt = dataloaders["retain_train"]
        self.loader_ul = dataloaders["unlearn"]
        self.test_loaders = (dataloaders["retain_test"], dataloaders["unlearn_test"])
        self.device = args.device
        self.model.to(self.device)
        self.optim = get_optim(args)(self.model.parameters())
        self.Loss = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, args.epochs)

    def fit(self, epoch: int) -> Dict:

        # 这里你原来的 out_dict() 我保留，但为了稳妥起见，
        # 再补上几个 key，分别记录 retain / unlearn 的指标
        _out = out_dict()
        for k in [
            "retain_train_loss", "retain_train_acc",
            "unlearn_train_loss", "unlearn_train_acc"
        ]:
            if k not in _out:
                _out[k] = []

        self.model.train()

        # ====== 1. 先训练 retain_train ======
        for idx, (in_nodes, out_nodes, blocks) in enumerate(self.loader_rt):
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

            _out["retain_train_loss"].append(_loss.item())
            _out["retain_train_acc"].append(acc)

        # ====== 2. 再训练 unlearn ======
        for idx, (in_nodes, out_nodes, blocks) in enumerate(self.loader_ul):
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

            _out["unlearn_train_loss"].append(_loss.item())
            _out["unlearn_train_acc"].append(acc)

        # ====== 3. 更新学习率 ======
        self.scheduler.step()

        # ====== 4. 把 list 变成 epoch 平均值 ======
        for k in _out.keys():
            if isinstance(_out[k], list) and len(_out[k]) > 0:
                _out[k] = sum(_out[k]) / len(_out[k])

        return _out
