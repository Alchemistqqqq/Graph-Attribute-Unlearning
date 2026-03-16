from abc import ABC, abstractmethod, abstractproperty
import torch
import torch.nn.functional as F

class BaseTrainer(ABC):

    @abstractmethod
    def fit(self):
        pass
    
    @torch.no_grad()
    def evaluate(self):
        
        accs = list()
        confs = list()
        for test_loader in self.test_loaders:
            _acc, _conf = self._evaluate(test_loader)
            accs.append(_acc)
            confs.append(_conf)

        return accs, confs

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_conf = 0
        for idx, (in_nodes, out_nodes, blocks) in enumerate(loader):
            
            blocks = [b.to(self.device) for b in blocks]
            data = blocks[0].srcdata["feat"]
            label = blocks[-1].dstdata["label"]
            _pred, feat = self.model(blocks, data)

            total_samples += len(out_nodes)
            _, pred = torch.max(_pred, 1)
            total_correct += (torch.eq(pred, label)).sum().item()
            conf, _ = torch.max(F.softmax(_pred, 1), 1)
            total_conf += conf.sum().item()

        print(total_correct, total_samples) 

        return total_correct/total_samples, total_conf/total_samples