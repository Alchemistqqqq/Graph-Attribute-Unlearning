import dgl
import torch
import numpy as np
from dgl import DGLGraph
from dgl.transforms import BaseTransform

class RandomNodeSplit(BaseTransform):

    def __init__(self, num_test: float,
                       num_val:  float,
                       shuffle: bool=True):

        self.train_split = 1-num_test-num_val
        self.test_split = num_test
        self.valid_split = num_val
        self.shuffle=shuffle

    def __call__(self, graph: DGLGraph)->DGLGraph:

        num_nodes = graph.num_nodes()
        num_test = int(self.test_split * num_nodes)
        num_valid = int(self.valid_split * num_nodes)

        nodes = np.arange(num_nodes)
        if self.shuffle:
            nodes = np.random.shuffle(nodes)

        valid_idx = nodes[:num_valid]
        test_idx = nodes[num_valid:num_valid+num_test]
        train_idx = nodes[num_valid+num_test:]

        train_mask = np.in1d(nodes, train_idx)
        test_mask = np.in1d(nodes, test_idx)
        valid_mask = np.in1d(nodes, valid_idx)

        graph.ndata["train_mask"] = torch.from_numpy(train_mask)
        graph.ndata["test_mask"] = torch.from_numpy(test_mask)
        graph.ndata["valid_mask"] = torch.from_numpy(valid_mask)

        return graph

        
        