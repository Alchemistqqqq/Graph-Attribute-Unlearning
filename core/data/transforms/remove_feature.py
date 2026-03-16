import dgl
import torch
import numpy as np 
from dgl.transforms import BaseTransform

class RemoveNodeFeatures(BaseTransform):
    
    def __init__(self, node_list: np.ndarray):
        super(RemoveNodeFeatures).__init__()
        self.node_list=node_list

    def __call__(self, graph): 

        node_features = graph.ndata["feat"]
        feat_shape = node_features.size(1)
        for n in self.node_list:
            node_features[n, :] = torch.zeros((1, feat_shape))

        graph.ndata["feat"] = node_features
        graph.unlearn_idx = torch.Tensor(self.node_list)
        return graph