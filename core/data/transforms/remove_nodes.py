import dgl
import copy
import torch
import numpy as np
import numpy as np
from dgl import DGLGraph
from dgl.transforms import BaseTransform


class RemoveNodes(BaseTransform):

    def __init__(self, node_list):
        self.node_list = node_list


    def __call__(self, graph:DGLGraph)->DGLGraph:

        src, dst = graph.edges()
        eid = graph.edges("eid").numpy()

        remove_idx_src = np.in1d(src.numpy(), self.node_list)
        remove_idx_dst = np.in1d(dst.numpy(), self.node_list)

        remove_idx = np.logical_or(remove_idx_src, remove_idx_dst)
        retain_idx = ~copy.deepcopy(remove_idx)

        unlearn_eid = eid[remove_idx]
        retain_eid = eid[retain_idx]

        out_graph = copy.deepcopy(graph)
        out_graph.remove_edges(unlearn_eid)

        node_features = out_graph.ndata["feat"]
        feat_shape = node_features.size(1)

        for n in self.node_list:
            node_features[n, :] = torch.zeros((1, feat_shape))

        unlearn_mask = torch.from_numpy(
            np.in1d(np.arange(graph.num_nodes()),
                    self.node_list))

        retain_mask = torch.logical_and(~unlearn_mask, out_graph.ndata["train_mask"])

        out_graph.ndata["feat"] = node_features
        out_graph.ndata["unlearn_mask"] = unlearn_mask
        out_graph.ndata["retain_mask"] = retain_mask

        return out_graph