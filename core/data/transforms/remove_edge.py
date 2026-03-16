import dgl
import copy
import torch
import numpy as np
from dgl import DGLGraph
from dgl.transforms import BaseTransform


class RemoveNodeEdges(BaseTransform):

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

        return out_graph
