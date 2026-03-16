import dgl
import torch
import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F

class GraphConv(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(GraphConv, self).__init__()

        self.gconv = dglnn.SGConv(in_dim, hidden_dim, bias=False)
    
    def forward(self, input):
        block, x = input
        src, dst = block.edges()
        dst_nodes = torch.unique(dst)
        _graph = dgl.graph(('coo', (src, dst)), num_nodes=x.shape[0], device=block.device)
        _graph.ndata['x'] = block.srcdata["feat"]
        _graph = _graph.add_self_loop()

        x  = self.gconv(_graph, x)
        x = x[dst_nodes, :]
        return x 
    
class SimpleGraphConv(nn.Module):

    def __init__(self, args, in_dim, hidden_dim, num_classes):
        super(SimpleGraphConv, self).__init__()

        self.dims = [(in_dim, hidden_dim)]
        for _ in range(args.depth-1):
            self.dims.append((hidden_dim, hidden_dim))

        layers = list()
        for dim in self.dims:
            layers.append(
                GraphConv(dim[0], dim[1])
            )
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, blocks, x):
        for layer, block in zip(self.layers, blocks):
            src, dst = block.edges()
            dst_unique = torch.unique(dst)

            _input = (block, x)
            x = layer(_input)
        
        feat = F.normalize(x)

        out = self.output(feat)
        
        return out, feat
