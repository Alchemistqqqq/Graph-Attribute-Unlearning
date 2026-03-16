import torch.nn as nn
import torch
import dgl.nn as dglnn
import torch.nn.functional as F


class MLP(nn.Module):
    """Construct two-layer MLP-type aggregator for GIN model"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, output_dim, bias=False))

    def forward(self, x):
        h = x
        h = F.relu(self.linears[0](h))
        return h


class GraphConv(nn.Module):
    def __init__(self, in_dim, hidden_dim, batchnorm=False, dropout=False, activation="relu"):
        super(GraphConv, self).__init__()

        # Replace the original GraphConv with GINConv and use MLP as the aggregation function
        mlp = MLP(input_dim=in_dim, output_dim=hidden_dim)
        self.gconv = dglnn.GINConv(mlp)

        layers = list()

        if batchnorm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        if dropout:
            layers.append(nn.Dropout())
        if activation == "relu":
            layers.append(nn.ReLU())
        if activation == "elu":
            layers.append(nn.ELU())
    
        self.layers = nn.Sequential(*layers)
    
    def forward(self, input):
        block, x = input
        x = self.gconv(block, x)  # Apply GINConv
        out = self.layers(x)  # Apply batchnorm, dropout, and activation
        return out


class UnfeatGraphIsomNet(nn.Module):
    def __init__(self, args, in_dim, hidden_dim, num_classes):
        super(UnfeatGraphIsomNet, self).__init__()

        self.dims = [(in_dim, hidden_dim)]
        for _ in range(args.depth-1):
            self.dims.append((hidden_dim, hidden_dim))

        _dropout = [args.dropout for _ in range(len(self.dims)-1)]
        _dropout.append(False)
        layers = list()
        for dim, _d in zip(self.dims, _dropout):
            layers.append(
                GraphConv(dim[0], dim[1], batchnorm=args.batchnorm, dropout=_d, activation="relu")
            )
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, blocks, x, unlearn_nodes_list=None):
        if unlearn_nodes_list is not None:
            unlearn_nodes_set = set(unlearn_nodes_list)  

        for layer, block in zip(self.layers, blocks):
            if unlearn_nodes_list is not None:
                src_nodes = block.srcdata['_ID']  
                mask = torch.isin(src_nodes, torch.tensor(list(unlearn_nodes_set), device=x.device))
                x = x.clone()
                x[mask] = 0  
            _input = (block, x)
            x = layer(_input)        
        
        feat = F.normalize(x)

        out = self.output(feat)

        return out, feat
