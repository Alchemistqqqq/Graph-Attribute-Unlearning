import torch.nn as nn
import dgl.nn as dglnn
import torch.nn.functional as F

class GraphAttn(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, batchnorm=True, activation="relu"):
        super(GraphAttn, self).__init__()

        self.gconv = dglnn.GATConv(in_dim, out_dim, num_heads)
        layers = list()
        layers.append(nn.Flatten(1,2))
        if batchnorm:
            layers.append(nn.BatchNorm1d(out_dim*num_heads))
        if activation == "relu":
            layers.append(nn.ReLU())
        if activation == "elu":
            layers.append(nn.ELU())

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        block, x = input
        x = self.gconv(block, x)
        out = self.layers(x)
        return out



class GraphAttNet(nn.Module):

    def __init__(self, args, in_dim, hidden_dim, num_classes):
        super(GraphAttNet, self).__init__()

        dims = [[in_dim, hidden_dim]]
        heads = list()
        for _ in range(args.depth-1):
            dims.append([hidden_dim, hidden_dim])
            heads.append(args.head)
        heads.append(1)

        for i in reversed(range(1, len(dims))):
            dims[i][0] *= heads[i-1]

        layers = list()
        for dim, head in zip(dims, heads):
            layers.append(GraphAttn(dim[0], dim[1], head))
        self.output = nn.Linear(hidden_dim, num_classes)
        self.layers = nn.Sequential(*layers)
        
    def forward(self, blocks, x):
        for layer, block in zip(self.layers, blocks):
            _input = (block, x)
            x = layer(_input)
        feat = F.normalize(x)
        out = self.output(feat)
        
        return out, feat

        