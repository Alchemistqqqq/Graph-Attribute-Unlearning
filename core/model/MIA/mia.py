import torch
import torch.nn as nn

class FCNet(nn.Module):
    def __init__(self, in_features, layer_dims):
        super(FCNet, self).__init__()

        self.module = nn.Sequential()
        for i, layer_dim in enumerate(layer_dims):
            self.module.add_module(
                "fc_" + str(i+1), 
                nn.Linear(in_features=in_features, out_features=layer_dim, bias=False)
            )

            self.module.add_module(
                "batchnorm_" + str(i+1), 
                nn.BatchNorm1d(num_features=layer_dim)
            )
            
            self.module.add_module(
                "relu_" + str(i+1), 
                nn.ReLU()
            )
            
            in_features = layer_dim

        self.fc_out = nn.Linear(in_features=in_features, out_features=1, bias=True)
        self.out_activation = nn.Sigmoid()

    def forward(self, x):
        x = self.module(x)
        x = self.fc_out(x)
        x = self.out_activation(x)

        return x