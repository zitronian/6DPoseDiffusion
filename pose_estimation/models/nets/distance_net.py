import torch.nn as nn


class DistanceNet(nn.Module):

    def __init__(self, in_dim=256, hidden_dim=256, out_dim=128):
        ''' Computes score of energy function in respect to x. Output is vector in lie algebra
        '''
        super(DistanceNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.net(x)
        
        return x