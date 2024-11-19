import torch.nn as nn


class ScoreNet(nn.Module):
    """
    Decoder Network
    """

    def __init__(self, in_dim, hidden_dim=256, out_dim=6):
        ''' Computes score of energy function in respect to x. Output is vector in lie algebra
        '''
        super(ScoreNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.relu = nn.ReLU()

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        x = self.fc3(x)
        
        return x