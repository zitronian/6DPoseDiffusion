import torch.nn as nn


class DummyNet(nn.Module):

    def __init__(self):
        ''' Dummy net that returns its input without any learnable parameters
        '''
        super(DummyNet, self).__init__()

    def forward(self, x):

        return x