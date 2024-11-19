import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi

        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class TimeDependentFeatureEncoder(nn.Module):

    def __init__(
            self,
            time_embed_dim,
            in_dim,
            layer_dims,
            dropout=None,
            dropout_prob=0.0,
            out_dim=128):
        ''' Computes score of energy function in respect to x. Output is vector in lie algebra
        '''
        super(TimeDependentFeatureEncoder, self).__init__()

        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=time_embed_dim),
                                   nn.Linear(time_embed_dim, time_embed_dim))
        self.act = lambda x: x * torch.sigmoid(x)

        self.num_layers = len(layer_dims)

        for i in range(0, self.num_layers):
            layer_in_dim = layer_dims[i-1] if i != 0 else (in_dim + time_embed_dim)
            layer_out_dim = layer_dims[i]
            setattr(
                self,
                "lin" + str(i),
                nn.utils.weight_norm(nn.Linear(layer_in_dim, layer_out_dim)),
            )

        setattr(
            self,
            "lin_final",
            nn.utils.weight_norm(nn.Linear(layer_dims[-1], out_dim)),
        )

        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.dropout = dropout

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x, time_t):
        # Obtain the Gaussian random feature embedding for t
        time_emb = self.act(self.embed(time_t))

        x = torch.cat([x, time_emb], dim=-1)

        for layer in range(self.num_layers):
            lin = getattr(self, "lin" + str(layer))
            x = self.relu(lin(x))

            if self.dropout is not None and layer in self.dropout:
                x = F.dropout(x, p=self.dropout_prob, training=self.training)

        # final output layer
        lin = getattr(self, "lin_final")
        x = self.relu(lin(x))

        return x


if __name__ == '__main__':
    model = TimeDependentFeatureEncoder(
        time_embed_dim=100,
        in_dim=500,
        layer_dims=[512, 512, 512, 512],
        dropout=[0, 1, 2, 3],
        dropout_prob=0.2,
        out_dim=128
    )

    latent = torch.randn(12, 500)
    time = torch.randn(12)
    model(latent, time)
    print('END')
