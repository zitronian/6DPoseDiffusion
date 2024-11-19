# adapted from https://github.com/FlyingGiraffe/vnn-neural-implicits/blob/master/im2mesh/encoder/vnn2.py

import torch
import torch.nn as nn
#from .vn_layers import *
#from .vn_utils import get_graph_feature_cross, get_graph_feature
from pose_estimation.models.nets.layers_equi import *
#from .layers_equi import *
from pose_estimation.utils.utils import transform_points_batch, transform_points

def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)

class SingleChannelVNNResnetWrapper(nn.Module):

    def __init__(self, z_dim=192, k=20, device='cpu'):
        super().__init__()
        if z_dim % 3 != 0:
            print('This might break. The module expects a feature to be a multiple of 3.')

        self.vnn_resnet = VNN_ResnetPointnet(c_dim=int(z_dim / 3), k=k, device=device)

    def forward(self, pc):
        xyz_feats = self.vnn_resnet(pc)
        return xyz_feats.reshape(pc.shape[0], -1)


class VNN_ResnetPointnet(nn.Module):
    ''' DGCNN-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, pooling='mean', meta_output=None, device='cpu'):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output
        self.device = device

        self.conv_pos = VNLinearLeakyReLU(3, 64, negative_slope=0.0, use_batchnorm=False)
        self.fc_pos = VNLinear(64, 2* hidden_dim)
        self.block_0 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, c_dim)

        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)

        self.pool_pos = mean_pool if pooling=='mean' else VNMaxPool(64)
        self.pool_0 = mean_pool if pooling=='mean' else VNMaxPool(hidden_dim)
        self.pool_1 = mean_pool if pooling=='mean' else VNMaxPool(hidden_dim)
        self.pool_2 = mean_pool if pooling=='mean' else VNMaxPool(hidden_dim)
        self.pool_3 = mean_pool if pooling=='mean' else VNMaxPool(hidden_dim)
        self.pool_4 = mean_pool if pooling=='mean' else VNMaxPool(hidden_dim)

        if meta_output == 'invariant_latent':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=False, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNStdFeature(c_dim, dim=3, normalize_frame=False, use_batchnorm=False)
            self.vn_inv = VNLinear(c_dim, 3)

    def forward(self, p):
        '''

        :param p: points of shape [B, N_samples, dims(=3?)]
        '''
        batch_size = p.size(0)
        p = p.unsqueeze(1).transpose(2, 3)
        feat = get_graph_feature_cross(p, k=self.k, device=self.device)
        net = self.conv_pos(feat)
        net = self.pool_pos(net)

        net = self.fc_pos(net)

        net = self.block_0(net)
        pooled = self.pool_0(net).unsqueeze(-1).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_1(net)
        pooled = self.pool_1(net).unsqueeze(-1).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_2(net)
        pooled = self.pool_2(net).unsqueeze(-1).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_3(net)
        pooled = self.pool_3(net).unsqueeze(-1).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.block_4(net)

        # Recude to  B x F
        net = self.pool_4(net)

        c = self.fc_c(self.actvn_c(net))

        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std

        return c


if __name__ == '__main__':
    from pose_estimation.utils.utils import SO3_R3

    model = VNN_ResnetPointnet(c_dim=8, k=5)
    device = 'cpu'
    pcl1 = torch.randn(24, 3) * 10

    # T = torch.tensor([[0, -1, 0, 10], [1, 0, 0, 10], [0, 0, 1, 10], [0, 0, 0, 1]], dtype=pcl1.dtype)
    # pcl2 = transform_points_batch(pcl1, T)
    H0 = SO3_R3().sample_marginal_std(batch=1)
    T = SO3_R3(R=H0[:, :3, :3], t=H0[:, :3, -1]).to_matrix().squeeze()

    # T = torch.tensor([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 10], [0, 0, 0, 1]], dtype=pcl1.dtype)
    pcl2 = transform_points(pcl1, T)
    # pcl2 = pcl1+10

    batch = torch.stack([pcl1, pcl2]).float().to(device)
    # batch2 = torch.stack([pcl1, pcl1]).float().to(device)
    # T = torch.tensor([
    #    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    #    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    # ], dtype=pcl1.dtype)
    # batch2_t = transform_points_batch(batch2, T)

    c = model(batch)
    print(c.shape)