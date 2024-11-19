# adapted from https://github.com/FlyingGiraffe/vnn-neural-implicits/blob/master/im2mesh/encoder/vnn2.py

import torch
import torch.nn as nn
from pose_estimation.models.nets.vnt.vnt_layers import *
from pose_estimation.models.nets.vnt.utils.vn_dgcnn_util import get_graph_feature_cross
from pose_estimation.utils.utils import transform_points_batch, transform_points

def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)

class SingleChannelVNTResnetWrapper(nn.Module):

    def __init__(self, z_dim=192, k=20, device='cpu'):
        super().__init__()
        if z_dim % 3 != 0:
            print('This might break. The module expects a feature to be a multiple of 3.')

        self.vnt_resnet = VNT_ResnetPointnet(c_dim=int(z_dim / 3), k=k, device=device)

    def forward(self, pc):
        xyz_feats = self.vnt_resnet(pc)
        return xyz_feats.reshape(pc.shape[0], -1)


class VNT_ResnetPointnet(nn.Module):
    ''' DGCNN-based VNN encoder network with ResNet blocks.

    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, k=20, pooling='max', meta_output=None, device='cpu'):
        super().__init__()
        self.c_dim = c_dim
        self.k = k
        self.meta_output = meta_output
        self.device = device

        self.conv_pos = VNTLinearLeakyReLU(3, 64, negative_slope=0.0, use_batchnorm=False)
        self.fc_pos = VNTLinear(64, 2* hidden_dim)
        self.block_0 = VNTResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = VNTResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = VNTResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = VNTResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = VNTResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = VNTLinear(hidden_dim, c_dim)

        self.actvn_c = VNTLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)

        self.pool_pos = mean_pool if pooling=='mean' else VNTMaxPool(64)
        self.pool_0 = mean_pool if pooling=='mean' else VNTMaxPool(hidden_dim)
        self.pool_1 = mean_pool if pooling=='mean' else VNTMaxPool(hidden_dim)
        self.pool_2 = mean_pool if pooling=='mean' else VNTMaxPool(hidden_dim)
        self.pool_3 = mean_pool if pooling=='mean' else VNTMaxPool(hidden_dim)
        self.pool_4 = mean_pool if pooling=='mean' else VNTMaxPool(hidden_dim)

        if meta_output == 'invariant_latent':
            self.std_feature = VNTStdFeature(c_dim, dim=3, normalize_frame=False, use_batchnorm=False)
        elif meta_output == 'invariant_latent_linear':
            self.std_feature = VNTStdFeature(c_dim, dim=3, normalize_frame=False, use_batchnorm=False)
            self.vn_inv = VNTLinear(c_dim, 3)

    def forward(self, p, equ_rot=None):
        '''

        :param p: points of shape [B, N_samples, dims(=3?)]
        :param equ_rot: 4x4 transformation matrix. Can be used to debug equivariance property. Provide batch with two
        pcl. 2nd pcl is same as 1st pcl but transformed in the input space. During forward pass it is checked if
        latent for 2nd pcl eauals the transformed latent of the 1st pcl.
        '''

        if equ_rot is not None:
            assert p.shape[0] == 2

        p = p.unsqueeze(1).transpose(2, 3)
        feat = get_graph_feature_cross(p, k=self.k, use_global=False, device=self.device)
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

        if equ_rot is not None:
            equiv = torch.all(torch.isclose(transform_points(c[0], T), c[1]))
            print('Equiv. Latent: ', equiv.item())

        if self.meta_output == 'invariant_latent':
            c_std, z0 = self.std_feature(c)
            return c, c_std
        elif self.meta_output == 'invariant_latent_linear':
            c_std, z0 = self.std_feature(c)
            c_std = self.vn_inv(c_std)
            return c, c_std

        return c


if __name__ == '__main__':

    pass