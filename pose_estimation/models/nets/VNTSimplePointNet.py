# adapted from https://github.com/FlyingGiraffe/vnn-neural-implicits/blob/master/im2mesh/encoder/vnn2.py

import torch
import torch.nn as nn
from pose_estimation.models.nets.vnt.vnt_layers import *
from pose_estimation.models.nets.vnt.utils.vn_dgcnn_util import get_graph_feature_cross
from pose_estimation.models.nets.layers_equi import VNLinear, VNLinearLeakyReLU, VNStdFeature, VNMaxPool, VNResnetBlockFC, VNLeakyReLU
#from pose_estimation.models.nets.layers_equi import get_graph_feature_cross, get_graph_feature
from pose_estimation.utils.utils import transform_points_batch, transform_points

def mean_pool(x, dim=-1, keepdim=False):
    return x.mean(dim=dim, keepdim=keepdim)

class SingleChannelVNTSimpleWrapper(nn.Module):

    def __init__(self, z_dim=192, k=20, device='cpu'):
        super().__init__()
        if z_dim % 3 != 0:
            print('This might break. The module expects a feature to be a multiple of 3.')

        self.vnt_resnet = VNT_SimplePointnet(c_dim=int(z_dim / 3), k=k, device=device)

    def forward(self, pc):
        xyz_feats = self.vnt_resnet(pc)
        return xyz_feats.reshape(pc.shape[0], -1)

class SingleChannelVNTDistangledWrapper(nn.Module):

    def __init__(self, z_dim=192, k=20, device='cpu'):
        super().__init__()
        if z_dim % 3 != 0:
            print('This might break. The module expects a feature to be a multiple of 3.')

        self.vnt_resnet = VNTDistResnetEncoder(c_dim=int(z_dim / 3), k=k, device=device)
        #self.vnt_resnet = VNTSimpleEncoder(c_dim=int(z_dim / 3), k=k, device=device)

    def forward(self, pc):
        xyz_feats = self.vnt_resnet(pc)
        return xyz_feats.reshape(pc.shape[0], -1)

class VNT_SimplePointnet(nn.Module):
    ''' DGCNN-based VNN encoder network.

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
        self.device=device
        which_norm_VNT='norm'


        self.conv_pos = VNTLinearLeakyReLU(3, 64, negative_slope=0.0, use_batchnorm=False, which_norm_VNT=which_norm_VNT)
        self.fc_pos = VNTLinear(64, 2 * hidden_dim,which_norm_VNT=which_norm_VNT)
        self.fc_0 = VNTLinear(2 * hidden_dim, hidden_dim,which_norm_VNT=which_norm_VNT)
        self.fc_1 = VNTLinear(2 * hidden_dim, hidden_dim,which_norm_VNT=which_norm_VNT)
        self.fc_2 = VNTLinear(2 * hidden_dim, hidden_dim,which_norm_VNT=which_norm_VNT)
        self.fc_3 = VNTLinear(2 * hidden_dim, hidden_dim,which_norm_VNT=which_norm_VNT)
        self.fc_c = VNTLinear(hidden_dim, c_dim,which_norm_VNT=which_norm_VNT)

        self.actvn_0 = VNTLeakyReLU(2 * hidden_dim, negative_slope=0.0)
        self.actvn_1 = VNTLeakyReLU(2 * hidden_dim, negative_slope=0.0)
        self.actvn_2 = VNTLeakyReLU(2 * hidden_dim, negative_slope=0.0)
        self.actvn_3 = VNTLeakyReLU(2 * hidden_dim, negative_slope=0.0)
        self.actvn_c = VNTLeakyReLU(hidden_dim, negative_slope=0.0)

        self.pool = mean_pool

        self.conv1 = VNTLinearLeakyReLU(64, hidden_dim, negative_slope=0.0, use_batchnorm=False, which_norm_VNT='norm')
        self.conv2 = VNTLinearLeakyReLU(2 * hidden_dim, hidden_dim, negative_slope=0.0, use_batchnorm=False, which_norm_VNT='norm')
        self.conv3 = VNTLinearLeakyReLU(2 * hidden_dim, hidden_dim, negative_slope=0.0, use_batchnorm=False,
                                        which_norm_VNT='norm')

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
        #feat = get_graph_feature_cross(p, k=self.k, use_global=False, device=self.device)
        feat = get_graph_feature_cross(p, k=self.k, device=self.device)
        net = self.conv_pos(feat)
        net = self.pool(net, dim=-1)

        net = self.conv1(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.conv2(net)
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.conv3(net)
        net = self.pool(net, dim=-1)
        print('Here')
        '''
        net = self.fc_pos(net)

        net = self.fc_0(self.actvn_0(net))
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.fc_1(self.actvn_1(net))
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.fc_2(self.actvn_2(net))
        pooled = self.pool(net, dim=-1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=1)

        net = self.fc_3(self.actvn_3(net))

        net = self.pool(net, dim=-1)

        c = self.fc_c(self.actvn_c(net))
        '''
        if equ_rot is not None:
            equiv = torch.all(torch.isclose(transform_points(c[0], T), c[1]))
            print('Equiv. Latent: ', equiv.item())

        return c


class STNkd(nn.Module):
    def __init__(self, d=64, pooling='mean', use_batchnorm=True):
        super(STNkd, self).__init__()

        self.use_batchnorm = use_batchnorm
        self.conv1 = VNLinearLeakyReLU(d, 64 // 3, dim=4, negative_slope=0.0, use_batchnorm=self.use_batchnorm)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 128 // 3, dim=4, negative_slope=0.0, use_batchnorm=self.use_batchnorm)
        self.conv3 = VNLinearLeakyReLU(128 // 3, 1024 // 3, dim=4, negative_slope=0.0, use_batchnorm=self.use_batchnorm)

        self.fc1 = VNLinearLeakyReLU(1024 // 3, 512 // 3, dim=3, negative_slope=0.0, use_batchnorm=self.use_batchnorm)
        self.fc2 = VNLinearLeakyReLU(512 // 3, 256 // 3, dim=3, negative_slope=0.0, use_batchnorm=self.use_batchnorm)

        if pooling == 'max':
            self.pool = VNMaxPool(1024 // 3)
        elif pooling == 'mean':
            self.pool = mean_pool

        self.fc3 = VNLinear(256 // 3, d)
        self.d = d

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class VNTSimpleEncoder(nn.Module):
    def __init__(self, c_dim=128, dim=3, hidden_dim=64, trans_c_dim=12, k=20, pooling='mean', global_feat=True, feature_transform=False, device='cpu'):
        super(VNTSimpleEncoder, self).__init__()
        self.n_knn = k
        self.device=device
        which_norm_VNT = 'norm'


        base_ch = hidden_dim * 3
        output_ch_trans = trans_c_dim
        output_ch_rot = c_dim - output_ch_trans
        output_ch = c_dim

        self.conv_pos = VNTLinearLeakyReLU(3, base_ch // 3, dim=5, negative_slope=0.0,use_batchnorm=False,
                                                      which_norm_VNT=which_norm_VNT)
        self.conv_center_ = VNTLinearLeakyReLU(base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0,use_batchnorm=False,
                                                          which_norm_VNT=which_norm_VNT)
        self.conv_center = VNTLinearLeakyReLU(base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0,use_batchnorm=False,
                                                         which_norm_VNT=which_norm_VNT)
        self.conv_trans = VNTLinearLeakyReLU(base_ch // 3, output_ch_trans, dim=4, negative_slope=0.0,use_batchnorm=False,
                                              which_norm_VNT=which_norm_VNT)

        self.conv1 = VNLinearLeakyReLU(base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0, use_batchnorm=False,)
        #self.conv2 = VNLinearLeakyReLU(base_ch // 3 * 2, (2 * base_ch) // 3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(base_ch // 3, base_ch// 3, dim=4, negative_slope=0.0, use_batchnorm=False,)

        self.conv3 = VNLinear(base_ch // 3, output_ch_rot)


        self.pool = mean_pool

        self.feature_transform = feature_transform

        if self.feature_transform:
            self.fstn = STNkd(d=base_ch // 3, use_batchnorm=False)

    def forward(self, x, T=None):
        B, D, N = x.size()

        x = x.unsqueeze(1).transpose(2,3)
        feat = get_graph_feature_cross(x, k=self.n_knn, use_global=False, device=self.device)

        x = self.conv_pos(feat)
        x = self.pool(x)
        x2 = self.conv_center_(x)
        x_center = self.conv_center(x2)
        trans_x = self.conv_trans(x_center)
        trans_latent = mean_pool(trans_x)
        #center_loc = torch.mean(x_center, dim=-1, keepdim=True)

        if T is not None:
            #equiv = torch.all(torch.isclose(transform_points(center_loc.squeeze(dim=-1)[0], T), center_loc.squeeze(dim=-1)[1], atol=1e-4))
            equiv = torch.all(
                torch.isclose(transform_points(trans_latent[0], T), trans_latent[1],
                              atol=1e-4))
            print('Trans equiv: ', equiv.item())
        # make feature x translation invariant
        x = x - x_center
        x = self.conv1(x)

        if self.feature_transform:
            x_global = self.fstn(x).unsqueeze(-1).repeat(1, 1, 1, N)
            x = torch.cat((x, x_global), 1)

        x = self.conv2(x)
        x = self.conv3(x)

        # translation invariant and rotation equivariant feature x_mean_out
        rot_latent = mean_pool(x) #x.mean(dim=-1, keepdim=True)

        if T is not None:
            only_rot = T
            only_rot[:3, -1] = torch.zeros(3)
            equiv = torch.all(torch.isclose(transform_points(rot_latent[0], only_rot), rot_latent[1], atol=1e-4))
            print('Rot Equiv. Latent: ', equiv.item())

        c = torch.cat([rot_latent, trans_latent], dim=1)

        return c




class VNTDistResnetEncoder(nn.Module):
    def __init__(self, c_dim=128, dim=3, hidden_dim=64, trans_c_dim=12, k=20, pooling='mean', global_feat=True, feature_transform=False, device='cpu'):
        super(VNTDistResnetEncoder, self).__init__()
        self.n_knn = k
        self.device=device
        which_norm_VNT = 'norm'


        base_ch = hidden_dim * 3
        output_ch_trans = trans_c_dim
        output_ch_rot = c_dim - output_ch_trans
        output_ch = c_dim

        self.conv_pos = VNTLinearLeakyReLU(3, base_ch // 3, dim=5, negative_slope=0.0,use_batchnorm=False,
                                                      which_norm_VNT=which_norm_VNT)
        self.conv_center_ = VNTLinearLeakyReLU(base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0,use_batchnorm=False,
                                                          which_norm_VNT=which_norm_VNT)
        self.conv_center = VNTLinearLeakyReLU(base_ch // 3, base_ch // 3, dim=4, negative_slope=0.0,use_batchnorm=False,
                                                         which_norm_VNT=which_norm_VNT)
        self.conv_trans = VNTLinearLeakyReLU(base_ch // 3, output_ch_trans, dim=4, negative_slope=0.0,use_batchnorm=False,
                                              which_norm_VNT=which_norm_VNT)

        self.fc_pos = VNLinear(64, 2 * hidden_dim)
        self.block_0 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_1 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_2 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_3 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.block_4 = VNResnetBlockFC(2 * hidden_dim, hidden_dim)
        self.fc_c = VNLinear(hidden_dim, output_ch_rot)

        self.actvn_c = VNLeakyReLU(hidden_dim, negative_slope=0.0, share_nonlinearity=False)

        self.pool_pos = mean_pool if pooling == 'mean' else VNMaxPool(64)
        self.pool_0 = mean_pool if pooling == 'mean' else VNMaxPool(hidden_dim)
        self.pool_1 = mean_pool if pooling == 'mean' else VNMaxPool(hidden_dim)
        self.pool_2 = mean_pool if pooling == 'mean' else VNMaxPool(hidden_dim)
        self.pool_3 = mean_pool if pooling == 'mean' else VNMaxPool(hidden_dim)
        self.pool_4 = mean_pool if pooling == 'mean' else VNMaxPool(hidden_dim)

        self.pool = mean_pool

        self.feature_transform = feature_transform


    def forward(self, x, T=None):
        B, D, N = x.size()

        x = x.unsqueeze(1).transpose(2,3)
        feat = get_graph_feature_cross(x, k=self.n_knn, use_global=False, device=self.device)

        x = self.conv_pos(feat)
        x = self.pool(x)
        x2 = self.conv_center_(x)
        x_center = self.conv_center(x2)
        trans_x = self.conv_trans(x_center)
        trans_latent = mean_pool(trans_x)

        if T is not None:
            equiv = torch.all(
                torch.isclose(transform_points(trans_latent[0], T), trans_latent[1],
                              atol=1e-4))
            print('Trans equiv: ', equiv.item())
        # make feature x translation invariant
        x = x - x_center

        # Resnet
        net = self.fc_pos(x)

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

        rot_latent = self.fc_c(self.actvn_c(net))

        if T is not None:
            only_rot = T
            only_rot[:3, -1] = torch.zeros(3)
            equiv = torch.all(torch.isclose(transform_points(rot_latent[0], only_rot), rot_latent[1], atol=1e-4))
            print('Rot Equiv. Latent: ', equiv.item())

        c = torch.cat([rot_latent, trans_latent], dim=1)

        return c


if __name__ == '__main__':
    from pose_estimation.utils.utils import SO3_R3

    model = VNTDistResnetEncoder(c_dim=64, k=5, feature_transform=True)#VNT_SimplePointnet(c_dim=8, k=5)
    device = 'cpu'
    pcl1 = torch.randn(24, 3)*10

    #T = torch.tensor([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]], dtype=pcl1.dtype)
    #pcl2 = transform_points_batch(pcl1, T)
    H0 = SO3_R3().sample_marginal_std(batch=1)
    T = SO3_R3(R=H0[:, :3, :3], t=H0[:, :3, -1]).to_matrix().squeeze()


    #T = torch.tensor([[0, -1, 0, 1], [1, 0, 0, 2], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=pcl1.dtype)
    pcl2 = transform_points(pcl1, T)
    #pcl2 = pcl1+10


    batch = torch.stack([pcl1, pcl2]).float().to(device)
    #batch2 = torch.stack([pcl1, pcl1]).float().to(device)
    #T = torch.tensor([
    #    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    #    [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    #], dtype=pcl1.dtype)
    #batch2_t = transform_points_batch(batch2, T)

    c = model(batch, T)
    print(c.shape)