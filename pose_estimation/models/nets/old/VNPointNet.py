import torch.nn.parallel
import torch.utils.data
from pose_estimation.models.nets.old.vn_layers import *
from pose_estimation.models.nets.vn_utils import get_graph_feature_cross


class VNPointNet(nn.Module):
    ''' Based on proposed PointNet architecture.
    '''
    def __init__(self, pooling='max', n_knn=20, global_feat=True, feature_transform=False, channel=3, z_dim=128, device='cpu'):
        super(VNPointNet, self).__init__()
        self.n_knn = n_knn
        self.pooling = pooling
        self.device = device

        self.conv_pos = VNLinearLeakyReLU(channel, 64//3, dim=5, negative_slope=0.0, use_batchnorm=False)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0, use_batchnorm=False)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3, dim=4, negative_slope=0.0, use_batchnorm=False)

        self.conv3 = VNLinear(128//3, 1024//3)
        # self.bn3 = VNBatchNorm(1024//3, dim=4)

        self.std_feature = VNStdFeature(1024//3 *2, dim=4, normalize_frame=False, negative_slope=0.0, use_batchnorm=False)

        self.fc1 = nn.Linear(1024//3*6, 512)
        self.fc2 = nn.Linear(512, z_dim)
        self.relu = nn.ReLU()

        self.mlp = nn.Sequential(
            self.fc1,
            self.relu,
            self.fc2,
        )

        if self.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif self.pooling == 'mean':
            self.pool = mean_pool

        self.global_feat = global_feat
        self.feature_transform = feature_transform

        if self.feature_transform:
            self.fstn = STNkd(pooling=self.pooling, d=64//3)

    def forward(self, x):

        x = x.transpose(1, 2)
        B, D, N = x.size()  # B = Batch size, D=dimension=3, N = number of points(e.g. 1000)

        # x = x.unsqueeze(1)
        x = x.unsqueeze(1)  # .transpose(2, 3)  # necessary for get_graph_feature function
        feat = get_graph_feature_cross(x, k=self.n_knn, device=self.device)
        x = self.conv_pos(feat)
        x = self.pool(x)

        x = self.conv1(x)

        if self.feature_transform:
            # compute global features
            x_global = self.fstn(x).unsqueeze(-1).repeat(1 ,1 ,1 ,N)
            x = torch.cat((x, x_global), 1)

        # point features are local features?
        pointfeat = x
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))

        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(B, -1, N)

        x = torch.max(x, -1, keepdim=False)[0]

        trans_feat = None
        if self.global_feat:
            x = self.mlp(x)
            return x  # , trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class STNkd(nn.Module):
    def __init__(self, pooling='max', d=64, use_batchnorm=True):
        ''' Computes global features of pcl
        '''
        super(STNkd, self).__init__()
        self.pooling = pooling
        self.use_batchnorm = use_batchnorm

        self.conv1 = VNLinearLeakyReLU(d, 64 // 3, dim=4, negative_slope=0.0, use_batchnorm=self.use_batchnorm)
        self.conv2 = VNLinearLeakyReLU(64 // 3, 128 // 3, dim=4, negative_slope=0.0, use_batchnorm=self.use_batchnorm)
        self.conv3 = VNLinearLeakyReLU(128 // 3, 1024 // 3, dim=4, negative_slope=0.0, use_batchnorm=self.use_batchnorm)

        self.fc1 = VNLinearLeakyReLU(1024 // 3, 512 // 3, dim=3, negative_slope=0.0, use_batchnorm=self.use_batchnorm)
        self.fc2 = VNLinearLeakyReLU(512 // 3, 256 // 3, dim=3, negative_slope=0.0, use_batchnorm=self.use_batchnorm)

        if self.pooling == 'max':
            self.pool = VNMaxPool(1024 // 3)
        elif self.pooling == 'mean':
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