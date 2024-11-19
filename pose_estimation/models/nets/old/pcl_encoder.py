### This file is based on https://github.com/FlyingGiraffe/vnn/tree/master

import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pose_estimation.models.nets.old.vn_layers import *
from pose_estimation.models.nets.vn_utils import get_graph_feature_cross, get_graph_feature


class DGCNNEncoder(nn.Module):
    ''' Based on proposed DGCNN architecture.
    '''
    def __init__(self, pooling='max', n_knn = 20, z_dim=128, device='cpu'):
        super(DGCNNEncoder, self).__init__()
        self.pooling = pooling
        self.n_knn = n_knn
        self.device = device
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3, use_batchnorm=False)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3, use_batchnorm=False)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 128//3, use_batchnorm=False)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 256//3, use_batchnorm=False)

        self.conv5 = VNLinearLeakyReLU(256//3+128//3+64//3*2, 1024//3, dim=4, share_nonlinearity=True, use_batchnorm=False)
        
        self.fc_c = VNLinear((1024//3)*3*2, 128)

        self.linear1 = nn.Linear((1024//3)*6, 512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, z_dim)

        if self.pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
        elif self.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    def forward(self, x):
        x = x.transpose(1, 2)
        batch_size = x.size(0)
        x = x.unsqueeze(1)#.transpose(2,3) # necessary for get_graph_feature function

        x = get_graph_feature(x, k=self.n_knn, device=self.device)
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.n_knn, device=self.device)
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.n_knn, device=self.device)
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = get_graph_feature(x3, k=self.n_knn, device=self.device)
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.conv5(x)
        
        num_points = x.size(-1)
        
        x = x.reshape(batch_size, -1, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).reshape(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).reshape(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.linear1(x))
        x = self.dp1(x)
        x = F.leaky_relu(self.linear2(x))
        x = self.dp2(x)

        x = self.linear3(x)

        return x


class STNkd(nn.Module):
    def __init__(self, pooling='max', d=64, use_batchnorm=True):
        ''' Computes global features of pcl
        '''
        super(STNkd, self).__init__()
        self.pooling = pooling
        self.use_batchnorm=use_batchnorm
        
        self.conv1 = VNLinearLeakyReLU(d, 64//3, dim=4, negative_slope=0.0, use_batchnorm=self.use_batchnorm)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0, use_batchnorm=self.use_batchnorm)
        self.conv3 = VNLinearLeakyReLU(128//3, 1024//3, dim=4, negative_slope=0.0, use_batchnorm=self.use_batchnorm)

        self.fc1 = VNLinearLeakyReLU(1024//3, 512//3, dim=3, negative_slope=0.0, use_batchnorm=self.use_batchnorm)
        self.fc2 = VNLinearLeakyReLU(512//3, 256//3, dim=3, negative_slope=0.0, use_batchnorm=self.use_batchnorm)
        
        if self.pooling == 'max':
            self.pool = VNMaxPool(1024//3)
        elif self.pooling == 'mean':
            self.pool = mean_pool
        
        self.fc3 = VNLinear(256//3, d)
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


class PointNetEncoder(nn.Module):
    ''' Based on proposed PointNet architecture.
    '''
    def __init__(self, pooling='max', n_knn=20, global_feat=True, feature_transform=False, channel=3, z_dim=128, device='cpu'):
        super(PointNetEncoder, self).__init__()
        self.n_knn = n_knn
        self.pooling = pooling
        self.device = device
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0, use_batchnorm=False)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0, use_batchnorm=False)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3, dim=4, negative_slope=0.0, use_batchnorm=False)
        
        self.conv3 = VNLinear(128//3, 1024//3)
        #self.bn3 = VNBatchNorm(1024//3, dim=4)
        
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False, negative_slope=0.0, use_batchnorm=False)

        self.fc1 = nn.Linear(1024 // 3 * 6, 512)
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
            self.fstn = STNkd(pooling=self.pooling, d=64//3, use_batchnorm=False)

    def forward(self, x):

        x = x.transpose(1,2)
        B, D, N = x.size() #B = Batch size, D=dimension=3, N = number of points(e.g. 1000)
        
        #x = x.unsqueeze(1)
        x = x.unsqueeze(1)#.transpose(2, 3)  # necessary for get_graph_feature function
        feat = get_graph_feature_cross(x, k=self.n_knn, device=self.device)
        x = self.conv_pos(feat)
        x = self.pool(x)
        
        x = self.conv1(x)
        
        if self.feature_transform:
            # compute global features
            x_global = self.fstn(x).unsqueeze(-1).repeat(1,1,1,N)
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
            return x #, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat
        

class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, z_dim=128, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.z_dim=z_dim
        self.stn = STN3d()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, z_dim)
        self.bn4 = nn.BatchNorm1d(512)
        self.relu = nn.ReLU()
        #if self.feature_transform:
        #    self.fstn = STNkd(k=64)

    def forward(self, x):
        x = x.transpose(1, 2)
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            #trans_feat = self.fstn(x)
            #x = x.transpose(2,1)
            #x = torch.bmm(x, trans_feat)
            #x = x.transpose(2,1)
            trans_feat = None
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc4(x)))
        x = self.fc5(x)

        if self.global_feat:
            return x # , trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat