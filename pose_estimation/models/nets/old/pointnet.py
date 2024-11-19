import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
from pose_estimation.models.nets.old.vn_layers import *


class STN3d(nn.Module):
    def __init__(self, channel=3):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.ln1 = nn.LayerNorm([64, 2048]) # 1024
        self.ln2 = nn.LayerNorm([128, 2048]) # 1024
        self.ln3 = nn.LayerNorm(2048) # 1024
        self.ln4 = nn.LayerNorm(512)
        self.ln5 = nn.LayerNorm(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.relu(self.ln2(self.conv2(x)))
        x = F.relu(self.ln3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.ln4(self.fc1(x)))
        x = F.relu(self.ln5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class PointNetLayerNorm(nn.Module):
    def __init__(self, z_dim=128, global_feat = True, feature_transform = False, channel=3):
        super(PointNetLayerNorm, self).__init__()
        self.z_dim=z_dim
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.ln1 = nn.LayerNorm([64, 2048]) # 2048
        self.ln2 = nn.LayerNorm([128, 2048])
        self.ln3 = nn.LayerNorm(2048)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, z_dim)
        self.ln4 = nn.LayerNorm(512)
        self.relu = nn.ReLU()
        #if self.feature_transform:
        #    self.fstn = STNkd(k=64)

    def forward(self, x):
        x = x.transpose(1, 2)
        B, D, n_pts = x.size()

        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            x, feature = x.split(3, dim=2)
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.ln1(self.conv1(x)))
        if self.feature_transform:
            #trans_feat = self.fstn(x)
            #x = x.transpose(2,1)
            #x = torch.bmm(x, trans_feat)
            #x = x.transpose(2,1)
            trans_feat = None
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.ln2(self.conv2(x)))
        x = self.ln3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.ln4(self.fc4(x)))
        x = self.fc5(x)

        if self.global_feat:
            return x # , trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat