import torch

import numpy as np
import torch
import os
import theseus as th
from theseus.geometry import SO3
from enum import Enum
from mesh_to_sdf import sample_sdf_near_surface
from pynvml import *

class SceneObjDescriptor(Enum):
    NONE = 'flattened_points_descriptor'
    PAIRWISE_DIST = 'pairwise_dist_descriptor'
    DIST = 'dist_descriptor'

class SceneObjEncoderMode(Enum):

    SHARED_ENCODER = 'shared_encoder'  # model and scene are encoded with the same instance of pcl encoder
    DISTINCT_ENCODER = 'distinct_encoder'  # model and scene are encoded with distinct instances of pcl encoders


class Embedding(Enum):
    JOINT = 'joint'  # obj and scene are embedded jointly as one point clud
    SEPARATE = 'separate'  # obj and scene are embedded separately


class PclEncoder(Enum):
    POINTNET = 'Pointnet'
    ResnetPointnet = 'ResnetPointnet'
    VNNResnetPointnet = 'VNNResnetPointnet'
    VNTResnetPointnet = 'VNTResnetPointnet'
    VNTSimplePointnet = 'VNTSimplePointnet'
    VNTDistEncoder = 'VNTDistEncoder'


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

def get_points_faces(obj_mesh, n_points=5000):
    """
    Returns points and respective normal vectors from faces of the mesh. Point is the centroid if the face and
    the normal the respective normal vector of the face.

    :param obj_mesh: trimesh mesh object
    :param n_points: number of points to get
    :return: points and respective normal vectors as np.arrays
    """

    face_idxs = np.random.permutation(obj_mesh.faces.shape[0])[:n_points]
    obj_points = obj_mesh.triangles_center[face_idxs] # centers of faces
    obj_points_normals = obj_mesh.face_normals[face_idxs] # normals of faces

    return obj_points, obj_points_normals


def transform_points(points, matrix):
    """
    Transforms points according to transformation matrix

    :param points: (tensor) N x 3 points
    :param matrix: (tensor) 4 x 4 transformation matrices
    :return: transformed points
    """

    stack = torch.cat([points, points.new_ones((*points.size()[:-1], 1))], dim=-1)
    x = (matrix @ stack.T).T

    # back to cartesian coordiantes
    x = x[..., :-1] / x[..., -1, None]
    return x

def transform_points_batch(points, matrix):
    """
    Transforms batch of points according to transformation matrices

    :param points: (tensor) Batch x N x 3 points
    :param matrix: (tensor) Batch x 4 x 4 transformation matrices
    :return: batch-wise transformed points
    """

    # to homogenous coordinates
    stack = torch.cat([points, points.new_ones((*points.size()[:-1], 1))], dim=-1)

    # apply batch wise matrices
    x = torch.bmm(matrix, stack.transpose(1, 2)).transpose(1, 2)

    # back to cartesian coordiantes
    x = x[..., :-1] / x[..., -1, None]
    return x

def sample_sdf_from_mesh(mesh, absolute=True, n_points=128):
    """
    Samples SDF data near surface of object.

    :param mesh: trimesh mesh model of object
    :param absolute: compute unsigned distance function
    :param n_points: number of points to query
    :return: query point and respective signed distance function value
    """

    query_points, sdf = sample_sdf_near_surface(mesh, number_of_points=n_points, return_gradients=False)

    if absolute:
        neg_sdf_idxs = np.argwhere(sdf < 0)[:, 0]
        sdf[neg_sdf_idxs] = -sdf[neg_sdf_idxs]

    return query_points, sdf


class SO3_R3():
    """
    Adopted from: https://github.com/TheCamusean/grasp_diffusion
    """
    def __init__(self, R=None, t=None):
        self.R = SO3()
        if R is not None:
            self.R.update(R)
        self.w = self.R.log_map()
        if t is not None:
            self.t = t

    def log_map(self):
        """

        :return: translation stays as is. w \in [-pi*rad; pi*rad] (in radians)
        """

        return torch.cat((self.t, self.w), -1)

    def exp_map(self, x):
        """
         SE(3) -> R^6 [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z] rotation in radians

        :param x:
        :return:
        """

        self.t = x[..., :3]
        self.w = x[..., 3:]
        self.R = SO3().exp_map(self.w)
        return self

    def to_matrix(self):
        H = torch.eye(4).unsqueeze(0).repeat(self.t.shape[0], 1, 1).to(self.t)
        H[:, :3, :3] = self.R.to_matrix()
        H[:, :3, -1] = self.t
        return H

    # The quaternion takes the [w x y z] convention
    def to_quaternion(self):
        return self.R.to_quaternion()

    def sample(self, batch=1):
        R = SO3().rand(batch)
        t = torch.randn(batch, 3)
        H = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1).to(t)

        H[:, :3, :3] = R.to_matrix()
        H[:, :3, -1] = t
        return H

    def get_inverse(self):
        H_inv = torch.eye(4, device=self.R.device)[None, :].repeat(self.R.shape[0], 1, 1)

        R_ = self.R.inverse().to_matrix()
        t_ = -R_ @ self.t[:, :, None]

        H_inv[:, :3, :3] = R_
        H_inv[:, :3, -1] = t_.squeeze(2)

        return H_inv

    def sample_marginal_std(self, batch=1, sigma=0.5):
        """
        sample transformation from with marginal_std with time step=1.

        :param batch: batch size
        :param sigma: hyper-parameters determining std
        :return:
        """
        t = torch.ones(batch)
        marginal_prob = torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

        R = SO3().randn(batch)
        t = torch.randn(batch, 3)
        H = torch.eye(4).unsqueeze(0).repeat(batch, 1, 1).to(t)
        H[:, :3, :3] = SO3().exp_map(R.log_map()*marginal_prob[:, None]).to_matrix()
        H[:, :3, -1] = t*marginal_prob[:, None]
        return H


def makedir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if __name__ == '__main__':

    H = torch.tensor([[1,0,0,1],[0,0,-1,1],[0,1,0,1],[0,0,0,1]], dtype=torch.float32).reshape(-1,4,4)
    H2 = torch.tensor([[1,0,0,1],[0,1,0,0],[0,0,1,1],[0,0,0,1]], dtype=torch.float32).reshape(-1,4,4)
    H_in = torch.concatenate([H,H2])
    H_so3 = SO3_R3(R=H_in[:, :3, :3], t=H_in[:, :3, -1])
    inverted = H_so3.get_inverse()
