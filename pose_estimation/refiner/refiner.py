import numpy as np
import torch
from pose_estimation.utils.utils import SO3_R3
from pose_estimation.utils.utils import transform_points_batch
class LatentDifferenceRefiner():

    def __init__(self, only_rotation=True):
        self.only_rotation=only_rotation

    def match_pointsSVD(self, A, B):
        ''' Based on https://igl.ethz.ch/projects/ARAP/svd_rot.pdf and
        https://colab.research.google.com/drive/1Zcgr9RdQoXcujfKSkiQCdgVBiA1OK38M

        :param A:
        :param B:
        :return:
        '''

        N = A.shape[1]
        A_centroid = A.mean(axis=1)
        B_centroid = B.mean(axis=1)

        AA = A - A_centroid[:, None].repeat(1, N, 1)
        BB = B - B_centroid[:, None].repeat(1, N, 1)

        cov = torch.bmm(AA.transpose(1, 2), BB)

        U, S, Vt = torch.linalg.svd(cov)

        R = torch.bmm(Vt.transpose(1, 2), U.transpose(1, 2))

        for i in range(R.shape[0]):
            if torch.linalg.det(R[i]) < 0:
                Vt[i, 2, :] *= -1
                R[i] = Vt[i].T * U[i].T

        t = -R @ A_centroid.unsqueeze(2) + B_centroid.unsqueeze(2)

        return R, t.squeeze()

    def refine(self, poses, obj_latent, scene_latent):
        ''' Computes relative rotation between points in obj_latent and scene_latent with SVD

        :param poses: tensor: n_samples x 6
        :param obj_latent: n_samples x 64 x 3
        :param scene_latent: n_samples x 64 x 3
        :return: refined pose (given pose + relative posed based on SVD)
        '''

        R, t = self.match_pointsSVD(obj_latent, scene_latent)
        relative_t = SO3_R3(R=R, t=t).log_map().detach()
        if self.only_rotation:
            refined_poses = poses
            refined_poses[:, 3:] = poses[:, 3:] + relative_t[:, 3:]
        else:
            refined_poses = poses+relative_t

        return refined_poses


if __name__ == '__main__':
    refiner = LatentDifferenceRefiner()
    obj_latent = torch.rand(2, 64, 3)
    T = SO3_R3().sample(2)
    scene_latent = transform_points_batch(obj_latent, T)
    R,t = refiner.match_pointsSVD(obj_latent, scene_latent)

    reconstructed_scene_latent = transform_points_batch(obj_latent, T)
    print('Here')