import torch.nn as nn
from pose_estimation.utils.utils import SO3_R3, transform_points_batch
from pose_estimation.utils.transformation_utils import pertub_H
import numpy as np
import torch

'''
Legacy, not used in current 6D pose estimation method
'''

class SDFLoss:

    def __init__(self, marginal_prob_std, with_transformations=True, n_points=1024, device='cpu'):
        self.device=device
        self.marginal_prob_std=marginal_prob_std
        self.n_points = n_points
        self.with_transformations = with_transformations

    def loss_fn(self, model, model_input, ground_truth, weights, eps=1e-5):

        # sample from object pcl and transform
        point_idxs = np.random.permutation(self.n_points)

        if self.with_transformations:
            # transform obj points with transformation and add noise, apply same transformation to sdf_query_points!
            H_hat_lg, random_t, z, std = pertub_H(ground_truth, self.marginal_prob_std, eps=eps)
            H = SO3_R3().exp_map(H_hat_lg).to_matrix()
            H[:, :3, -1] = torch.zeros(3) # only rotation

            transformed_obj_points = transform_points_batch(model_input['obj_points'][:, point_idxs], H)

            # transform query points with same transformation
            transformed_query_points = transform_points_batch(model_input['sdf_query_points'], H)

            sdf_predict = model.compute_sdf(transformed_obj_points, transformed_query_points)
        else:
            sdf_predict = model.compute_sdf(model_input['obj_points'][:, point_idxs], model_input['sdf_query_points'])

        sdf_target = model_input['sdf'].reshape(-1, 1)

        loss = nn.L1Loss(reduction='mean')
        sdf_loss = loss(sdf_predict, sdf_target)

        return sdf_loss