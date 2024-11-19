import torch.nn as nn
from pose_estimation.utils.utils import SO3_R3
from pose_estimation.utils.transformation_utils import pertub_H
from pose_estimation.utils.transformation_utils import batch_visible_mesh_to_pcl_scene
from pose_estimation.utils.utils import transform_points_batch
import trimesh
import torch
import theseus as th

'''
Legacy, not used in current 6D pose estimation method
'''

def weighted_mse_loss(input, target, weight):

    return (weight * (input - target) ** 2).mean()

class DistanceAwarnessLoss:

    def __init__(self, marginal_prob_std, n_points=128, n_noise_scales_per_sample=1,device='cpu', lds=False):
        self.marginal_prob_std = marginal_prob_std
        self.n_noise_scales_per_sample = n_noise_scales_per_sample
        self.n_points = n_points
        self.lds = lds
        self.device = device

    def loss_fn(self, model, model_input, groundtruth, weights, eps=1e-5):
        weights_ext = weights.unsqueeze(1).repeat(1, self.n_noise_scales_per_sample).reshape(-1)
        ground_truth_ext = groundtruth.unsqueeze(1).repeat(1, self.n_noise_scales_per_sample, 1, 1).reshape(-1, 4, 4)
        obj_faces = []
        obj_vertices = []

        for i in range(groundtruth.shape[0]):
            obj_faces += [model_input['obj_faces'][i]] * self.n_noise_scales_per_sample
            obj_vertices += [model_input['obj_vertices'][i]] * self.n_noise_scales_per_sample

        H_hat_lg, random_t, z, std = pertub_H(ground_truth_ext, self.marginal_prob_std, eps=eps)
        H_gt_lg = SO3_R3(R=ground_truth_ext[:, :3, :3], t=ground_truth_ext[:, :3, -1])

        #H_gt_lg = SO3_R3(R=groundtruth[:, :3, :3], t=groundtruth[:, :3, -1])
        #H_hat_lg, random_t, z, std = pertub_H(groundtruth, self.marginal_prob_std, eps=eps)

        x_in = SO3_R3().exp_map(H_hat_lg).log_map()
        H_hat = x_in.detach().requires_grad_(True)

        # sample points from full object model and apply transformation H_hat
        H_hat_obj_pcl = batch_visible_mesh_to_pcl_scene(obj_vertices,#model_input['obj_vertices'],
                                                        obj_faces,#model_input['obj_faces'],
                                                        SO3_R3().exp_map(H_hat).to_matrix().detach(),
                                                        n_points=self.n_points,
                                                        mark_points=False,
                                                        ray_casting=False,
                                                        front_facing=False,
                                                        device=self.device)

        # undo H_hat and apply H_gt
        gt_H = H_gt_lg.to_matrix()
        reverse_H = SO3_R3().exp_map(H_hat_lg).get_inverse()

        #
        obj_origin_pcl = transform_points_batch(H_hat_obj_pcl, reverse_H)
        H_gt_obj_pcl = transform_points_batch(obj_origin_pcl, gt_H)

        # encode scene
        model.set_scene_latent(scene_pcl=model_input['scene_pcl'],
                               batch=self.n_noise_scales_per_sample)

        # y_pred is output of network based on feature compute_distance
        y_pred = model.compute_distance(x=H_hat_obj_pcl,
                                       time_t=random_t,
                                       H=H_hat)

        # y_true is distance between points in pertubed pcl and gt pcl
        y_true = (H_gt_obj_pcl - H_hat_obj_pcl).pow(2).sum(2).sqrt()

        if self.lds:
            weights = torch.outer(weights_ext, torch.ones(self.n_points, device=self.device))
        else:
            weights = torch.ones((ground_truth_ext.shape[0], self.n_points), device=self.device)
        #loss = loss_fn(score * std.reshape(-1, 1), -z)
        loss = weighted_mse_loss(input=y_pred, target=y_true, weight=weights)
        #loss_fn = nn.L1Loss()
        #loss = loss_fn(y_pred, y_true)

        return loss