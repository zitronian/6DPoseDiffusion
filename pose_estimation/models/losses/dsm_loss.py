import torch
from pose_estimation.utils.utils import SO3_R3
from pose_estimation.utils.transformation_utils import pertub_H
from theseus import SO3
import theseus as th
from pose_estimation.utils.utils import Embedding


def weighted_mse_loss(input, target, weight):

    return torch.sum(weight * (input - target) ** 2, dim=1).mean()


class DSMLoss:
    
    def __init__(self, marginal_prob_std, loss_weights, embedding_mode: Embedding, n_noise_scales_per_sample=1,
                 device='cpu', lds=False, diffusion_in_latent=False):
        """

        :param marginal_prob_std: function with that povides standard deviation for specific time step t
        :param loss_weights: weights for samples. W
        :param embedding_mode:
        :param n_noise_scales_per_sample: how many pertubed versions of each sample in the set are created and
        passed through the network. E.g. n_noise_scales = 4 and batch size = 4 -> effective batch size is 4x4=16
        :param device: cuda or cpu
        :param lds: if true, loss is weighted with loss_weights (label distribution smoothing)
        :param diffusion_in_latent: legacy function, always set false
        """
        self.loss_weights = loss_weights
        self.marginal_prob_std = marginal_prob_std
        self.embedding_mode = embedding_mode
        self.n_noise_scales_per_sample = n_noise_scales_per_sample
        self.lds = lds
        self.device = device
        self.diffusion_in_latent = diffusion_in_latent
    
    def loss_fn(self, model, model_input, ground_truth, weights, eps=1e-5):
        """
        Compute DSM loss. Pertubes ground_truth based on some time step t and trained NCSM to learn score.


        :param model: Noise Conditioned Score Model (NCSM)
        :param model_input: dict with input for the model (holds #batch_size samples)
        :param ground_truth: target for the samples
        :param weights: potential weights for the samples, if label distribution sampling (LDS) is used
        :param eps: smallest time step
        :return:
        """

        # extend batch with #n_noise_scales_per_sample per sample
        weights_ext = weights.unsqueeze(1).repeat(1, self.n_noise_scales_per_sample).reshape(-1)
        ground_truth_ext = ground_truth.unsqueeze(1).repeat(1, self.n_noise_scales_per_sample, 1, 1).reshape(-1, 4, 4)
        obj_points_ext = model_input['obj_points'].unsqueeze(1).repeat(
            1,self.n_noise_scales_per_sample,1,1).reshape(-1, model_input['obj_points'].shape[1], 3)
        obj_point_normals_ext = model_input['obj_point_normals'].unsqueeze(1).repeat(
            1,self.n_noise_scales_per_sample,1,1).reshape(-1, model_input['obj_point_normals'].shape[1], 3)

        # pertub extended batch
        H_hat_lg, random_t, z, std = pertub_H(ground_truth_ext, self.marginal_prob_std, eps=eps)

        x_in = SO3_R3().exp_map(H_hat_lg).log_map()
        H_hat = x_in.detach().requires_grad_(True)

        with torch.set_grad_enabled(True):
            # set scene latent
            model.set_scene_latent(scene_pcl=model_input['scene_pcl'],
                                   batch=self.n_noise_scales_per_sample)
            if not self.diffusion_in_latent:
                # set object latent
                model.set_obj_latent(obj_points=obj_points_ext,
                                     obj_point_normals=obj_point_normals_ext,
                                     H=H_hat)

            else:
                raise NotImplementedError

            # compute score
            score = model(time_t=random_t)

        # potentially weight loss per sample differently
        if self.lds:
            weights = torch.outer(weights_ext, self.loss_weights)
        else:
            weights = torch.outer(torch.ones(ground_truth_ext.shape[0], device=self.device), self.loss_weights)

        # compute (weighted) mse
        loss = weighted_mse_loss(input=(score * std.reshape(-1, 1)), target=-z, weight=weights)

        return loss
        