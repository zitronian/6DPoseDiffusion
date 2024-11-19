from pose_estimation.models import DSMLoss, ClassificationLoss, DistanceAwarnessLoss, SDFLoss
from typing import Dict, Tuple, List
import torch
import numpy as np

def marginal_prob_std(t, sigma=0.5):
    ''' Compute standard deviation of t-pertubed given unpertubed sample x $p_{0t}(x(t) | x(0))$.

    :param t: A vector of time steps
    :param sigma: Sigma in SDE ()default 0.5
    :return: Standard deviation
    '''
    return torch.sqrt((sigma ** (2 * t) - 1.) / (2. * np.log(sigma)))

def get_losses(args, device):
    losses = []
    for key, value in args.items():
        if key == 'SDFLoss':
            losses.append((
                key,
                {
                    'device': device,
                    'marginal_prob_std': marginal_prob_std,
                    'with_transformations': value['with_transformations'],
                }
            ))

        elif key == 'DSM':
            losses.append((
                key,
                {
                    'marginal_prob_std': marginal_prob_std,
                    'embedding_mode': value['embedding_mode'],
                    'n_noise_scales_per_sample': value['n_noise_scales_per_sample'],
                    'loss_weights': torch.tensor(value['loss_weights'], device=device),
                    'lds': value['lds'],
                    'diffusion_in_latent': value['diffusion_in_latent'],
                    'device': device,
                }
            ))

    return Losses(losses)

class Losses:
    """
    Class that contains all losses deployed during training.
    """

    def __init__(self, losses: List[Tuple[str, Dict]]):
        """

        :param losses: dict with name of loss as key and kwargs as value.
        """
        self.losses = []

        for (loss, kwargs) in losses:

            if loss == 'DSM':
                self.losses.append((loss, DSMLoss(**kwargs)))
            elif loss == 'BCE':
                self.losses.append((loss, ClassificationLoss(**kwargs)))
            elif loss == 'DistanceAwareness':
                self.losses.append((loss, DistanceAwarnessLoss(**kwargs)))
            elif loss == 'LatentRegularization':
                self.losses.append((loss, LatentRegularizationLoss(**kwargs)))
            elif loss == 'SDFLoss':
                self.losses.append((loss, SDFLoss(**kwargs)))
            else:
                raise NotImplementedError



    def loss_fn(self, model, model_input, ground_truth, weights):

        loss_dict = {}
        for (key, Loss) in self.losses:
            loss = Loss.loss_fn(model, model_input, ground_truth, weights)
            loss_dict[key] = loss

        return loss_dict