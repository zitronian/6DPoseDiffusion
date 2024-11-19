import torch
from pose_estimation.utils.utils import SO3_R3
import numpy as np
import matplotlib.pyplot as plt

def plot(poses, plot_args):
    for i, pose in enumerate(poses):
        if i % plot_args['plot_interval'] == 0:
            plt.figure()
            lineoffsets = np.array([1, 3, 5, 7, 9, 11])
            plt.eventplot(pose.detach().cpu().numpy().T,
                          orientation='horizontal',
                          colors='b',
                          lineoffsets=lineoffsets)
            if plot_args['gt'] is not None:
                for j, offset in enumerate(lineoffsets):
                    plt.scatter(plot_args['gt'].detach().cpu().numpy()[j], [offset], color='green')
            # get current axes
            ax = plt.gca()
            ax.get_yaxis().set_visible(False)
            if plot_args['const_limits']:
                plt.xlim([-3, 3])
            plt.title(f'Step: {i}')
            plt.show()


class Pose_PredictorCorrector:
    '''
    Inference Algorithm based on Predictor Corrector scheme as outlines in
    https://yang-song.net/blog/2021/score/
    '''

    def __init__(self, model, render_interval, adaptiveRendering=False, refiner=None, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.render_interval = render_interval
        self.adaptiveRendering = adaptiveRendering
        self.refiner = refiner

    def diffusion_coeff(self, t, sigma=0.5):

        return torch.tensor(sigma**t, device=self.device).clone().detach()

    def sample(self, model_input, n_samples, snr=0.16, num_steps=100, init=None, eps=1e-3,
               return_as_matrix=False, plot_args=None):

        if init is not None:
            H0 = init.repeat(n_samples, 1)
        else:
            H0 = SO3_R3().sample_marginal_std(batch=n_samples)
            H0 = SO3_R3(R=H0[:, :3, :3], t=H0[:, :3, -1]).log_map()

        model_input = {
            'scene_pcl': model_input['scene_pcl'].repeat(n_samples, 1, 1).to(self.device),
            'obj_points': model_input['obj_points'].repeat(n_samples, 1, 1).to(self.device),
            'obj_point_normals': model_input['obj_point_normals'].repeat(n_samples, 1, 1).to(self.device),
        }

        time_steps = torch.linspace(1., eps, num_steps, device=self.device)
        step_size = time_steps[0] - time_steps[1]

        Ht = H0
        # H_prev = torch.zeros_like(H0)#SO3_R3(R=torch.eye(3).unsqueeze(0).repeat(n_samples), t=torch.tensor(torch.zeros()))
        poses = torch.empty((num_steps + 1, Ht.shape[0], 6), device=self.device)
        scores = torch.empty((num_steps, Ht.shape[0], 6), device=self.device)
        obj_latent = torch.empty((n_samples, 64, 3), device=self.device)
        scene_latent = torch.empty((n_samples, 64, 3), device=self.device)

        self.model.set_scene_latent(scene_pcl=model_input['scene_pcl'])
        scene_latent = self.model.scene_latent

        summed_unrendered_transformation = torch.zeros_like(Ht)

        no_noise_from_step = num_steps - 5

        self.model.set_scene_latent(scene_pcl=model_input['scene_pcl'])
        with torch.no_grad():
            for i, time_step in enumerate(time_steps):
                Ht = Ht.to(self.device)
                n_samples_time_step = torch.ones(n_samples, device=self.device) * time_step

                # Corrector step (LV)
                self.model.set_obj_latent(obj_points=model_input['obj_points'],
                                          obj_point_normals=model_input['obj_point_normals'],
                                          H=Ht)
                score = self.model(time_t=n_samples_time_step)
                score_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(Ht.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / score_norm) ** 2
                if i >= no_noise_from_step:
                    Ht = Ht + langevin_step_size * score
                else:
                    Ht = Ht + langevin_step_size * score + torch.sqrt(2 * langevin_step_size) * torch.randn_like(
                        Ht) * 0.01

                # Predictor step (Euler-Maruyama)
                self.model.set_obj_latent(obj_points=model_input['obj_points'],
                                          obj_point_normals=model_input['obj_point_normals'],
                                          H=Ht)
                score = self.model(time_t=n_samples_time_step)
                g = self.diffusion_coeff(n_samples_time_step)
                mean_Ht = Ht + (g**2)[:, None] * score * step_size

                if i >= no_noise_from_step:
                    Ht = mean_Ht + torch.sqrt(g ** 2 * step_size)[:, None]
                else:
                    Ht = mean_Ht + torch.sqrt(g**2 * step_size)[:, None] * torch.randn_like(Ht) * 0.01

                poses[i] = Ht
                scores[i] = score

        obj_latent = self.model.obj_latent
        if self.refiner is not None:
            Ht = self.refiner.refine(Ht, obj_latent, scene_latent)

        if plot_args is not None:
            plot(poses, plot_args)

        if return_as_matrix:
            return SO3_R3().exp_map(Ht).to_matrix().cpu(), SO3_R3().exp_map(poses.reshape(-1, 6)).to_matrix().reshape(
                (poses.shape[0], -1, 4, 4)).cpu(), scores.cpu(), \
                scene_latent.cpu(), obj_latent.cpu()
            # return SO3_R3().exp_map(Ht).to_matrix(), SO3_R3().exp_map(torch.cat(poses)).to_matrix().reshape((len(poses), -1, 4, 4)), torch.stack(scores)
        else:
            return Ht, poses, scores, scene_latent, obj_latent
            # return Ht, torch.cat(poses), torch.stack(scores)


class Pose_EulerMaruyama:
    """
    Sampling via Euler Maruyama.
    """

    def __init__(self, model, device='cpu'):

        self.model = model.to(device)
        self.device = device

    def diffusion_coeff(self, t, sigma=0.5):

        return torch.tensor(sigma**t, device=self.device)

    def sample(self, model_input, n_samples, num_steps=100, init=None, eps=1e-3,
               plot_args=None, return_as_matrix=False):

        if init is not None:
            H0 = init.repeat(n_samples, 1)
        else:
            H0 = SO3_R3().sample_marginal_std(batch=n_samples)
            H0 = SO3_R3(R=H0[:, :3, :3], t=H0[:, :3, -1]).log_map()

        scene_pcl = model_input['scene_pcl'].repeat(n_samples, 1, 1)
        obj_vertices = model_input['obj_vertices'].repeat(n_samples, 1, 1)
        obj_faces = model_input['obj_faces'].repeat(n_samples, 1, 1)
        # obj_pcl = model_input['obj_pcl'].repeat(n_samples, 1)

        model_input = {
            'scene_pcl': scene_pcl.to(self.device),
            'obj_vertices': obj_vertices.to(self.device),
            'obj_faces': obj_faces.to(self.device)
            # 'obj_pcl': obj_pcl,#torch.from_numpy(np.full((n_samples, 1), 8)).float()
        }

        time_steps = torch.linspace(1., eps, num_steps, device=self.device)
        step_size = time_steps[0] - time_steps[1]

        Ht = H0
        poses = [Ht]
        scores = []
        noise_scale = 0.01
        self.model.set_scene_latent(scene_pcl=model_input['scene_pcl'])

        with torch.no_grad():
            for time_step in time_steps:
                Ht = Ht.to(self.device)
                n_samples_time_step = torch.ones(n_samples, device=self.device)*time_step
                g = self.diffusion_coeff(n_samples_time_step)
                self.model.set_obj_latent(obj_vertices=model_input['obj_vertices'],
                                          obj_faces=model_input['obj_faces'],
                                          H=Ht)
                score = self.model(time_t=n_samples_time_step)
                mean_Ht = Ht + (g ** 2)[:, None] * score * step_size
                Ht = mean_Ht + torch.sqrt(step_size) * g[:, None] * torch.randn_like(Ht) * noise_scale

                Ht = Ht.cpu()
                score_t = score.cpu()
                scores.append(score_t)
                if time_step != time_steps[-1]:
                    poses.append(Ht)
                else:
                    # no noise for last step
                    poses.append(mean_Ht.cpu())

        if plot_args is not None:
            plot(poses, plot_args)

        if return_as_matrix:
            return SO3_R3().exp_map(Ht).to_matrix(), SO3_R3().exp_map(torch.cat(poses)).to_matrix().reshape((len(poses), -1, 4, 4)), torch.stack(scores)
        else:
            return Ht, torch.cat(poses), torch.stack(scores)


class Pose_AnnealedLD:
    """
    Sampling via Langevin dynamics.
    """
    def __init__(self, model, render_interval, adaptiveRendering=False, refiner=None, device='cpu'):
        """

        :param model: NCSM for inference
        :param render_interval: ever render_interval the object model is partially rendered. In between intervals
        only the object latent is transformed.
        :param adaptiveRendering: Legacy
        :param refiner: final sampled poses can be refined. Method: e.g. Computes transformation between scene and object
        with SVD latent and adds it to sampled poses.
        :param device: cuda, cpu
        """
        self.model = model.to(device)
        self.device = device
        self.render_interval = render_interval
        self.adaptiveRendering=adaptiveRendering
        self.refiner = refiner

    def sample(self, model_input, n_samples, num_steps=100, init=None, eps=1e-5,
               plot_args=None, snr=0.2, return_as_matrix=False):
        """
        Samples poses given model_input.

        :param model_input:
        :param n_samples: number of particles to sample
        :param num_steps: number of iterations in langevin dynamics process
        :param init: initial pose to start inference from. if none, it is randomly drawn from some prior
        :param eps: minimual step size in inference
        :param plot_args:
        :param snr:
        :param return_as_matrix: if True, samples poses are returned as 4x4 matrix else as 1x6

        :return: sampled poses, history of poses during process, history of score values, final scene latent,
         final object latent
        """

        # get initial pose
        if init is not None:
            H0 = init.repeat(n_samples, 1)
        else:
            H0 = SO3_R3().sample_marginal_std(batch=n_samples)
            H0 = SO3_R3(R=H0[:, :3, :3], t=H0[:, :3, -1]).log_map()

        model_input = {
            'scene_pcl': model_input['scene_pcl'].repeat(n_samples, 1, 1).to(self.device),
            'obj_points': model_input['obj_points'].repeat(n_samples, 1, 1).to(self.device),
            'obj_point_normals': model_input['obj_point_normals'].repeat(n_samples, 1, 1).to(self.device),
        }


        time_steps = torch.linspace(1., eps, num_steps, device=self.device)

        Ht = H0
        poses = torch.empty((num_steps+1, Ht.shape[0], 6), device=self.device)
        scores = torch.empty((num_steps, Ht.shape[0], 6), device=self.device)
        poses[0] = Ht

        self.model.set_scene_latent(scene_pcl=model_input['scene_pcl'])
        scene_latent = self.model.scene_latent

        summed_unrendered_transformation = torch.zeros_like(Ht, device=self.device)

        # no noise added in final 5 iterations, no more exploration required
        no_noise_from_step = num_steps - 5
        threshold_degree = 10

        with torch.no_grad():
            for i, time_step in enumerate(time_steps):

                Ht = Ht.to(self.device)

                n_samples_time_step = torch.ones(n_samples, device=self.device) * time_step

                # legacy, idea was to only render if delta in rotation between current pose and last pose that
                # was rendered is large. Alternative is to render in fixed intervals.
                if self.adaptiveRendering:

                    x = SO3_R3().exp_map(summed_unrendered_transformation).to_matrix()[:,:3,:3]
                    #y = SO3_R3().exp_map(last_rendered_transformation).to_matrix()[:,:3,:3]
                    y = SO3_R3().exp_map(torch.zeros_like(Ht)).to_matrix()[:, :3, :3]

                    difference_cos = 0.5 * (torch.vmap(torch.trace)(torch.bmm(x, torch.linalg.inv(y))) - 1.0)
                    difference_cos = torch.minimum(torch.tensor(1.0), torch.maximum(torch.tensor(-1.0), difference_cos))
                    ang_difference = torch.acos(difference_cos)
                    ang_difference = 180*ang_difference/torch.pi

                    cond = torch.any(ang_difference > threshold_degree).item() or i == 0
                else:
                    cond = not (i % self.render_interval)

                # only set object latent (implying partial rendering and embedding object point cloud) if cond is met
                if cond:
                    self.model.set_obj_latent(obj_points=model_input['obj_points'],
                                              obj_point_normals=model_input['obj_point_normals'],
                                              H=Ht)

                    summed_unrendered_transformation[:] = 0
                else:
                    # transformation with Ht-H_prev!
                    relative_H = Ht - self.model.H
                    self.model.transform_obj_latent(relative_H, relative=True)

                    # keep track of sum of unredered transformations for adaptive rendering
                    summed_unrendered_transformation += relative_H

                score = self.model(time_t=n_samples_time_step)
                score_norm = torch.norm(score.reshape(score.shape[0], -1), dim=-1).mean()
                noise_norm = np.sqrt(np.prod(Ht.shape[1:]))
                langevin_step_size = 2 * (snr * noise_norm / score_norm) ** 2

                # don't add noise in final few iterations of process
                if i >= no_noise_from_step:
                    Ht = Ht + langevin_step_size * score
                else:
                    Ht = Ht + langevin_step_size * score + torch.sqrt(2 * langevin_step_size) * torch.randn_like(Ht)*0.01

                poses[i+1] = Ht
                scores[i] = score

        obj_latent = self.model.obj_latent

        if self.refiner is not None:
            Ht = self.refiner.refine(Ht, obj_latent, scene_latent)

        if plot_args is not None:
            plot(poses, plot_args)

        if return_as_matrix:
            return SO3_R3().exp_map(Ht).to_matrix().cpu(), SO3_R3().exp_map(poses.reshape(-1,6)).to_matrix().reshape((poses.shape[0], -1, 4, 4)).cpu(), scores.cpu(), \
                scene_latent.cpu(), obj_latent.cpu()
        else:
            return Ht, poses, scores, scene_latent, obj_latent


if __name__ == '__main__':

    pass
