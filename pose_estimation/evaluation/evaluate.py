import torch
import torch.nn as nn
from pose_estimation.utils.utils import SO3_R3
import theseus as th
from theseus import SO3
import matplotlib.pyplot as plt
from pose_estimation.models import DGCNNEncoder, PoseModel, ScoreNet, DummyNet, PointNetEncoder, PointNetLayerNorm, \
    SingleChannelVNNResnetWrapper, TimeDependentFeatureEncoder, SimplePointnet
from pose_estimation.datasets import dataset
from pose_estimation.utils.transformation_utils import pertub_H
from pose_estimation.samplers import Pose_AnnealedLD, Pose_PredictorCorrector, Pose_EulerMaruyama
from torch.utils.data import DataLoader
from pose_estimation import datasets
from pose_estimation.evaluation.pose_error import add, re, te, adds
from pose_estimation.evaluation.score import calc_scores
from pose_estimation.evaluation.sample_selection import select_by_score_hist, select_by_latent, select_by_gt, median_selection
import trimesh
import time
import tqdm
import numpy as np
import os
import json
import wandb

def evaluate(sampler, eval_data, sample_kwargs={}, plot=False, n_samples=2, scale=500,
             init_close = False, verbose=False):
    """
    Conducts inference for all samples in dataset.

    :param sampler: sampler class
    :param eval_data: dataset
    :param sample_kwargs:
    :param plot:
    :param n_samples: number of particles to draw
    :param scale: point cloud is scaled
    :param init_close: if true, init pose in inference is sampled close to ground truth
    :param verbose:
    :return: Results dictionary.
    """

    # get respective latent size
    z_dim = sampler.model.scene_encoder.c_dim

    results_dict = {
        'add': np.empty((len(eval_data), n_samples)),
        're': np.empty((len(eval_data), n_samples)),
        'te': np.empty((len(eval_data), n_samples)),
        'sample_time': np.empty(len(eval_data)),
        'score_hist': np.empty((len(eval_data), sample_kwargs['num_steps'], n_samples, 6)),
        'pose_hist': np.empty((len(eval_data), sample_kwargs['num_steps']+1, n_samples, 4, 4)),
        'scene_latent': np.empty((len(eval_data), n_samples, z_dim, 3)),
        'obj_latent': np.empty((len(eval_data), n_samples, z_dim, 3)),
        'obj_id': np.empty((len(eval_data),1)),
        'scene_id': np.empty((len(eval_data),1)),
        'block_id': np.empty((len(eval_data), 1)),
        'scene_mean': np.empty((len(eval_data), 3)),
    }

    pbar = tqdm.tqdm(desc='Iterating over dataset', total=len(eval_data))
    with pbar:
        for i, (X, y, _) in enumerate(eval_data):

            gt = SO3_R3(R=y[None, :3, :3], t=y[None, :3, -1])
            plot_args=None
            if plot:
                plot_args = {
                    'gt': gt.log_map()[0],
                    'const_limits': True,
                    'plot_interval': 10
                }
            if init_close:
                # add std 0.5
                init = gt.log_map() + torch.randn_like(gt.log_map()) * 0.5
            else:
                init = None


            start_time = time.time()
            sampled_poses, pose_hist, score_hist, scene_latent, obj_latent = sampler.sample(model_input=X, init=init, n_samples=n_samples, plot_args=plot_args,
                                              return_as_matrix=True, **sample_kwargs)
            sampled_poses = SO3_R3(R=sampled_poses[:, :3, :3], t=sampled_poses[:, :3, -1])
            sample_time = time.time()-start_time

            # rescale translation and uncenter
            pcl_mean = X['scene_mean']
            gt.t = (gt.t + pcl_mean)*scale
            sampled_poses.t = (sampled_poses.t + pcl_mean) * scale

            # rescale object pcl (object is not centered only scaled!)
            #obj_mesh = trimesh.Trimesh(vertices=X['obj_vertices'], faces=X['obj_faces'])
            #object_pcl = np.array(np.array(obj_mesh.sample(1024)) * scale)
            object_pcl = X['obj_points'][:1024].numpy()*scale

            avg_distances = np.empty(sampled_poses.t.shape[0])
            re_errors = np.empty(sampled_poses.t.shape[0])
            te_errors = np.empty(sampled_poses.t.shape[0])

            for j, (R, t) in enumerate(list(zip(sampled_poses.R, sampled_poses.t))):
                R_est = R.numpy()
                t_est = t[:, None].numpy()
                R_gt = gt.R[0].numpy()
                t_gt = gt.t[0][:, None].numpy()

                # compute ADD, rotation error and translation error in original scale and uncentered!
                # of symmetric object compute adds
                if X['obj_id'] in [3, 10, 11]:
                    avg_distances[j] = adds(R_est, t_est, R_gt, t_gt, object_pcl)
                else:
                    avg_distances[j] = add(R_est, t_est, R_gt, t_gt, object_pcl)
                re_errors[j] = re(R_est, R_gt)
                te_errors[j] = te(t_est, t_gt)

            results_dict['add'][i] = avg_distances
            results_dict['re'][i] = re_errors
            results_dict['te'][i] = te_errors
            results_dict['sample_time'][i] = sample_time
            results_dict['score_hist'][i] = score_hist.numpy()
            results_dict['pose_hist'][i] = pose_hist.numpy()
            results_dict['obj_id'][i] = X['obj_id'].numpy()
            results_dict['scene_id'][i] = X['scene_id'].numpy()
            results_dict['block_id'][i] = X['block_id'].numpy()
            results_dict['scene_mean'][i] = X['scene_mean'].numpy()
            results_dict['scene_latent'][i] = scene_latent.detach().numpy()
            results_dict['obj_latent'][i] = obj_latent.detach().numpy()

            if verbose:
                print(
                    f'i: {i}, te: {te_errors.mean():.4f}, re: {re_errors.mean():.4f}, add: {avg_distances.mean():.4f}, '.format(
                        results_dict['add'][i]))
            pbar.update()

            # get pose estimation according to methods
            selection_res = {}
            selection_input = {key: value[i][None, ...] for key, value in results_dict.items()}
            for key, func in [('by_gt', select_by_gt), ('by_latent', select_by_latent),
                              ('by_score', select_by_score_hist), ('by_median', median_selection)]:
                selection_res[key] = func(selection_input, metric='add', n_best=1).item()
            wandb.log(
                selection_res,
                step=i
            )

            wandb.log({
                'add mean': avg_distances.mean(),
                're mean': re_errors.mean(),
                'te mean': te_errors.mean(),
                'sample_time': sample_time,
                'identifier': '{}-{}-{}'.format(X['block_id'].item(), X['scene_id'].item(), X['obj_id'].item()),
            }, step=i)

    results_dict['obj_id'] = results_dict['obj_id'].flatten()
    results_dict['scene_id'] = results_dict['scene_id'].flatten()

    return results_dict


def eval_scores(results_dict, models_info, correct_ths=np.linspace(0, 0.1, 10), normalize_by_diameter=True, error_type='add', n_top=5):
    # auc as mean of recalls over different thresholds

    avg_recalls = []
    obj_recalls = {obj: [] for obj in np.unique(results_dict['obj_id'])}
    for th in correct_ths:
        score_dict = calc_scores(results_dict, models_info, n_top=n_top, correct_th=th, error_type=error_type,
                                 normalize_by_diameter=normalize_by_diameter)
        avg_recalls.append(score_dict['avg_recall'])
        for obj, recall in score_dict['obj_recalls'].items():
            obj_recalls[obj].append(recall)

    overall_auc = np.array(avg_recalls).mean()
    obj_auc = {obj: None for obj in obj_recalls.keys()}
    for obj, recalls in obj_recalls.items():
        obj_auc[obj] = np.array(recalls).mean()

    eval_scores_dict = {
        'auc': float(overall_auc),  # auc as mean of all obj_auc
        'obj_auc': obj_auc,  # auc per object as mean of all recalls and thresholds for respective object
        'avg_recalls': avg_recalls,  # average recalls over all objects for each threshold (overall auc curve)
        'obj_avg_recalls': obj_recalls,  # average recalls for each object and each threshold (object specific auc curve)
        'correct_ths': correct_ths,  # thresholds
    }

    return eval_scores_dict


if __name__ == '__main__':

    pass
