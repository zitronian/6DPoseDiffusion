import sys
import os
import yaml
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'pose_estimation'))
import wandb
from pose_estimation.samplers import Pose_AnnealedLD, Pose_PredictorCorrector, Pose_EulerMaruyama
from pose_estimation.datasets import dataset
from pose_estimation.evaluation.evaluate import evaluate
import os
import torch
from config.train_opts import parse_args, load_inference_configuration
import numpy as np
from pose_estimation.models.loader import load_model
from pose_estimation.utils.utils import PclEncoder, SceneObjEncoderMode, SceneObjDescriptor
from pose_estimation.refiner.refiner import LatentDifferenceRefiner

def run_evaluation(dataset_root, sampling_kwargs, dataset_kwargs, base_path,
                   wandb_mode='online', wandb_tags=[], wandb_note = '', cfg_file=None, device='cpu'):
    """
    Samples particles for each sample in dataset. All sampled pose hypothesis are stored in .npy as dictionary.
    Results are stored in results/wandb_run_name/results.npy

    :param dataset_root: path to dataset
    :param sampling_kwargs: kwargs for sampler
    :param dataset_kwargs: kwargs to initialize custom dataset
    :param base_path:
    :param wandb_mode: online | offline
    :param wandb_tags:
    :param wandb_note:
    :param cfg_file: <file_name>.yaml
    :param device: cuda, cpu
    :return:
    """

    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    run = wandb.init(project='Inference PoseEstimation',
                     config=config,
                     reinit=True,
                     tags=wandb_tags,
                     notes=wandb_note,
                     mode=wandb_mode)

    # store config in wandb
    if cfg_file is not None:
        run.save(cfg_file, policy='now')

    n_samples = sampling_kwargs['n_samples']
    num_steps = sampling_kwargs['num_steps']

    model.eval()

    res_path = os.path.join(base_path, 'results', run.name)
    if not os.path.exists(res_path):
        os.makedirs(res_path)

    data = dataset.CustomDataset(os.path.join(base_path, dataset_root), **dataset_kwargs)

    refiner = None
    if sampling_kwargs['refiner'] == 'latentdifference':
        refiner = LatentDifferenceRefiner()

    if sampling_kwargs['sampler'] == 'LD':
        sampler = Pose_AnnealedLD(model, render_interval=sampling_kwargs['render_interval'], device=device, refiner=refiner)
    elif sampling_kwargs['sampler'] == 'PredictorCorrector':
        sampler = Pose_PredictorCorrector(model, render_interval=config.render_interval, device=device)
    elif sampling_kwargs['sampler'] == 'EulerMaruyama':
        sampler = Pose_EulerMaruyama(model, device=device)
    else:
        raise NotImplementedError

    sample_kwargs = dict(
        num_steps=num_steps
    )
    res = evaluate(sampler, data, sample_kwargs, n_samples=n_samples, plot=False, init_close=False)
    res['num_steps'] = num_steps
    # store results
    np.save(os.path.join(res_path, 'results.npy'), res)

    with open(os.path.join(res_path, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

    run.finish()
    print('-------------------')


if __name__ == '__main__':

    args = parse_args()
    wandb_mode = args.wandb_mode

    wandb_info, model_info, sampling_kwargs, model_kwargs, \
        dataset_kwargs, config_file_path = load_inference_configuration(args.config_file)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_root = 'bop_data/lm'
    model_path = os.path.join(base_path, 'models', model_info['model_dir'], 'ckpt', model_info['model_weights'])
    model = load_model(**model_kwargs, model_path=model_path, device=device)

    wandb_tags = wandb_info['wandb_tags'] + ['_'.join([str(k) for k in dataset_kwargs['obj_ids']])]
    wandb_note = wandb_info['description']

    run_evaluation(dataset_root, sampling_kwargs, dataset_kwargs, base_path=base_path,
                   wandb_mode=wandb_mode, wandb_tags=wandb_tags, wandb_note=wandb_note,
                   cfg_file=config_file_path, device=device)
