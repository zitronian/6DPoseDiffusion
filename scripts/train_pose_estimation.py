import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'pose_estimation'))

from pose_estimation.models import get_losses
from pose_estimation import datasets
import wandb
import torch
from pose_estimation.trainer import train
from torch.utils.data import DataLoader
import numpy as np
from pose_estimation.models.loader import load_model
from typing import Dict
from config.train_opts import parse_args, load_train_configuration
import yaml


def run_train(args, model_kwargs: Dict, dataset_kwargs: Dict, test_data_kwargs: Dict,
              train_data_kwargs: Dict, loss_kwargs, wandb_mode='online', wandb_tags=[],
              wandb_note = '', cfg_file=None):
    """
    Method that train Noise Conditioned Score Model for 6D pose estimation and stores the model in
    models/<wandb_run_name>. Training is logged via wandb.

    :param args: dict with general training parameters
    :param model_kwargs: dict with model keyword arguments, used to instantiate model
    :param dataset_kwargs: dict with dataset kwargs, that are applicable to train and test set
    :param test_data_kwargs: dict with test data specific arguments
    :param train_data_kwargs: dict with train data specific arguments
    :param loss_kwargs: dict that holds specificationf or loss functions and arguments to instantiate
    :param wandb_mode: either 'online' or 'offline'. Online leads to synchronization with wandb server online
    :param wandb_tags: list of strings, specify tags logged to wandb run
    :param wandb_note: string, note added to wandb run
    :param cfg_file: name of the yaml config file scripts/config/model_config/<cfg_file>.yaml

    :return: name of wandb run that identifies the training session
    """

    epochs = args['epochs']
    ckpt_interval = args['ckpt_interval']
    train_sub_dir = args['train_sub_dir']
    data_dir = args['data_dir']
    dataset = args['dataset']
    test_sub_dir = args['test_sub_dir']
    batch_size = args['batch_size']
    ckpt = args['ckpt']
    load_modules = args['load_modules']
    ckpt_as_pretrained = args['ckpt_as_pretrained']
    scheduler_kwargs = args['scheduler_kwargs']

    scheduler = None
    if 'scheduler' in args.keys():
        scheduler = args['scheduler']

    # load ckpt model and continue wandb run
    if ckpt is not None and not ckpt_as_pretrained:
        with open(os.path.abspath(os.path.join(ckpt, os.pardir, os.pardir, 'wandb_run_path.txt'))) as f:
            wb_run_path = f.read()
        api = wandb.Api()
        wandb_run = api.run(wb_run_path)
        wandb_run_id = wandb_run.id

        run = wandb.init(project='PoseEstimation', id=wandb_run_id, resume='must')
    # start new wandb run and load ckpt if provided
    else:
        config = args
        if cfg_file is not None:
            with open(cfg_file, 'r') as file:
                config = yaml.safe_load(file)
        run = wandb.init(project='PoseEstimation',
                         config=config,
                         reinit=True,
                         tags=wandb_tags,
                         notes=wandb_note,
                         mode=wandb_mode)

    # store config in wandb
    if cfg_file is not None:
        run.save(cfg_file, policy='now')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_dir_name = run.name

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # get path to ckpt model
    if ckpt is not None:
        ckpt = os.path.join(base_path, 'models', ckpt)
    model_dir = os.path.join(base_path, 'models', model_dir_name)

    dataset_root = os.path.join(base_path, data_dir, dataset)

    # load model
    pose_model = load_model(**model_kwargs, device=device)
    losses = get_losses(loss_kwargs, device=device)

    num_workers = 0 if device == 'cpu' else 4

    # dataset for training and testing is stored seperatly
    if args['train_test_split'] is None:
        train_data = datasets.CustomDataset(dataset_root, sub_dir=train_sub_dir, **dataset_kwargs,
                                            **train_data_kwargs)

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True, drop_last=True)

        test_data = datasets.CustomDataset(dataset_root, sub_dir=test_sub_dir, **dataset_kwargs, **test_data_kwargs)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=True, drop_last=True)
    # one dataset, needs to be split into train and test
    else:
        data = datasets.CustomDataset(dataset_root, sub_dir=train_sub_dir, **dataset_kwargs, **train_data_kwargs)
        indices = np.random.permutation(len(data))
        n_training_samples = int(args['train_test_split']*len(data))
        train_data = torch.utils.data.Subset(data, indices[:n_training_samples])
        test_data = torch.utils.data.Subset(data, indices[n_training_samples:])

        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                      num_workers=num_workers, pin_memory=True, drop_last=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, pin_memory=True, drop_last=True)


    train(pose_model, losses, model_dir=model_dir, train_dataloader= train_dataloader, test_dataloader=test_dataloader,
          epochs=epochs, ckpt_interval=ckpt_interval, scheduler_kwargs=scheduler_kwargs, scheduler=scheduler,
          ckpt=ckpt, load_modules=load_modules, ckpt_as_pretrained=ckpt_as_pretrained, device=device)

    # finish wandb run
    run.finish()

    return model_dir_name


if __name__ == '__main__':

    args = parse_args()
    wandb_mode = args.wandb_mode

    # load arguments from configuration file in scripts/config/model_config/
    args, model_kwargs, dataset_kwargs, test_data_kwargs, train_data_kwargs, \
        loss_kwargs, wandb_info, config_file_path = load_train_configuration(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), 'config', 'model_config', args.config_file)
    )

    wandb_tags = wandb_info['wandb_tags'] + ['_'.join([str(k) for k in dataset_kwargs['obj_ids']])]
    wandb_note = wandb_info['description']

    _ = run_train(args, model_kwargs=model_kwargs, dataset_kwargs=dataset_kwargs,
                  test_data_kwargs=test_data_kwargs,
                  train_data_kwargs=train_data_kwargs,
                  loss_kwargs=loss_kwargs,
                  wandb_mode=wandb_mode, wandb_tags=wandb_tags, wandb_note=wandb_note,
                  cfg_file=config_file_path)  # disabled


