import torch
import wandb
import tqdm
import torch.optim as optim
import os
from pose_estimation.utils.utils import makedir
from pose_estimation.utils.utils import print_gpu_utilization
from pose_estimation.models import Losses
import time


def train(model, losses: Losses, model_dir, train_dataloader, test_dataloader,
          epochs=10, ckpt_interval=20, scheduler_kwargs={}, device='cpu',
          ckpt = None, load_modules=[],scheduler = None, ckpt_as_pretrained=False):
    """

    :param model:
    :param losses:
    :param model_dir:
    :param train_dataloader:
    :param test_dataloader:
    :param epochs:
    :param ckpt_interval:
    :param scheduler_kwargs:
    :param device:
    :param ckpt:
    :param load_modules:
    :param scheduler:
    :param ckpt_as_pretrained:
    :return:
    """
    config = wandb.config

    model.to(device)
    if device != 'cpu': print_gpu_utilization()
    makedir(model_dir)
    ckpt_dir = os.path.join(model_dir, 'ckpt')
    makedir(ckpt_dir)

    # one optimizer for all nets
    if 'optimizer' in config['train_arguments'] and config['train_arguments']['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=config['train_arguments']['lr'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['train_arguments']['lr'])
    if scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_kwargs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_kwargs)

    # load ckpt model
    if ckpt is not None:
        print('Loading checkpoint ...')
        checkpoint = torch.load(ckpt, map_location=torch.device(device))

        # if specified in config, only load specifc pre-trained modules (e.g. pretrained point cloud encoder)
        if len(load_modules) > 0:
            # only load specifc modules of the pre-trained model
            print('Loading modules: ', load_modules)
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {key: value for key, value in checkpoint['model_state_dict'].items() if key.split('.')[0] in load_modules}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
        else:
            # load entire pre-trained model
            model.load_state_dict(checkpoint['model_state_dict'])

        # get back to ckpt training progress
        if ckpt_as_pretrained is False:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            epoch = checkpoint['scheduler']['last_epoch']
            print('Loaded Checkpoint: {}. Starting at epoch: {}'.format(ckpt, epoch))

        # restart training with pretrained model
        else:
            epoch = 0
            print('Checkpoint loaded as pretrained model.')
    else:
        epoch = 0

    for i_epoch in tqdm.tqdm(range(epoch, epochs)):

        model.train()

        # keep track of different losses
        train_loss = {
            'combined_loss': 0.0,
        }
        for (key, _) in losses.losses:
            train_loss[key] = 0.0
        test_loss = {
            'test_combined_loss': 0.0,
        }
        for (key, _) in losses.losses:
            test_loss['test_'+key] = 0.0

        start_time = time.time()
        for step, (X, gt_H, weights) in enumerate(train_dataloader):

            model_input = {
                'scene_pcl': X['scene_pcl'].to(device),
                'obj_pcl': X['obj_id'].to(device),
                'obj_points': X['obj_points'].to(device),
                'obj_point_normals': X['obj_point_normals'].to(device),
            }

            gt_H = gt_H.to(device)
            weights = weights.to(device)

            loss = losses.loss_fn(model, model_input, gt_H, weights)

            # add all loses
            batch_loss = 0.0
            for key, single_loss in loss.items():
                batch_loss += single_loss

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()


            for key, single_loss in loss.items():
                train_loss[key] += single_loss.item()
            train_loss['combined_loss'] += batch_loss.item()

        scheduler.step()

        # print gpu utilization after first epoch
        if i_epoch == 0:
            if device != 'cpu': print_gpu_utilization()

        # Evaluate on test set
        with torch.no_grad():
            model.eval()

            for step, (X, gt_H, weights) in enumerate(test_dataloader):

                model_input = {
                    'scene_pcl': X['scene_pcl'].to(device),
                    'obj_pcl': X['obj_id'].to(device),
                    'obj_points': X['obj_points'].to(device),
                    'obj_point_normals': X['obj_point_normals'].to(device),
                }

                gt_H = gt_H.to(device)
                weights = weights.to(device)

                loss = losses.loss_fn(model, model_input, gt_H, weights)

                # add all loses
                batch_loss = 0.0
                for key, single_loss in loss.items():
                    batch_loss += single_loss

                for key, single_loss in loss.items():
                    test_loss['test_'+key] += single_loss.item()
                test_loss['test_combined_loss'] += batch_loss.item()

            model.train()


        tr_ = ''.join(f"{key}: {value:.10f} " for key, value in train_loss.items())
        te_ = ''.join(f"{key}: {value:.10f} " for key, value in test_loss.items())
        print(f"Epoch {i_epoch}: ", tr_, te_)
        
        # save model in intervals
        if i_epoch % ckpt_interval == 0:
            print('store checkpoint')
            model_path = os.path.join(ckpt_dir, f'model_epoch_{i_epoch}.pth')
            torch.save({
                'epoch': i_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'config': config._as_dict()
            }, model_path)

        # log epoch stats to wandb
        wandb.log(train_loss, step=i_epoch)
        wandb.log(test_loss, step=i_epoch)

    # save final model
    model_path = os.path.join(ckpt_dir, 'model_final.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'config': config._as_dict()
    }, model_path)
