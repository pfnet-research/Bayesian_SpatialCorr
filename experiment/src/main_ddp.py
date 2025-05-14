import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import gc
import time
import math
import uuid
from tqdm import tqdm
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms as T

import timm
from timm.scheduler import CosineLRScheduler

import hydra
from hydra.utils import instantiate

import shutil

# Local modules
import utils
import dataset
import train_ddp, train_ddp_bayesian_spatialcorr
import model_loss_wrapper
import nnUNet


@hydra.main(config_path='../config', config_name='main', version_base='1.1')
def main(cfg):
    """
    Main entry point for training.
    
    This function sets up Distributed Data Parallel (DDP), loads the dataset,
    instantiates the model and loss functions via Hydra, and starts the training loop.
    """
    # Setup for DDP 
    gpu_id = utils.ddp_setup()

    if cfg.debug:
        print('Debugging!')
        cfg.data.data_size.downsample_ratio = 0.1
        cfg.train.epoch = 10
        cfg.val.per_epoch = 1
        cfg.train.batch_size = 4

    if gpu_id == 0:
        cfg.exp_randv = str(uuid.uuid4())[:8]
        # For Bayesian Spatial Correction loss, reset and create the parameters storage directory.
        if cfg.loss.name == 'bayesian_spatialcorr':
            cfg.loss.params_base_storage = os.path.join(cfg.loss.params_base_storage, cfg.exp_randv)  # Update cfg.loss.params_base_storage in order to avoid conflict with other experiment
            os.makedirs(cfg.loss.params_base_storage, exist_ok=False)


    dist.barrier()
    
    # Broadcast configuration to all processes.
    cfg_list = [cfg]
    dist.broadcast_object_list(cfg_list, src=0)
    cfg = cfg_list[0]
    cfg.utils.device = gpu_id 

    # Save stdout log to a file if not debugging.
    if (not cfg.debug) and (cfg.utils.device == 0):
        log_path = f"{cfg.utils.save_dir}/log.txt"
        log_file = open(log_path, "w")
        sys.stdout = log_file

    # Set random seed.
    utils.seed_everything(cfg.train.seed)

    # Set dataset.
    train_loader, val_loader, test_loader, train_filenames, train_maskdir = dataset.set_loader(cfg)
    cfg.data.train_maskdir = train_maskdir

    # Set save directory for image IDs.
    cfg.utils.saveimageids_dir = f"{cfg.utils.save_dir}/saveimageids"
    if (cfg.utils.device == 0) and (cfg.data.save_imageids is not None):
        for i in cfg.data.save_imageids:
            if cfg.loss.name == 'bayesian_spatialcorr':
                os.makedirs(f'{cfg.utils.saveimageids_dir}/post_dist/{i}', exist_ok=True)
                os.makedirs(f'{cfg.utils.saveimageids_dir}/params/{i}', exist_ok=True)

    # Set model.
    model = instantiate(cfg.model.object).to(cfg.utils.device)
    train_model = instantiate(cfg.loss.object, model=model)

    # Set optimizer based on loss type.
    if cfg.loss.name == 'bayesian_spatialcorr':
        optimizer = getattr(torch.optim, cfg.optimizer.name)([
            {'params': train_model.model.parameters(), 'lr': cfg.train.max_lr, 'weight_decay': cfg.train.weight_decay, 'name': 'main_param'},
            {'params': [train_model.rho_sigma, train_model.rho_gamma], 'lr': cfg.loss.rho_lr, 'weight_decay': 0, 'name': 'rho_param'},
            {'params': [train_model.mu], 'lr': cfg.loss.mu_lr, 'weight_decay': 0, 'name': 'mu_param'},
            {'params': [train_model.sigma], 'lr': cfg.loss.sigma_lr, 'weight_decay': 0, 'name': 'sigma_param'},
        ])
    elif cfg.loss.name == 'tloss':
        optimizer = getattr(torch.optim, cfg.optimizer.name)([
            {'params': train_model.model.parameters(), 'lr': cfg.train.max_lr, 'weight_decay': cfg.train.weight_decay, 'name': 'main_param'},
            {'params': [train_model.nu], 'lr': cfg.loss.loss_lr, 'weight_decay': 0, 'name': 'nu_param'},
        ])
    else:
        optimizer = getattr(torch.optim, cfg.optimizer.name)([
            {'params': train_model.parameters(), 'lr': cfg.train.max_lr, 'weight_decay': cfg.train.weight_decay, 'name': 'main_param'}
        ])
    
    criterion_val = instantiate(cfg.val_loss.object).to(cfg.utils.device)
    sched = CosineLRScheduler(
        optimizer,
        t_initial=int(cfg.train.epoch),
        lr_min=cfg.train.max_lr / cfg.train.minlr_scale,
        warmup_t=3,
        warmup_lr_init=cfg.train.max_lr / 10,
        warmup_prefix=True
    )
   
    train_model.to(cfg.utils.device)

    if cfg.resume_basedir not in [None, "None"]:
        if not os.path.exists(cfg.resume_basedir):
            raise ValueError(f'cfg.resume_basedir {cfg.resume_basedir} does not exist!')
        save_path = f"{cfg.resume_basedir}/ckpt_current.pt"
        ckp_dep = torch.load(save_path, map_location=torch.device('cpu'))
        assert ckp_dep['method'] == cfg.loss.name
        msg = train_model.model.load_state_dict(ckp_dep['model_state_dict'])
        print(f'Loaded Model: {msg}')
        opt_state = ckp_dep['optimizer_state_dict']
        filtered_groups = [pg for pg in opt_state['param_groups'] if pg.get('name') != 'dynamic_params']
        opt_state['param_groups'] = filtered_groups
        msg = optimizer.load_state_dict(opt_state)
        print(f'Loaded Optimizer: {msg}')
        msg = sched.load_state_dict(ckp_dep['scheduler_state_dict'])
        print(f'Loaded Scheduler: {msg}')

        start_epoch = ckp_dep['epoch'] + 1
        print(f">>> Resuming from epoch {start_epoch}!")
    else:
        start_epoch = 0

    train_model = DDP(train_model, device_ids=[cfg.utils.device], find_unused_parameters=True)

    # Determine the appropriate training function based on loss type.
    train_func_name = f'train_ddp_{cfg.loss.name}' if cfg.loss.name in ['bayesian_spatialcorr'] else 'train_ddp'
    print("Using training function:", train_func_name, "for loss:", cfg.loss.name)
    history = eval(f'{train_func_name}.fit')(
        cfg, train_model, train_loader, val_loader, test_loader,
        criterion_val, optimizer, sched, start_epoch=start_epoch, train_filenames=train_filenames
    )
    if cfg.utils.device == 0:
        pd.to_pickle(history, f"{cfg.utils.save_dir}/history.pkl")
        if cfg.loss.name == 'bayesian_spatialcorr':
            shutil.rmtree(cfg.loss.params_base_storage)
            post_dist_savedir = os.path.join(cfg.utils.save_dir, 'tmp_postdists')
            shutil.rmtree(post_dist_savedir)
        if not cfg.debug:
            sys.stdout.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
