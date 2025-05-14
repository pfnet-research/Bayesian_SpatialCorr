import os, random, yaml
import re
import uuid
import subprocess
from datetime import datetime
import numpy as np
import torch
import cv2
import albumentations as A
from torch.distributed import init_process_group


def ddp_setup():
    init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(gpu_id)

    return gpu_id


def flatten_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_training_augmentation(cfg): 
    """
    ※※※※※ 幾何変換に関しては、Spatial EMLossではηの事後分布再保存時に逆変換を適用する必要があるので別途実装 ※※※※※
    """

    if cfg.data.augment_type == "default":
        train_transform = A.Compose([
                        A.Resize(cfg.data.imsize, cfg.data.imsize, interpolation=cv2.INTER_NEAREST), 
                        A.RandomBrightnessContrast((0,0.5),(0,0.5)),
                        A.GaussNoise(),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))]
                        , additional_targets={'clean_mask': 'mask', 'pseudo_mask': 'mask'})
    elif cfg.data.augment_type == "none":
        train_transform = A.Compose([
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        ]
                        , additional_targets={'clean_mask': 'mask', 'pseudo_mask': 'mask'})
    else:
        raise ValueError(f'cfg.data.augment_type {cfg.data.augment_type} does not exists!')    

    return train_transform


def get_validation_augmentation(cfg):
    val_transform = A.Compose([
                        A.Resize(cfg.data.imsize, cfg.data.imsize, interpolation=cv2.INTER_NEAREST),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                    ])

    return val_transform

