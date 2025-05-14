import os 
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms as T
import torchvision.transforms.functional as F
import torch.distributed as dist

from utils import get_training_augmentation, get_validation_augmentation

# Supported file extensions
DEFAULT_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.npy')


###############################################################################
# DataFrame creation helper functions (set_df)
###############################################################################

def downsample_df(cfg, df, downsample_ratio=1.0):
    """
    Drops samples without any masked pixels and downsamples the training data.

    Args:
        cfg: Configuration object.
        df (DataFrame): Input DataFrame.
        downsample_ratio (float): Ratio to downsample train data.

    Returns:
        DataFrame: Downsampled DataFrame.
    """
    train_df = df[df["split"] == 'train']
    downsample_num = int(len(train_df) * downsample_ratio)
    print('Reduce Data {} ---> {}!'.format(len(train_df), min(len(train_df), downsample_num)))
    sampled_train_df = train_df.iloc[:min(len(train_df), downsample_num)]
    val_df = df[df["split"] == 'val']
    test_df = df[df["split"] == 'test']
    downsampled_df = pd.concat([sampled_train_df, val_df, test_df])
    return downsampled_df 


def print_datanum(df):
    """
    Print the number of samples per split.
    """
    for mode in ["train", "val", "test"]:
        print(f"{mode} : {len(df[df['split'] == mode])}")


def set_df_generic(cfg, image_subdir="image", mask_subdir="mask", image_replace_fn=None):
    """
    Create a DataFrame of image, mask, and clean mask paths for all splits using common logic.
    
    Args:
        cfg: Configuration object.
        image_subdir (str): Subdirectory for images under data_dir.
        mask_subdir (str): Subdirectory for masks under data_dir.
        image_replace_fn (callable, optional): Function to modify filenames in the noisy branch.
        
    Returns:
        Tuple[DataFrame, dict, str]: The combined DataFrame, a dictionary with filenames for each split,
                                     and the noisemask directory for the train split.
    """
    filenames_dict = {}
    df_list = []
    # noisemask_dir_train = None

    for mode in ['train', 'val', 'test']:
        # For training (or validation if using synthetic noisy labels), use the noisymask directory.
        if mode == 'train':
            print('Learning with Synthetic Noisy Label!')
            # noisemask_dir = getattr(cfg.utils, f"{mode}_noisymask_basedir")
            noisemask_dir = f"{cfg.utils.data_dir}/train/{cfg.data.noise_type}/"
            masknames = sorted([o for o in os.listdir(noisemask_dir)
                                if o.lower().endswith(DEFAULT_EXT)])
            mask_paths = [os.path.join(noisemask_dir, o) for o in masknames]
            if image_replace_fn:
                image_paths = [os.path.join(cfg.utils.data_dir, mode, image_subdir,
                                             image_replace_fn(o)) for o in masknames]
            else:
                image_paths = [os.path.join(cfg.utils.data_dir, mode, image_subdir, o)
                               for o in masknames]
        # Otherwise, use the normal subdirectories.
        else:
            images_dir = os.path.join(cfg.utils.data_dir, mode, image_subdir)
            image_names = sorted([o for o in os.listdir(images_dir)
                                  if o.lower().endswith(DEFAULT_EXT)])
            image_paths = [os.path.join(cfg.utils.data_dir, mode, image_subdir, o)
                           for o in image_names]
            masks_dir = os.path.join(cfg.utils.data_dir, mode, mask_subdir)
            mask_names = sorted([o for o in os.listdir(masks_dir)
                                 if o.lower().endswith(DEFAULT_EXT)])
            mask_paths = [os.path.join(cfg.utils.data_dir, mode, mask_subdir, o)
                          for o in mask_names]
            masknames = mask_names

        tmp_df = pd.DataFrame({
            'image_path': image_paths,
            'mask_path': mask_paths,
        })
        tmp_df['split'] = mode
        df_list.append(tmp_df)
        filenames_dict[mode] = masknames

    df = pd.concat(df_list)
    if cfg.data.data_size.downsample_ratio > 0:
        df = downsample_df(cfg, df, cfg.data.data_size.downsample_ratio)
    print_datanum(df)

    return df, filenames_dict, noisemask_dir


def setdf_WHU_building(cfg, debug=False):
    """
    Set DataFrame for WHU_building dataset.
    In the noisy branch, replace '.png' with '.tif' in the image filename.
    """
    def whu_replace(filename):
        return filename.replace('.png', '.tif')
    return set_df_generic(cfg, image_subdir="image", mask_subdir="mask",
                          image_replace_fn=whu_replace)


def setdf_isic2017(cfg, debug=False):
    """
    Set DataFrame for ISIC2017 dataset.
    Uses 'label_png' as the mask subdirectory.
    """
    return set_df_generic(cfg, image_subdir="image", mask_subdir="label_png")


def setdf_jsrt(cfg, debug=False):
    """
    Set DataFrame for JSRT dataset.
    In the noisy branch, remove the '_label' substring from image filenames.
    """
    def jsrt_replace(filename):
        return filename.replace("_label", "")
    return set_df_generic(cfg, image_subdir="image", mask_subdir="mask",
                          image_replace_fn=jsrt_replace)


###############################################################################
# Dataset Classes
###############################################################################

class Dataset(torch.utils.data.Dataset):
    """
    Generic Dataset class.
    Returns image, mask, and image_id.
    """
    
    def __init__(self, images_fps, masks_fps, cfg, mode="train",
                 perturbate=False, num_classes=2, classes=None, augmentation=None,
                 geo_aug=False, output_name=False):
        self.images_fps = images_fps
        self.masks_fps = masks_fps
        self.mode = mode
        self.perturbate = perturbate
        self.num_classes = num_classes
        self.class_values = list(range(num_classes))
        self.augmentation = augmentation
        self.geo_aug = geo_aug
        self.cfg = cfg
        self.output_name = output_name

    def _process_mask(self, mask):
        # For binary segmentation, convert to binary mask.
        if self.num_classes == 2 and len(mask.shape) == 3:
            mask = np.where(mask > 0, 1, 0)
            mask = mask[..., 0]
            masks = [(mask == v) for v in self.class_values]
            mask = np.stack(masks, axis=-1).astype('float')
        return mask

    def read_img(self, path):
        if path.lower().endswith(".npy"):
            img = np.load(path)
        elif path.lower().endswith(DEFAULT_EXT):
            img = cv2.imread(path)
        else:
            raise ValueError("Unsupported file extension in {}".format(path))
        return img

    def __getitem__(self, i):
        image = self.read_img(self.images_fps[i])
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.read_img(self.masks_fps[i])
        mask = self._process_mask(mask)
           
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        t = T.Compose([T.ToTensor()])
        image = t(image)
        mask = t(mask).float()
        image_id = self.images_fps[i].split('/')[-1].split(".")[0]

        if self.geo_aug:
            flip_horizontal = np.random.rand() > 0.5
            flip_vertical = np.random.rand() > 0.5
            if flip_horizontal:
                image = F.hflip(image)
                mask = F.hflip(mask)
                
            if flip_vertical:
                image = F.vflip(image)
                mask = F.vflip(mask)
                
        return (image, mask, image_id)

    def __len__(self):
        return len(self.images_fps)


class Dataset_w_params(Dataset):
    """
    Dataset class extended to also load pseudo-parameters stored on disk.
    """
    def __getitem__(self, i):
        image = self.read_img(self.images_fps[i])
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.read_img(self.masks_fps[i])
        mask = self._process_mask(mask)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        t = T.Compose([T.ToTensor()])
        image = t(image)
        mask = t(mask).float()
        image_id = self.images_fps[i].split('/')[-1].split(".")[0]

        if self.mode == 'train':
            param_path = f'{self.cfg.loss.params_base_storage}/{image_id}.npy'
            # print("os.path.abspath(cfg.loss.params_base_storage): ", os.path.abspath(self.cfg.loss.params_base_storage))
            q_param_numpy = np.load(param_path)
            q_param = torch.tensor(q_param_numpy, dtype=torch.float32)
            q_param = torch.nn.Parameter(q_param, requires_grad=False)
            if self.geo_aug:
                flip_horizontal = np.random.rand() > 0.5
                flip_vertical = np.random.rand() > 0.5
                if flip_horizontal:
                    image = F.hflip(image)
                    mask = F.hflip(mask)
                    q_param = F.hflip(q_param)
                if flip_vertical:
                    image = F.vflip(image)
                    mask = F.vflip(mask)
                    q_param = F.vflip(q_param)
            else:
                flip_horizontal, flip_vertical = False, False

            return (image, mask, image_id, q_param, (flip_horizontal, flip_vertical))
        else:
            return (image, mask, image_id)


class Dataset_w_plabels(Dataset):
    """
    Dataset class extended to load temporary pseudo-labels from disk.
    """
    def __getitem__(self, i):
        image = self.read_img(self.images_fps[i])
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = self.read_img(self.masks_fps[i])
        mask = self._process_mask(mask)

        image_id = self.masks_fps[i].split('/')[-1]
        if self.mode == 'train':
            plabel_path = f'{self.cfg.loss.plabels_base_storage}/{image_id}'
            p_label = self.read_img(plabel_path)[0]
            p_label = self._process_mask(p_label)
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask, pseudo_mask=p_label)
                image, mask, p_label = sample['image'], sample['mask'], sample['pseudo_mask']
            t = T.Compose([T.ToTensor()])
            image = t(image)
            mask = t(mask).float()
            p_label = t(p_label).float()
            if self.geo_aug:
                flip_horizontal = np.random.rand() > 0.5
                flip_vertical = np.random.rand() > 0.5
                if flip_horizontal:
                    image = F.hflip(image)
                    mask = F.hflip(mask)
                    p_label = F.hflip(p_label)
                if flip_vertical:
                    image = F.vflip(image)
                    mask = F.vflip(mask)
                    p_label = F.vflip(p_label)
            else:
                flip_horizontal, flip_vertical = False, False
            return (image, mask, image_id, p_label, (flip_horizontal, flip_vertical))
        else:
            if self.augmentation:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']
            t = T.Compose([T.ToTensor()])
            image = t(image)
            mask = t(mask).float()
            return (image, mask, image_id)


###############################################################################
# Data Loader Setup
###############################################################################

def set_loader(cfg):
    if 'bayesian_spatialcorr' in cfg.loss.name:
        dataset_class = Dataset_w_params
    else:
        dataset_class = Dataset

    # Choose the correct set_df function based on dataset name.
    if "jsrt" in cfg.data.dataset_name:
        df, filenames_dict, train_maskdir = setdf_jsrt(cfg, debug=cfg.debug)
    else:
        df, filenames_dict, train_maskdir = eval("setdf_{}".format(cfg.data.dataset_name))(cfg, debug=cfg.debug)
    
    df["image_id"] = df["image_path"].apply(lambda x: x.split('/')[-1].split(".")[0])
    train_filenames = filenames_dict['train']

    train_dataset = dataset_class(
        df[df["split"] == "train"].image_path.values.tolist(),
        df[df["split"] == "train"].mask_path.values.tolist(),
        cfg,
        mode="train",
        num_classes=cfg.data.num_classes,
        augmentation=get_training_augmentation(cfg),
        geo_aug=cfg.data.geo_aug,
        output_name=True
    )
   
    val_dataset = dataset_class(
        df[df["split"] == "val"].image_path.values.tolist(),
        df[df["split"] == "val"].mask_path.values.tolist(),
        cfg,
        mode="val",
        num_classes=cfg.data.num_classes,
        augmentation=get_validation_augmentation(cfg),
        geo_aug=False,
        output_name=True
    )

    test_dataset = dataset_class(
        df[df["split"] == "test"].image_path.values.tolist(),
        df[df["split"] == "test"].mask_path.values.tolist(),
        cfg,
        mode="test",
        num_classes=cfg.data.num_classes,
        augmentation=get_validation_augmentation(cfg),
        geo_aug=False,
        output_name=True
    )

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=cfg.utils.device,
        shuffle=True
    ) if cfg.distributed else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )

    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=dist.get_world_size(),
        rank=cfg.utils.device,
        shuffle=False
    ) if cfg.distributed else None
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=(val_sampler is None),
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        sampler=val_sampler
    )

    test_sampler = DistributedSampler(
        test_dataset,
        num_replicas=dist.get_world_size(),
        rank=cfg.utils.device,
        shuffle=False
    ) if cfg.distributed else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=(test_sampler is None),
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        sampler=test_sampler
    )

    return train_loader, val_loader, test_loader, train_filenames, train_maskdir
