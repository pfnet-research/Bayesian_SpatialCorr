#!/usr/bin/env python3
"""
add_omissionmoise.py

This script applies omission and margin noise to segmentation masks.
It performs the following operations:
  - Reads clean mask images from an input directory.
  - For each mask, extracts connected components (objects) and randomly removes some 
    objects based on a removal ratio.
  - Optionally applies margin noise (by eroding or dilating the object boundaries).
  - Adds new objects by randomly selecting objects from other mask images based on 
    an addition ratio.
  - Saves the resulting noisy masks in an output directory.

Usage:
    python add_omissionmoise.py [options]

Example:
    python add_omissionmoise.py --seed 3 --base_maskdir "/path/to/clean_mask" \
       --add_ratio 0.05 --remove_ratio 0.05 \
       --marginnoise_prob 0.1 --not_add_anno_noise False \
       --margin_noise_option both
"""

import os
import argparse
import math
from copy import deepcopy
import numpy as np
import cv2
import random
from multiprocessing import Pool
from distutils.util import strtobool

import utils

# Total number of mask patterns available for seed selection
MASK_PATTERNS = 10

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Add omission and margin noise to segmentation masks.")
parser.add_argument('--seed', type=int, default=0, choices=list(range(MASK_PATTERNS)),
                    help="Random seed index (0 to MASK_PATTERNS-1)")
parser.add_argument('--base_maskdir', type=str,
                    help="Directory containing the base (clean) masks.")
parser.add_argument('--add_ratio', type=float, default=0.05,
                    help="Fraction of objects to add from other masks (0.0-1.0).")
parser.add_argument('--remove_ratio', type=float, default=0.05,
                    help="Fraction of objects to remove (0.0-1.0).")
parser.add_argument('--marginnoise_prob', type=float, default=0.0,
                    help="Probability to apply margin noise (0.0-1.0).")
parser.add_argument('--not_add_anno_noise', type=strtobool, default=False,
                    help="If True, do not add annotation noise.")
parser.add_argument('--margin_noise_option', default='both', choices=['both', 'erode', 'dilate'],
                    help="Specify which morphological operation to use for margin noise: both, erode, or dilate.")
args = parser.parse_args()

# Global counters for removal and addition processes
REMOVE_I = 0
ADD_I = 0

# Pre-generate random numbers to decide which objects to add or remove
random_numbers1 = np.random.rand(100000000)
BOOLIAN_VALUES_TOADD = random_numbers1 < args.add_ratio
random_numbers2 = np.random.rand(100000000)
BOOLIAN_VALUES_TOREMOVE = random_numbers2 < args.remove_ratio

def apply_random_morphology(mask, operation, kernel_size, iterations):
    """
    Apply a random morphological operation (erosion or dilation) on a binary mask.
    
    Args:
        mask (np.array): Input binary mask.
        operation (str): 'erode' or 'dilate'.
        kernel_size (int): Size of the morphological kernel.
        iterations (int): Number of iterations.
    
    Returns:
        np.array: Mask after morphological transformation.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'erode':
        return cv2.erode(mask, kernel, iterations=iterations)
    elif operation == 'dilate':
        return cv2.dilate(mask, kernel, iterations=iterations)
    else:
        return mask

def add_marginnoise(args, tmp_mask, noise_level=-1, noise_prob=0.5):
    """
    Apply margin noise to an individual object mask based on its area.
    
    Args:
        args: Parsed command-line arguments.
        tmp_mask (np.array): Binary mask of an object.
        noise_level (int): Noise level (currently not used).
        noise_prob (float): Probability of applying margin noise.
    
    Returns:
        np.array: The (possibly) modified mask.
    """
    if random.random() > noise_prob:
        return tmp_mask  # Do not apply noise
    
    building_size = np.sum(tmp_mask > 0)
    
    if args.margin_noise_option == 'both':
        operation = random.choice(['erode', 'dilate'])
    else:
        operation = args.margin_noise_option

    # Set parameters based on object size
    if building_size < 150:         # Small object
        max_kernel_size = 1
        max_iterations = 1
    elif building_size < 400:       # Medium object
        max_kernel_size = 3
        max_iterations = 2
    else:                           # Large object
        max_kernel_size = 5
        max_iterations = 3

    kernel_size = random.randint(1, max_kernel_size)
    iterations = random.randint(1, max_iterations)

    return apply_random_morphology(tmp_mask, operation, kernel_size, iterations)

def remove_objs(result, n_labels, labels, args):
    """
    For every object (excluding background) in the mask, decide probabilistically to remove it.
    Apply margin noise if needed.
    
    Args:
        result (np.array): Current result mask.
        n_labels (int): Total number of labels (including background).
        labels (np.array): Labeled mask from connected components.
        args: Parsed command-line arguments.
    
    Returns:
        np.array: Updated result mask after removal.
    """
    global REMOVE_I
    for label in range(1, n_labels):
        if BOOLIAN_VALUES_TOREMOVE[REMOVE_I]:
            REMOVE_I += 1
            continue

        tmp_mask = (labels == label).astype(np.uint8)
        if not args.not_add_anno_noise:
            tmp_mask = add_marginnoise(args, tmp_mask, noise_prob=args.marginnoise_prob)
        result = cv2.bitwise_or(result, tmp_mask)
        REMOVE_I += 1
    return result

def add_objs(result, n_labels, area_dist, original_mask):
    """
    Add new objects to the mask by selecting candidate objects from random masks.
    
    Args:
        result (np.array): Current result mask.
        n_labels (int): Total number of connected components in the current mask.
        area_dist (np.array): Array of object areas.
        original_mask (np.array): Original mask image.
    
    Returns:
        np.array: Updated result mask with added objects.
    """
    global ADD_I

    # Randomly select a mask from available files until one with at least one object is found
    while True:
        random_maskname = random.choice(args.filename_l)
        mask_path = os.path.join(args.base_maskdir, random_maskname)
        tmp_mask = cv2.imread(mask_path)[..., 0]
        ot_n_labels, ot_labels = cv2.connectedComponents(tmp_mask)
        if ot_n_labels > 1:
            break

    # Create candidate masks by cropping and resizing each object in the selected mask
    cand_masks = np.zeros((128, 128, ot_n_labels - 1), dtype=bool)
    for i in range(1, ot_n_labels):
        tmp_label = (ot_labels == i).astype(np.uint8)
        # Use the crop_resize utility with error handling
        cand_masks[:, :, i - 1] = utils.crop_resize(tmp_label)
    
    addobj_num = BOOLIAN_VALUES_TOADD[ADD_I:ADD_I + n_labels].sum()  # Count objects to add in this image
    ADD_I += n_labels
    print('addobj_num:', addobj_num)
    for i in range(addobj_num):
        result, original_mask = utils.add_obj(cand_masks, result, area_dist, original_mask)
        print(f"Added object index: {i}")
    return result

def add_noise(filename, args):
    """
    Process a single mask: remove some objects, add new ones, and save the noisy mask.
    
    Args:
        filename (str): File name of the mask.
        args: Parsed command-line arguments.
    """
    mask_path = os.path.join(args.base_maskdir, filename)
    mask = cv2.imread(mask_path)[..., 0]
    n_labels, labels = cv2.connectedComponents(mask)

    # Decompose the mask into individual object masks to compute area distribution
    discomposed_masks = np.zeros((labels.shape[0], labels.shape[1], n_labels - 1), dtype=bool)
    for i in range(1, n_labels):
        discomposed_masks[:, :, i - 1] = (labels == i).astype(np.uint8)
    area_dist = discomposed_masks.sum(axis=(0, 1))

    result = np.zeros_like(mask)
    result = remove_objs(result, n_labels, labels, args)
    print('Removal phase complete')

    original_mask = deepcopy(mask)
    result = add_objs(result, n_labels, area_dist, original_mask)
    print('Addition phase complete')

    result = np.where(result > 0, 255, 0)
    save_filename = os.path.join(args.save_dir, filename.replace('.tif', '.png'))
    cv2.imwrite(save_filename, result, [cv2.IMWRITE_JPEG_QUALITY, 100])
    print(f"Saved noisy mask: {save_filename}")

def process_file(args_tuple):
    """
    Worker function for multiprocessing over files.
    
    Args:
        args_tuple (tuple): Tuple containing (filename, args).
    """
    filename, args = args_tuple
    save_path = os.path.join(args.save_dir, filename.replace('.tif', '.png'))
    if not os.path.exists(save_path):
        print(f"Processing file: {save_path}")
        add_noise(filename, args)

def main():
    # Set seeds for reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)

    filename_l = os.listdir(args.base_maskdir)
    args.filename_l = filename_l
    args.save_dir = args.base_maskdir.replace(
        "clean_mask",
        f"removeratio-{args.remove_ratio}_addratio-{args.add_ratio}_margin-{args.margin_noise_option}-marginprob{args.marginnoise_prob}_anno-{not args.not_add_anno_noise}"
    )
    os.makedirs(args.save_dir, exist_ok=True)

    with Pool() as pool:
        pool.map(process_file, [(filename, args) for filename in filename_l])
    print('Finished processing Train set!')


if __name__ == '__main__':
    main()
