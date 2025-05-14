import os
import argparse
import math
from copy import deepcopy
import numpy as np
import cv2
import random
from multiprocessing import Pool


def resize_with_aspect_ratio(image, target_length=128):
    """
    Resize the input image (or mask) while preserving its aspect ratio.
    Ensures that the longer side equals target_length and that both new dimensions
    are at least 1 pixel.
    
    Args:
        image (np.array): Input image or mask.
        target_length (int): Target length for the longer edge.
    
    Returns:
        np.array: Resized image.
    """
    height, width = image.shape[:2]
    if width > height:
        scale = target_length / width
        new_width = target_length
        new_height = max(int(height * scale), 1)
    else:
        scale = target_length / height
        new_height = target_length
        new_width = max(int(width * scale), 1)
        
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return resized_image


def center_image_on_canvas(image, canvas_size=(128, 128)):
    canvas = np.zeros((canvas_size[1], canvas_size[0]), dtype=np.uint8)
    resized_height, resized_width = image.shape[:2]

    start_x = (canvas_size[0] - resized_width) // 2
    start_y = (canvas_size[1] - resized_height) // 2

    canvas[start_y:start_y + resized_height, start_x:start_x + resized_width] = image

    return canvas

def crop_resize(mask, target_size=128):
    """
    Crop the mask to its bounding box, resize it while preserving the aspect ratio,
    and then center it on a canvas of size target_size x target_size.

    If the mask is completely empty, return a blank canvas.
    
    Args:
        mask (np.array): Binary mask.
        target_size (int): Size of the output canvas (default is 128).
    
    Returns:
        np.array: Processed mask with dimensions (target_size, target_size).
    """
    # Identify rows and columns containing non-zero pixels
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    # If mask is completely empty, return a blank canvas
    if not np.any(rows) or not np.any(cols):
        return np.zeros((target_size, target_size), dtype=np.uint8)
    
    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]
    ymin, ymax = y_indices[0], y_indices[-1]
    xmin, xmax = x_indices[0], x_indices[-1]
    
    cropped_mask = mask[ymin:ymax+1, xmin:xmax+1]
    resized_mask = resize_with_aspect_ratio(cropped_mask, target_length=target_size)
    centered_mask = center_image_on_canvas(resized_mask, canvas_size=(target_size, target_size))
    return centered_mask

def select_mask(cand_masks, area_dist):
    if len(area_dist) != 0:
        tmp_area = random.choice(area_dist)
    else:
        tmp_area = 400
    tmp_mask = cand_masks[..., random.randint(0, cand_masks.shape[-1]-1)].astype(np.uint8)
    pre_area = tmp_mask.sum()
    ratio = math.sqrt(tmp_area/pre_area)
    resized_mask = cv2.resize(tmp_mask, (int(tmp_mask.shape[0]*ratio), int(tmp_mask.shape[1]*ratio)))

    return resized_mask

def put_obj(cand_masks, pres_masks, area_dist, original_mask):
    height, width = pres_masks.shape[:2]
    counter = 0
    while 1:
        original_mask_copy = deepcopy(original_mask)
        original_mask_copy = np.where(original_mask_copy>0, 1, 0)
        resized_mask = select_mask(cand_masks, area_dist)
        w,h = resized_mask.shape[:2]
        if (w >= height-w) or (h >= width-h):
            continue
        rx, ry = random.randint(w, height-w), random.randint(h, width-h)
        original_mask_copy[rx:rx+w, ry:ry+h] += resized_mask
        occu = (original_mask_copy>1).sum()
        print(occu)
        if occu == 0:
            pres_masks[rx:rx+w, ry:ry+h] = resized_mask
            original_mask_copy[rx:rx+w, ry:ry+h] = resized_mask
            return pres_masks, original_mask_copy

        counter += 1
        if counter > 100:
            return pres_masks, original_mask_copy

    
def add_obj(cand_masks, pres_mask, area_dist, original_mask):
    pres_mask = np.where(pres_mask>0, 1, 0)
    mask_ = deepcopy(pres_mask)
    pres_masks, original_mask = put_obj(cand_masks, mask_, area_dist, original_mask)

    return pres_masks, original_mask