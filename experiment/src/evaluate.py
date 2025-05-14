import cv2
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import h5py
import numpy as np
import torch

from metrics import *



"""
For 2D Dataset
"""
def test(cfg, model, dataloader, criterion):
    model.eval()
    test_loss = 0
    test_bg_iou_score = 0
    test_iou_score = 0
    test_dice_score = 0
    test_recall_score = 0
    test_precision_score = 0
    test_recall_bg_score = 0
    test_precision_bg_score = 0

    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, disable=(not cfg.debug or cfg.utils.device != 0), ncols=80, leave=False)):
            batch_size = batch[0].shape[0]
            image = batch[0].to(cfg.utils.device)
            mask = batch[1].to(cfg.utils.device)

            output = model(image)
            # Evaluation metrics
            tmp_bg_iou, tmp_iou, tmp_dice = mIoU(output.cpu(), mask.cpu(), n_classes=cfg.data.num_classes)
            tmp_recall, tmp_precision, tmp_recall_bg, tmp_precision_bg = calculate_recall_precision(output.cpu(), mask.cpu())
            test_bg_iou_score += tmp_bg_iou * batch_size
            test_iou_score += tmp_iou * batch_size
            test_dice_score += tmp_dice * batch_size
            test_recall_score += tmp_recall * batch_size
            test_precision_score += tmp_precision * batch_size
            test_recall_bg_score += tmp_recall_bg * batch_size
            test_precision_bg_score += tmp_precision_bg * batch_size
            # Loss
            loss = criterion(output, mask, None)
            test_loss += loss.item() * batch_size
            total_samples += batch_size

            del image, output, mask, loss
            gc.collect()
            torch.cuda.empty_cache()
            
    dist.barrier()

    # Gather metrics from all processes
    metrics = torch.tensor([
        test_loss, test_bg_iou_score, test_iou_score, test_dice_score, test_recall_score,
        test_precision_score, test_recall_bg_score, test_precision_bg_score,
        total_samples
    ], device=cfg.utils.device)

    gathered_metrics = [torch.zeros_like(metrics) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_metrics, metrics)

    if dist.get_rank() == 0:
        gathered_metrics = torch.stack(gathered_metrics)
        test_loss, test_bg_iou_score, test_iou_score, test_dice_score, test_recall_score, test_precision_score, test_recall_bg_score, test_precision_bg_score, total_samples = gathered_metrics.sum(dim=0).cpu().numpy()

        test_loss /= total_samples
        test_bg_iou_score /= total_samples
        test_iou_score /= total_samples
        test_dice_score /=total_samples
        test_recall_score /= total_samples
        test_precision_score /= total_samples
        test_recall_bg_score /= total_samples
        test_precision_bg_score /= total_samples

    dist.barrier()

    return test_loss, test_iou_score, test_dice_score, test_bg_iou_score, test_recall_score, test_precision_score, test_recall_bg_score, test_precision_bg_score


def predict_image_mask_score(model, image, mask, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]):
    model.eval()
    image = image.cuda()
    mask = mask.cuda()
    with torch.no_grad():
        output = model(image)

        iou = mIoU(output.cpu(), mask.cpu(), n_classes=cfg.data.num_classes)
        recall, precision = calculate_recall_precision(output.cpu(), mask.cpu())
        # acc = pixel_accuracy(output, mask)

        masked = torch.argmax(output, dim=1)
        masked = masked.cpu().squeeze(0)

    return masked, iou, recall, precision


def predict_image_mask_score2(model, image, mask, mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25]):
    model.eval()
    image = image.cuda()
    mask = mask.cuda()
    with torch.no_grad():
        output = model(image)
        iou, recall, precision = calc_metrics(output.cpu(), mask.cpu(), num_classes=2, smooth=1e-6)

    return iou, recall, precision

def test_score(model, test_loader):
    score_iou = 0
    score_recall = 0
    score_precision = 0
    for images, masks, image_ids in tqdm(test_loader, ncols=80, leave=False):
        pred_mask, iou, recall, precision = predict_image_mask_score(model, images, masks)
        score_iou += iou
        score_recall += recall
        score_precision += precision
    return score_iou / len(test_loader), score_recall / len(test_loader), score_precision / len(test_loader)

def test_score2(model, test_loader):
    score_iou = 0
    score_recall = 0
    score_precision = 0
    for images, masks, image_ids in tqdm(test_loader, ncols=80, leave=False):
        iou, recall, precision = predict_image_mask_score2(model, images, masks)
        score_iou += iou
        score_recall += recall
        score_precision += precision
    return score_iou / len(test_loader), score_recall / len(test_loader), score_precision / len(test_loader)

def miou_score_one_batch(model, test_loader):
    """
    バッチサイズ1を想定した関数
    """
    score_list = {}
    for images, masks, image_ids in tqdm(test_loader, ncols=80, leave=False):
        pred_mask, score = predict_image_mask_score(model, images, masks)
        score_list[image_ids[0]] = score.item()
    return score_list

def calc_score_one_batch(model, test_loader):
    """
    バッチサイズ1を想定した関数
    """
    score_list = {}
    for images, masks, image_ids in tqdm(test_loader, ncols=80, leave=False):
        score = calc_metrics(model, images, masks)
        score_list[image_ids[0]] = score
    return score_list


