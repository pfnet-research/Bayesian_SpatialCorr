import os
import time
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils import get_lr
from evaluate import test

def gather_tensors(tensor):
    tensor = tensor.unsqueeze(0)
    output_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    return torch.cat(output_tensors, dim=0)

def fit(cfg, train_model, train_loader, val_loader, test_loader, criterion_val, optimizer, scheduler, start_epoch=0, train_filenames=None):
    torch.cuda.empty_cache()
    since = time.time()
    train_losses = []
    val_losses = []
    val_iou = []; val_dice = []; val_bg_iou = []; val_acc = []
    val_recall = []; val_precision = []; val_recall_bg = []; val_precision_bg = []
    last10ep_val_iou = []; last10ep_val_dice = []; last10ep_val_recall = []; last10ep_val_precision = []
    train_iou = []; train_acc = []
    lrs = []

    fit_time = time.time()
    scaler = GradScaler(enabled=bool(cfg.amp))

    for e in range(start_epoch, cfg.train.epoch):
        epoch_start = time.time()
        running_loss = 0.0
        total_samples = 0

        train_model.train()
        train_loader.sampler.set_epoch(e)

        for i, batch in enumerate(tqdm(train_loader, disable=(cfg.utils.device != 0), ncols=80, leave=False)):
            image = batch[0].to(cfg.utils.device)
            mask = batch[1].to(cfg.utils.device)
            image_id = batch[2]
            batch_size = image.size(0)

            optimizer.zero_grad()
            with autocast(enabled=bool(cfg.amp)):
                output, loss = train_model(image, mask, image_id, cfg.utils.device)
            scaler.scale(loss).backward()

            if cfg.train.grad_clip:
                if bool(cfg.amp):
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(train_model.module.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            total_samples += batch_size
            running_loss += loss.item() * batch_size

            del image, mask, output, loss
            gc.collect()
            torch.cuda.empty_cache()

        dist.barrier()

        # Gather running loss across processes
        local_stats = torch.tensor([running_loss, total_samples], device=cfg.utils.device)
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
        global_running_loss = local_stats[0].item() / local_stats[-1].item()

        if dist.get_rank() == 0:
            print(f"Epoch {e+1}: Global Train Loss: {global_running_loss:.3f}")

        dist.barrier()

        if (e + 1) % cfg.val.per_epoch == 0 or cfg.train.epoch - e <= 10:
            val_loss, val_iou_score, val_dice_score, val_bg_iou_score, val_recall_score, \
            val_precision_score, val_recall_bg_score, val_precision_bg_score = test(
                cfg, train_model.module.model, val_loader, criterion_val
            )
            save_path = os.path.join(cfg.utils.save_dir, 'ckpt_current.pt')
            if cfg.utils.device == 0:
                train_losses.append(global_running_loss)
                val_losses.append(val_loss)
                if cfg.train.epoch - e <= 10:
                    last10ep_val_iou.append(val_iou_score)
                    last10ep_val_dice.append(val_dice_score)
                    last10ep_val_recall.append(val_recall_score)
                    last10ep_val_precision.append(val_precision_score)
                print('Saving model checkpoint...')
                save_dict = {
                    'method': cfg.loss.name,
                    'epoch': e,
                    'model_state_dict': train_model.module.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }
                if os.path.exists(save_path):
                    os.remove(save_path)
                torch.save(save_dict, save_path)

                val_iou.append(val_iou_score)
                val_dice.append(val_dice_score)
                val_bg_iou.append(val_bg_iou_score)
                val_recall.append(val_recall_score)
                val_precision.append(val_precision_score)
                val_recall_bg.append(val_recall_bg_score)
                val_precision_bg.append(val_precision_bg_score)
                print("Epoch: {}/{}.. Train Loss: {:.3f}.. Val Loss: {:.3f}.. Val mIoU: {:.3f}.. "
                      "Val bg mIoU: {:.3f}.. Val Dice: {:.3f}.. Val Recall: {:.3f}.. "
                      "Val Precision: {:.3f}.. Val bg Recall: {:.3f}.. Val bg Precision: {:.3f}.. "
                      "Time: {:.2f}m".format(
                          e + 1, cfg.train.epoch, global_running_loss, val_loss, val_iou_score, 
                          val_bg_iou_score, val_dice_score, val_recall_score, val_precision_score,
                          val_recall_bg_score, val_precision_bg_score, (time.time() - since) / 60
                      ))
            dist.barrier()

        lrs.append(get_lr(optimizer))
        scheduler.step(e)

    # Last checkpoint saving
    save_path = os.path.join(cfg.utils.save_dir, f'last_ckpt_{cfg.train.epoch}ep.pt')
    dist.barrier()
    if cfg.utils.device == 0:
        print('Inference with last checkpoint...')
        save_dict = {
            'method': cfg.loss.name,
            'epoch': e,
            'model_state_dict': train_model.module.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict()
        }
        if cfg.utils.save:
            torch.save(save_dict, save_path)
    dist.barrier()

    state_dict = torch.load(save_path, map_location='cpu')['model_state_dict']
    train_model.module.model.load_state_dict(state_dict)
    test_loss, test_iou_score, test_dice_score, test_bg_iou_score, test_recall_score, \
    test_precision_score, test_recall_bg_score, test_precision_bg_score = test(
        cfg, train_model.module.model, test_loader, criterion_val
    )

    if cfg.utils.device == 0:
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'train_miou': train_iou,
            'val_miou': val_iou,
            'val_bg_iou': val_bg_iou,
            'val_dice': val_dice,
            'val_recall': val_recall,
            'val_precision': val_precision,
            'val_recall_bg': val_recall_bg,
            'val_precision_bg': val_precision_bg,
            'test_loss': test_loss,
            'test_iou_score': test_iou_score,
            'test_bg_iou_score': test_bg_iou_score,
            'test_dice_score': test_dice_score,
            'test_recall_score': test_recall_score,
            'test_precision_score': test_precision_score,
            'test_recall_bg_score': test_recall_bg_score,
            'test_precision_bg_score': test_precision_bg_score,
        }

        print("Test Loss: {:.3f}.. Test mIoU: {:.3f}.. Test bg mIoU: {:.3f}.. "
              "Test Dice: {:.3f}.. Test Recall: {:.3f}.. Test Precision: {:.3f}.. "
              "Test bg Recall: {:.3f}.. Test bg Precision: {:.3f}.. Time: {:.2f}m".format(
                  test_loss, test_iou_score, test_bg_iou_score, test_dice_score,
                  test_recall_score, test_precision_score, test_recall_bg_score, test_precision_bg_score,
                  (time.time() - since) / 60
              ))
        print("Last {} epoch mean scores: Val mIoU {:.3f}.. Val Dice {:.3f}.. Val Recall {:.3f}.. Val Precision {:.3f}..".format(
            len(last10ep_val_iou), np.mean(last10ep_val_iou).item(),
            np.mean(last10ep_val_dice).item(), np.mean(last10ep_val_recall).item(),
            np.mean(last10ep_val_precision).item()
        ))
        print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))

        return history
    else:
        return None
