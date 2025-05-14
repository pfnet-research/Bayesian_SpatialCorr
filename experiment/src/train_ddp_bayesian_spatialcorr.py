import os 
import math
import shutil
import time
import gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils import get_lr
from evaluate import test


def copy_file(params_base_storage, post_dist_savedir, name):
    """
    Copy a file from the source to the destination.
    """
    src = os.path.join(params_base_storage, name)
    dst = os.path.join(post_dist_savedir, name)
    shutil.copyfile(src, dst)


def copy_postdists(cfg):
    """
    Copies .npy files from the resume directory (if any) to the target parameters directory.
    Files are copied in batches using multithreading.
    """
    source_dir = os.path.join(cfg.resume_basedir, 'tmp_postdists')
    dest_dir = cfg.loss.params_base_storage

    post_dist_names = [name for name in os.listdir(source_dir) if '.npy' in name]

    batch_size = 500
    for i in range(0, len(post_dist_names), batch_size):
        batch_files = post_dist_names[i:i + batch_size]
        with ThreadPoolExecutor() as executor:
            list(tqdm(executor.map(lambda name: copy_file(source_dir, dest_dir, name), batch_files),
                      total=len(batch_files)))
        print(f"Batch {i // batch_size + 1} copied. Waiting for 10 seconds...")
        time.sleep(10)
    print("All files copied!")


def print_gpu_memory_usage():
    """
    Print the allocated and cached GPU memory (in GB).
    """
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 3:.2f} GB")


def gather_tensors(tensor):
    """
    Gathers a tensor from all processes (for DistributedDataParallel).
    """
    tensor = tensor.unsqueeze(0)
    output_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output_tensors, tensor)
    return torch.cat(output_tensors, dim=0)


def param_group_exists(optimizer, name):
    """
    Check whether a parameter group with a given name exists in the optimizer.
    """
    for param_group in optimizer.param_groups:
        if param_group.get('name') == name:
            return True
    return False


def optimization_step(cfg, train_model, scaler, optimizer, image, mask, image_id, post_param, e, estep=False, end_estep=False):
    """
    Perform a single optimization step with AMP if enabled.
    """
    with autocast(enabled=bool(cfg.amp)):
        output, loss, (loss1, loss2, loss3, loss4, loss5) = train_model(
            image, mask, post_param, image_id, epoch=e, end_estep=end_estep
        )
    
    scaler.scale(loss).backward()

    if cfg.train.grad_clip:
        if bool(cfg.amp):
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(train_model.module.model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_([post_param], max_norm=1.0)

    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()

    return output, loss, (loss1, loss2, loss3, loss4, loss5)


def fit(cfg, train_model, train_loader, val_loader, test_loader, criterion_val, optimizer, scheduler, start_epoch=0, train_filenames=None):
    """
    Main training loop.
    """
    torch.cuda.empty_cache()
    since = time.time()
    train_losses = []
    val_losses = []
    val_iou = []; val_dice = []; val_bg_iou = []; val_acc = []
    val_recall = []; val_precision = []; val_recall_bg = []; val_precision_bg = []
    last10ep_val_iou = []; last10ep_val_dice = []; last10ep_val_recall = []; last10ep_val_precision = []
    train_iou = []; train_acc = []
    lrs = []

    loss1_l = []; loss2_l = []; loss3_l = []; loss4_l = []; loss5_l = []

    param_group_name = 'dynamic_params'
    post_dist_savedir = os.path.join(cfg.utils.save_dir, 'tmp_postdists')

    fit_time = time.time()
    scaler = GradScaler(enabled=bool(cfg.amp))

    # Create and save initial posterior distribution parameters for η
    train_model.module.add_variable(cfg)
    if cfg.utils.device == 0:
        os.makedirs(post_dist_savedir, exist_ok=True)
        if cfg.resume_basedir is not None and cfg.resume_basedir != "None":
            copy_postdists(cfg)
        else:
            train_model.module.set_postparams(cfg, train_filenames)
    dist.barrier()

    for e in range(start_epoch, cfg.train.epoch):
        running_loss = 0
        accuracy = 0
        total_samples = 0

        # Shuffle batches (for DDP)
        train_loader.sampler.set_epoch(e)

        for i, batch in enumerate(tqdm(train_loader, disable=(cfg.utils.device != 0), ncols=80, leave=False)):
            # Batch is (image, mask, image_id, post_param, (flip_horizontals, flip_verticals))
            batch_size = batch[0].shape[0]
            image = batch[0].to(cfg.utils.device)
            mask = batch[1].to(cfg.utils.device)
            image_id = batch[2]
            post_param = batch[3].to(cfg.utils.device)
            (flip_horizontals, flip_verticals) = batch[4]

            # Remove previous dynamic parameter groups from the optimizer
            if param_group_exists(optimizer, param_group_name):
                param_group = next(pg for pg in optimizer.param_groups if pg.get('name') == param_group_name)
                params_to_remove = param_group['params']
                for param in params_to_remove:
                    if param in optimizer.state:
                        del optimizer.state[param]
                optimizer.param_groups = [pg for pg in optimizer.param_groups if pg.get('name') != param_group_name]

            # M step (optimize model parameter θ)
            for tm_param in train_model.module.model.parameters():
                tm_param.requires_grad = True
            post_param.requires_grad = False
            output, loss, (loss1, loss2, loss3, loss4, loss5) = optimization_step(
                cfg, train_model, scaler, optimizer, image, mask, image_id, post_param, e
            )
            for term in range(1, 6):
                eval(f"loss{term}_l").append(eval(f"loss{term}"))

            # E step (optimize posterior distribution parameter m,Γ)
            for tm_param in train_model.module.model.parameters():
                tm_param.requires_grad = False
            post_param.requires_grad = True
            optimizer.add_param_group({
                'params': post_param,
                'lr': cfg.loss.imgwise_lr,
                'weight_decay': cfg.loss.imgwise_wd,
                'name': param_group_name
            })
            for enum in range(cfg.loss.estep):
                output, loss, (loss1, loss2, loss3, loss4, loss5) = optimization_step(
                    cfg, train_model, scaler, optimizer, image, mask, image_id, post_param, e,
                    estep=True, end_estep=((enum + 1) == cfg.loss.estep)
                )
                for term in range(1, 6):
                    eval(f"loss{term}_l").append(eval(f"loss{term}"))

            total_samples += batch_size
            running_loss += loss.item() * batch_size

            # Resave post parameters after updating (with possible geometric augmentation)
            train_model.module.resave_postparams(
                cfg, post_param, image_id, flip_horizontals, flip_verticals,
                save2nfs=((e + 1) % cfg.val.per_epoch == 0 or cfg.train.epoch - e <= 10)
            )
            dist.barrier()

            del image, output, mask, loss, post_param, batch, image_id
            gc.collect()
            torch.cuda.empty_cache()

        dist.barrier()

        # Gather running loss across processes
        local_stats = torch.tensor([running_loss, total_samples], device=cfg.utils.device)
        dist.all_reduce(local_stats, op=dist.ReduceOp.SUM)
        global_running_loss = local_stats[0].item() / local_stats[-1].item()

        if dist.get_rank() == 0:
            print(f"Epoch {e+1}: Global Train Loss: {global_running_loss:.3f}")
            for term in range(1, 6):
                mean_loss = torch.tensor(eval(f"loss{term}_l")).mean().item()
                print(f"loss{term}: {mean_loss} ", end="")
            print()

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
