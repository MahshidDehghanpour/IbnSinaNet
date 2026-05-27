"""
trainer.py — برای Linux با 4 GPU و nccl backend
تغییرات نسبت به نسخه اصلی:
  1. matplotlib.use('TkAgg') حذف شد
  2. backend از gloo به nccl تغییر کرد (برای Linux multi-GPU)
  3. all_reduce برای جمع metrics از همه GPU ها
  4. non_blocking=True برای سرعت بیشتر
  5. world_size از torch.cuda.device_count() گرفته میشه
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
import os
from torch.utils.tensorboard import SummaryWriter

from models.tiramisu import FCDenseNet67
from utils.general_utils import list_ave
from utils.losses import get_loss


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def compute_performance(preds, targets):
    assert preds.size() == targets.size()
    pr     = preds.bool()
    tgt    = targets.bool()
    tp_map = pr & tgt
    tp     = tp_map.sum().item()
    fp_map = pr & ~tgt
    fp     = fp_map.sum().item()
    fn_map = ~pr & tgt
    fn     = fn_map.sum().item()
    tn_map = ~pr & ~tgt
    tn     = tn_map.sum().item()
    return tp, tp_map, fp, fp_map, fn, fn_map, tn, tn_map


def dice(tp, fp, fn):
    if (2 * tp + fp + fn) > 0:
        return round(2 * tp / (2 * tp + fp + fn), 5)
    return 1


def load_train_objs(opt):
    from training.argprocess import process_args
    model     = FCDenseNet67(n_classes=opt.nclasses, grow_rate=12, opt=opt)
    optimizer = opt.optimizer(
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=opt.lr_decay)
    return model, optimizer, scheduler


def process(rank: int, world_size: int, opt, fold: int,
            train_dset, val_dset, writer_dir, image_dir):

    ddp_setup(rank, world_size)

    # ── model ──────────────────────────────────────────────────────────────
    model, optimizer, scheduler = load_train_objs(opt)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])

    criterion = get_loss(getattr(opt, 'loss_type', 'CE+Dice'))

    # ── dataloaders ────────────────────────────────────────────────────────
    train_sampler = DistributedSampler(
        train_dset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(
        val_dset, num_replicas=world_size, rank=rank, shuffle=False)

    train_data = DataLoader(
        train_dset, batch_size=opt.batch_size,
        sampler=train_sampler, num_workers=4,
        pin_memory=True, drop_last=True)
    val_data = DataLoader(
        val_dset, batch_size=opt.batch_size,
        sampler=val_sampler, num_workers=4,
        pin_memory=True)

    # ── tensorboard ────────────────────────────────────────────────────────
    writer = None
    if rank == 0:
        os.makedirs(writer_dir, exist_ok=True)
        writer = SummaryWriter(writer_dir)

    global_step = 1

    for epoch in range(1, opt.num_epochs + 1):

        # ── Train ───────────────────────────────────────────────────────────
        model.train()
        train_sampler.set_epoch(epoch)
        history_loss = []
        history_dice = []

        for idx, (source, targets) in enumerate(train_data):
            source  = source.to(rank, non_blocking=True)
            targets = targets[:, 1, :, :].to(rank, non_blocking=True)

            optimizer.zero_grad()
            net_out = model(source)
            loss    = criterion(net_out, targets.long())
            loss.backward()
            optimizer.step()

            if rank == 0:
                global_step += 1
                history_loss.append(loss.item())
                with torch.no_grad():
                    outputs = F.softmax(net_out, dim=1)[:, 1, :, :]
                    preds   = (outputs > 0.5).float()
                    tp, _, fp, _, fn, _, tn, _ = compute_performance(preds, targets)
                    history_dice.append(dice(tp, fp, fn))

                if (idx + 1) % 5 == 0 or idx == len(train_data) - 1:
                    writer.add_scalar("loss_train", list_ave(history_loss[-20:]), global_step)
                    writer.add_scalar("DSC_train",  list_ave(history_dice[-20:]), global_step)

        scheduler.step()

        # ── Validation ──────────────────────────────────────────────────────
        model.eval()
        val_sampler.set_epoch(epoch)
        val_loss = 0.0
        tp = fp = fn = tn = 0

        with torch.no_grad():
            for source, targets in val_data:
                source  = source.to(rank, non_blocking=True)
                targets = targets[:, 1, :, :].to(rank, non_blocking=True)
                net_out = model(source)
                loss    = criterion(net_out, targets.long())
                val_loss += loss.item()
                outputs = F.softmax(net_out, dim=1)[:, 1, :, :]
                preds   = (outputs > 0.5).float()
                t1, _, t2, _, t3, _, t4, _ = compute_performance(preds, targets)
                tp += t1; fp += t2; fn += t3; tn += t4

        # جمع metrics از همه GPU ها
        metrics = torch.tensor(
            [val_loss, tp, fp, fn, tn],
            dtype=torch.float64, device=rank)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        val_loss_all, tp_all, fp_all, fn_all, tn_all = metrics.tolist()

        if rank == 0:
            val_loss_all /= max(len(val_data) * world_size, 1)
            val_dice = dice(int(tp_all), int(fp_all), int(fn_all))
            writer.add_scalar("Loss_val", val_loss_all, epoch)
            writer.add_scalar("DSC_val",  val_dice, epoch)
            print(f"  Epoch {epoch}/{opt.num_epochs}  "
                  f"Loss={val_loss_all:.4f}  Dice={val_dice*100:.2f}%",
                  flush=True)

    # ── Save checkpoint ─────────────────────────────────────────────────────
    if rank == 0:
        tag  = os.environ.get('CKPT_TAG', f'fold{fold}')
        PATH = f"model-{tag}.pt"
        torch.save(model.module.state_dict(), PATH)
        print(f"  Checkpoint saved: {PATH}", flush=True)
        writer.close()

    destroy_process_group()


def train(opt, fold, train_dset, val_dset, writer_dir, image_dir):
    world_size = torch.cuda.device_count()
    mp.spawn(
        process,
        args=(world_size, opt, fold, train_dset, val_dset, writer_dir, image_dir),
        nprocs=world_size,
        join=True)
