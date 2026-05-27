"""
plot_curves.py
==============
دو نوع خروجی می‌ده:
  1. figure13_original.png  — نمودارهای جداگانه هر fold (مثل شکل 13 اصلی)
  2. figure13_overlay.png   — همه فلدها روی هم (چیزی که داور خواسته)

اجرا:
    python plot_curves.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.tensorboard.backend.event_processing.event_accumulator \
    import EventAccumulator

RESULTS_DIR = './results_ms'
N_FOLDS     = 5
REP         = 1
COLORS      = ['#378ADD', '#E24B4A', '#1D9E75', '#EF9F27', '#7F77DD']


def load_scalar(log_dir, tag):
    """مقادیر یه scalar رو از TensorBoard بخون"""
    try:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        if tag not in ea.Tags()['scalars']:
            return None, None
        events = ea.Scalars(tag)
        return [e.step for e in events], [e.value for e in events]
    except Exception:
        return None, None


def smooth(values, weight=0.85):
    """exponential moving average"""
    out, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        out.append(last)
    return out


def style_ax(ax, title, xlabel, ylabel, ylim=None):
    ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ylim:
        ax.set_ylim(ylim)


# ═══════════════════════════════════════════════════════════════════════════
# شکل ۱ — نمودارهای جداگانه هر fold (مثل شکل 13 اصلی)
# ═══════════════════════════════════════════════════════════════════════════

def plot_original():
    """
    مثل شکل 13 اصلی مقاله:
    برای هر fold دو نمودار Loss و Dice جداگانه
    """
    fig, axes = plt.subplots(N_FOLDS, 2,
                             figsize=(12, N_FOLDS * 3),
                             squeeze=False)
    fig.suptitle('IbnSinaNet — Training curves per fold',
                 fontsize=13, fontweight='bold')

    for fold in range(1, N_FOLDS + 1):
        log_dir = os.path.join(RESULTS_DIR, f'rep{REP:02d}',
                               f'fold{fold}', 'tb')
        color = COLORS[fold - 1]
        ax_loss = axes[fold - 1][0]
        ax_dice = axes[fold - 1][1]

        # ── Loss plot ──────────────────────────────────────────────────────
        steps, vals = load_scalar(log_dir, 'loss_train')
        if steps:
            ax_loss.plot(steps, smooth(vals), color=color,
                        linewidth=1.5, label='Training Loss')

        steps, vals = load_scalar(log_dir, 'Loss_val')
        if steps:
            ax_loss.plot(steps, smooth(vals), color=color,
                        linewidth=1.5, linestyle='--', label='Validation Loss')

        style_ax(ax_loss, f'Fold {fold} — Loss', 'Epoch', 'Loss')

        # ── Dice plot ──────────────────────────────────────────────────────
        steps, vals = load_scalar(log_dir, 'DSC_train')
        if steps:
            ax_dice.plot(steps, smooth(vals), color=color,
                        linewidth=1.5, label='Training Dice')

        steps, vals = load_scalar(log_dir, 'DSC_val')
        if steps:
            ax_dice.plot(steps, smooth(vals), color=color,
                        linewidth=1.5, linestyle='--', label='Validation Dice')

        style_ax(ax_dice, f'Fold {fold} — Dice', 'Epoch', 'Dice', (0, 1))

    plt.tight_layout()
    plt.savefig('figure13_original.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("  ذخیره شد: figure13_original.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# شکل ۲ — همه فلدها روی هم (چیزی که داور خواسته)
# ═══════════════════════════════════════════════════════════════════════════

def plot_overlay():
    """
    همه فلدها روی هم در 4 نمودار:
    Training Loss | Validation Dice (overlaid)
    Validation Loss | Mean ± Std Dice
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('IbnSinaNet — All folds overlaid',
                 fontsize=13, fontweight='bold')

    ax_tloss = axes[0][0]
    ax_vdice = axes[0][1]
    ax_vloss = axes[1][0]
    ax_mean  = axes[1][1]

    all_val_dice = []
    all_epochs   = None

    for fold in range(1, N_FOLDS + 1):
        log_dir = os.path.join(RESULTS_DIR, f'rep{REP:02d}',
                               f'fold{fold}', 'tb')
        color = COLORS[fold - 1]
        label = f'Fold {fold}'

        # Training Loss
        steps, vals = load_scalar(log_dir, 'loss_train')
        if steps:
            ax_tloss.plot(steps, smooth(vals), color=color,
                         linewidth=1.5, label=label, alpha=0.85)

        # Validation Loss
        steps, vals = load_scalar(log_dir, 'Loss_val')
        if steps:
            ax_vloss.plot(steps, smooth(vals), color=color,
                         linewidth=1.5, label=label, alpha=0.85)

        # Validation Dice
        steps, vals = load_scalar(log_dir, 'DSC_val')
        if steps:
            ax_vdice.plot(steps, smooth(vals), color=color,
                         linewidth=1.5, label=label, alpha=0.85)
            all_val_dice.append(smooth(vals))
            if all_epochs is None:
                all_epochs = steps

    # Mean ± Std
    if all_val_dice and all_epochs:
        min_len  = min(len(d) for d in all_val_dice)
        arr      = np.array([d[:min_len] for d in all_val_dice])
        epochs   = all_epochs[:min_len]
        mean_d   = arr.mean(axis=0)
        std_d    = arr.std(axis=0)

        # خط میانگین روی نمودار overlay هم
        ax_vdice.plot(epochs, mean_d, color='#2C2C2A',
                     linewidth=2.5, linestyle='--', label='Mean')

        # نمودار mean ± std
        ax_mean.plot(epochs, mean_d, color='#378ADD',
                    linewidth=2.5, label='Mean Dice')
        ax_mean.fill_between(epochs,
                             mean_d - std_d,
                             mean_d + std_d,
                             alpha=0.2, color='#378ADD', label='± Std')

    style_ax(ax_tloss, 'Training Loss — all folds',    'Epoch', 'Loss')
    style_ax(ax_vdice, 'Validation Dice — all folds',  'Epoch', 'Dice',  (0, 1))
    style_ax(ax_vloss, 'Validation Loss — all folds',  'Epoch', 'Loss')
    style_ax(ax_mean,  'Mean ± Std Dice across folds', 'Epoch', 'Dice',  (0, 1))

    plt.tight_layout()
    plt.savefig('figure13_overlay.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("  ذخیره شد: figure13_overlay.png")
    plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print(f"\nرسم نمودارها از rep={REP} ...")

    print("\n  شکل ۱: نمودارهای جداگانه هر fold (مثل شکل 13 اصلی)...")
    plot_original()

    print("\n  شکل ۲: همه فلدها روی هم (درخواست داور)...")
    plot_overlay()

    print("\nتمام! دو فایل ساخته شد:")
    print("  figure13_original.png  ← مثل شکل 13 اصلی مقاله")
    print("  figure13_overlay.png   ← درخواست داور (همه فلدها روی هم)")
