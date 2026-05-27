"""
save_results.py
===============
این فایل رو به test.py اضافه کن یا جداگانه ران کن.

برای هر slice یه تصویر ترکیبی می‌سازه با 5 ستون:
  [Input MRI] [Ground Truth] [Prediction] [FP/FN Map] [Overlay]

ساختار پوشه خروجی:
  results_ms/
    test_images/
      fold1/
        combined/     ← تصویر ترکیبی 5 ستونی
        input/        ← تصویر MRI خام
        target/       ← ماسک واقعی
        predict/      ← ماسک پیش‌بینی شده
        errors/       ← نقشه خطاها (FP قرمز، FN سبز)
        overlay/      ← پیش‌بینی روی MRI
"""

import os
import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from datasets.MSDataset import MSDataset
from datasets.datautils import get_transforms
from training.argprocess import process_args
from training.trainer import load_train_objs, compute_performance, dice
from utils.general_utils import save_tensor_batch

opt         = process_args()
dev         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_PATH = Path(opt.dataset_path)
RESULTS_DIR  = './results_ms/test_images'


def tensor_to_numpy_img(tensor_1chw):
    """تنسور یک کانالی رو به numpy uint8 تبدیل کن"""
    arr = tensor_1chw.squeeze().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def make_rgb(gray_np):
    """تصویر grayscale رو RGB کن"""
    return np.stack([gray_np, gray_np, gray_np], axis=-1)


def make_error_map(fp_map, fn_map):
    """
    نقشه خطا:
    FP = قرمز (مدل گفت ضایعه ولی نبود)
    FN = سبز  (مدل نگفت ضایعه ولی بود)
    """
    h, w = fp_map.shape
    rgb  = np.zeros((h, w, 3), dtype=np.uint8)
    fp   = fp_map.cpu().numpy().astype(bool)
    fn   = fn_map.cpu().numpy().astype(bool)
    rgb[fp] = [220, 50,  50 ]   # قرمز = FP
    rgb[fn] = [50,  180, 50 ]   # سبز  = FN
    return rgb


def make_overlay(mri_np, pred_np, target_np):
    """
    پیش‌بینی رو روی MRI overlay کن:
    TP = آبی روشن
    FP = قرمز
    FN = سبز
    """
    rgb = make_rgb(mri_np)
    pred   = pred_np.astype(bool)
    target = target_np.astype(bool)
    tp = pred & target
    fp = pred & ~target
    fn = ~pred & target
    rgb[tp] = [100, 149, 237]   # آبی = TP
    rgb[fp] = [220, 50,  50 ]   # قرمز = FP
    rgb[fn] = [50,  180, 50 ]   # سبز = FN
    return rgb


def add_label(img_np, label, font_size=14):
    """یه label به بالای تصویر اضافه کن"""
    pil = Image.fromarray(img_np if img_np.ndim == 3
                          else make_rgb(img_np))
    draw = ImageDraw.Draw(pil)
    draw.text((4, 2), label, fill=(255, 255, 0))
    return np.array(pil)


def make_combined_image(mri, target, pred, fp_map, fn_map, img_size=160):
    """
    5 تصویر رو کنار هم بذار در یه تصویر افقی
    """
    mri_np    = tensor_to_numpy_img(mri)
    target_np = tensor_to_numpy_img(target)
    pred_np   = tensor_to_numpy_img(pred)

    panels = [
        add_label(make_rgb(mri_np),    "Input MRI"),
        add_label(make_rgb(target_np), "Ground Truth"),
        add_label(make_rgb(pred_np),   "Prediction"),
        add_label(make_error_map(fp_map, fn_map), "FP=Red FN=Green"),
        add_label(make_overlay(mri_np, pred_np, target_np), "Overlay"),
    ]

    # اضافه کردن خط جداکننده سفید بین پانل‌ها
    sep = np.ones((img_size, 2, 3), dtype=np.uint8) * 200
    combined = panels[0]
    for p in panels[1:]:
        combined = np.concatenate([combined, sep, p], axis=1)

    return Image.fromarray(combined)


def save_fold_images(fold, rep=None):
    """
    همه تصاویر تست یه fold رو ذخیره کن
    """
    print(f"\n  Fold {fold}: بارگذاری مدل ...")

    _, eval_tf = get_transforms(opt.input_dim)
    test_dset  = MSDataset(DATASET_PATH, f'test{fold}',
                           input_dim=opt.input_dim, mean=opt.mean,
                           std=opt.std, joint_transform=eval_tf,
                           seq_size=opt.seq_size)
    test_loader = torch.utils.data.DataLoader(
        test_dset, batch_size=1, shuffle=False)

    model, _, _ = load_train_objs(opt)

    # انتخاب checkpoint
    if rep is not None:
        ckpt = f'./model-rep{rep:02d}-fold{fold}.pt'
    else:
        ckpt = f'./model-fold{fold}.pt'

    if not os.path.exists(ckpt):
        print(f"    checkpoint پیدا نشد: {ckpt}")
        return

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    model = model.to(dev).eval()
    criterion = nn.CrossEntropyLoss()

    # ساختن پوشه‌ها
    fold_dir     = os.path.join(RESULTS_DIR, f'fold{fold}')
    combined_dir = os.path.join(fold_dir, 'combined')
    input_dir    = os.path.join(fold_dir, 'input')
    target_dir   = os.path.join(fold_dir, 'target')
    predict_dir  = os.path.join(fold_dir, 'predict')
    error_dir    = os.path.join(fold_dir, 'errors')
    overlay_dir  = os.path.join(fold_dir, 'overlay')

    for d in [combined_dir, input_dir, target_dir,
              predict_dir, error_dir, overlay_dir]:
        os.makedirs(d, exist_ok=True)

    total_tp = total_fp = total_fn = total_tn = 0
    val_loss = 0

    for idx, (source, targets) in enumerate(test_loader):
        source  = source.to(dev)
        targets = targets[:, 1, :, :].to(dev)

        with torch.no_grad():
            net_out = model(source)
            loss    = criterion(net_out, targets.long())

        outputs = F.softmax(net_out, dim=1)[:, 1, :, :]
        preds   = (outputs > 0.5).float()
        val_loss += loss.item()

        tmp_tp, tp_map, tmp_fp, fp_map, tmp_fn, fn_map, tmp_tn, tn_map = \
            compute_performance(preds, targets)
        total_tp += tmp_tp; total_fp += tmp_fp
        total_fn += tmp_fn; total_tn += tmp_tn

        # ذخیره هر تصویر جداگانه
        mri    = source[0, 1, :, :].unsqueeze(0).unsqueeze(0)
        target = targets[0].unsqueeze(0).unsqueeze(0)
        pred   = preds[0].unsqueeze(0).unsqueeze(0)

        # input
        mri_arr = tensor_to_numpy_img(mri)
        mri_arr = (mri_arr * opt.std + opt.mean * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(mri_arr).save(
            os.path.join(input_dir, f'{idx:04d}.png'))

        # target
        Image.fromarray(tensor_to_numpy_img(target)).save(
            os.path.join(target_dir, f'{idx:04d}.png'))

        # prediction
        Image.fromarray(tensor_to_numpy_img(pred)).save(
            os.path.join(predict_dir, f'{idx:04d}.png'))

        # error map (FP قرمز، FN سبز)
        err_img = make_error_map(fp_map[0], fn_map[0])
        Image.fromarray(err_img).save(
            os.path.join(error_dir, f'{idx:04d}.png'))

        # overlay
        mri_np  = tensor_to_numpy_img(mri)
        pred_np = tensor_to_numpy_img(pred)
        tgt_np  = tensor_to_numpy_img(target)
        ov_img  = make_overlay(mri_np, pred_np, tgt_np)
        Image.fromarray(ov_img).save(
            os.path.join(overlay_dir, f'{idx:04d}.png'))

        # تصویر ترکیبی 5 ستونی
        combined = make_combined_image(
            mri, target, pred, fp_map[0], fn_map[0])
        combined.save(os.path.join(combined_dir, f'{idx:04d}.png'))

        if idx % 20 == 0:
            print(f"    {idx}/{len(test_loader)} اسلایس پردازش شد ...")

    # نتایج
    val_loss /= len(test_loader)
    test_dice = dice(total_tp, total_fp, total_fn)
    sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    spec = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
    ppv  = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

    print(f"  Fold {fold}: Dice={test_dice*100:.2f}%  "
          f"Sens={sens*100:.2f}%  Spec={spec*100:.2f}%  PPV={ppv*100:.2f}%")
    print(f"  تصاویر ذخیره شد در: {fold_dir}")
    print(f"    combined/  ← تصویر ۵ ستونی (input+GT+pred+errors+overlay)")
    print(f"    input/     ← MRI خام")
    print(f"    target/    ← ماسک واقعی")
    print(f"    predict/   ← ماسک پیش‌بینی")
    print(f"    errors/    ← FP قرمز، FN سبز")
    print(f"    overlay/   ← پیش‌بینی روی MRI")


if __name__ == '__main__':
    print("\nذخیره تصاویر تست برای همه فلدها ...")
    print("رنگ‌بندی خطاها: FP=قرمز  FN=سبز  TP=آبی")

    for fold in range(1, 6):
        save_fold_images(fold=fold, rep=1)

    print("\nتمام!")
    print(f"همه تصاویر در: {RESULTS_DIR}")
