"""
run_repeated_cv.py
==================
اجرا (همه چیز):        python run_repeated_cv.py -dp /path/to/dataset
فقط repeated CV:       python run_repeated_cv.py -dp /path/to/dataset --cv-only
فقط ablation study:    python run_repeated_cv.py -dp /path/to/dataset --ablation-only
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from pathlib import Path
from scipy import stats
from copy import deepcopy

# ── بهینه سازی برای Linux multi-GPU ────────────────────────────────────────
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
torch.backends.cudnn.enabled       = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
# ───────────────────────────────────────────────────────────────────────────

# ── parse --cv-only و --ablation-only قبل از process_args ──────────────────
# این لازمه چون process_args از sys.argv می‌خونه
_argv_backup = sys.argv[:]
sys.argv = [a for a in sys.argv if a not in ('--cv-only', '--ablation-only')]
from training.argprocess import process_args
opt = process_args()
sys.argv = _argv_backup
# ───────────────────────────────────────────────────────────────────────────

from datasets.MSDataset import MSDataset
from datasets.datautils import get_transforms
from training.trainer import load_train_objs, compute_performance, dice, train as trainer_train

# ─── CONFIG ─────────────────────────────────────────────────────────────────
N_CV_REPS       = 10
N_ABLATION_REPS = 10
N_FOLDS         = 5
SEEDS = [42, 123, 256, 512, 1024, 2048, 3141, 9999, 7777, 1111]
METRIC_KEYS = ['dice', 'sensitivity', 'specificity', 'ppv', 'npv', 'accuracy', 'f1']

ABLATION_CONFIGS = [
    {"name": "Baseline U-Net",     "use_sa": False, "csp": False, "gcnn": False, "dense": False, "tag": "baseline"},
    {"name": "+DenseNet",          "use_sa": False, "csp": False, "gcnn": False, "dense": True,  "tag": "dense"},
    {"name": "+DenseNet +CSP",     "use_sa": False, "csp": True,  "gcnn": False, "dense": True,  "tag": "dense_csp"},
    {"name": "+DenseNet +CSP +SA", "use_sa": True,  "csp": True,  "gcnn": False, "dense": True,  "tag": "dense_csp_sa"},
    {"name": "Full IbnSinaNet",    "use_sa": True,  "csp": True,  "gcnn": True,  "dense": True,  "tag": "full"},
]

DATASET_PATH = Path(opt.dataset_path)


# ═══════════════════════════════════════════════════════════════════════════
# توابع پایه
# ═══════════════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_ablation_opt(base_opt, config):
    o = deepcopy(base_opt)
    o.use_sa        = config["use_sa"]
    o.ablation_csp  = config["csp"]
    o.ablation_gcnn = config["gcnn"]
    if not config["dense"]:
        o.down_blocks = [1, 1, 1, 1, 1]
        o.up_blocks   = [1, 1, 1, 1]
        o.down_layers = [1, 1, 1, 1, 1]
        o.up_layers   = [1, 1, 1, 1]
    return o


# ═══════════════════════════════════════════════════════════════════════════
# train و test یک fold
# ═══════════════════════════════════════════════════════════════════════════

def train_fold(fold, seed, rep, tag="", cur_opt=None):
    if cur_opt is None:
        cur_opt = opt
    set_seed(seed)

    train_tf, eval_tf = get_transforms(cur_opt.input_dim)

    train_dset = MSDataset(DATASET_PATH, f'train{fold}',
        input_dim=cur_opt.input_dim, mean=cur_opt.mean, std=cur_opt.std,
        joint_transform=train_tf, seq_size=cur_opt.seq_size)

    val_dset = MSDataset(DATASET_PATH, f'val{fold}',
        input_dim=cur_opt.input_dim, mean=cur_opt.mean, std=cur_opt.std,
        joint_transform=eval_tf, seq_size=cur_opt.seq_size)

    prefix = f"{tag}-" if tag else ""
    os.environ['CKPT_TAG'] = f"{prefix}rep{rep:02d}-fold{fold}"

    base_dir   = f'./results_ms/{tag}/rep{rep:02d}' if tag else f'./results_ms/rep{rep:02d}'
    writer_dir = os.path.join(base_dir, f'fold{fold}', 'tb')
    image_dir  = os.path.join(base_dir, f'fold{fold}', 'imgs')
    os.makedirs(image_dir, exist_ok=True)

    trainer_train(cur_opt, fold, train_dset, val_dset, writer_dir, image_dir)


def test_fold(fold, rep, seed, tag="", cur_opt=None):
    if cur_opt is None:
        cur_opt = opt
    set_seed(seed)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, eval_tf = get_transforms(cur_opt.input_dim)

    test_dset = MSDataset(DATASET_PATH, f'test{fold}',
        input_dim=cur_opt.input_dim, mean=cur_opt.mean, std=cur_opt.std,
        joint_transform=eval_tf, seq_size=cur_opt.seq_size)

    loader = torch.utils.data.DataLoader(
        test_dset, batch_size=cur_opt.batch_size,
        shuffle=False, num_workers=4, pin_memory=True)

    model, _, _ = load_train_objs(cur_opt)
    prefix    = f"{tag}-" if tag else ""
    ckpt_path = f'./model-{prefix}rep{rep:02d}-fold{fold}.pt'
    model.load_state_dict(torch.load(ckpt_path, map_location=dev))
    model = model.to(dev).eval()

    criterion = nn.CrossEntropyLoss()
    val_loss = tp = fp = fn = tn = acc = 0

    with torch.no_grad():
        for source, targets in loader:
            source  = source.to(dev)
            targets = targets[:, 1, :, :].to(dev)
            net_out = model(source)
            loss    = criterion(net_out, targets.long())
            outputs = F.softmax(net_out, dim=1)[:, 1, :, :]
            preds   = (outputs > 0.5).float()
            val_loss += loss.item()
            t1, _, t2, _, t3, _, t4, _ = compute_performance(preds, targets)
            tp += t1; fp += t2; fn += t3; tn += t4
            acc += torch.mean((targets == preds).float()).item()

    n        = len(loader)
    val_loss /= n
    acc      /= n
    dv   = dice(tp, fp, fn)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1   = (2 * ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0.0

    return dict(loss=val_loss, dice=dv, sensitivity=sens, specificity=spec,
                ppv=ppv, npv=npv, accuracy=acc, f1=f1,
                fold=fold, rep=rep, seed=seed)


# ═══════════════════════════════════════════════════════════════════════════
# آمار
# ═══════════════════════════════════════════════════════════════════════════

def compute_stats(results):
    n      = len(results)
    t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))
    out    = {}
    for k in METRIC_KEYS:
        vals = np.array([r[k] for r in results])
        m    = float(vals.mean())
        s    = float(vals.std(ddof=1)) if n > 1 else 0.0
        se   = s / np.sqrt(n) if n > 1 else 0.0
        cv   = (s / m * 100) if m != 0 else 0.0
        out[k] = dict(
            mean   = round(m, 4),
            std    = round(s, 4),
            ci_lo  = round(m - t_crit * se, 4),
            ci_hi  = round(m + t_crit * se, 4),
            cv_pct = round(cv, 2),
        )
    return out


# ═══════════════════════════════════════════════════════════════════════════
# چاپ
# ═══════════════════════════════════════════════════════════════════════════

def sep(title=""):
    line = "=" * 72
    print(f"\n{line}")
    if title:
        print(f"  {title}")
    print(line)


def print_stats(summary, title=""):
    if title:
        print(f"\n  {title}")
    print(f"  {'Metric':<14} {'Mean%':>8} {'Std%':>7} {'CV%':>6}  {'95% CI':>28}")
    print("  " + "-" * 68)
    for k in METRIC_KEYS:
        s  = summary[k]
        ci = f"[{s['ci_lo']*100:.2f}%, {s['ci_hi']*100:.2f}%]"
        print(f"  {k:<14} {s['mean']*100:>7.2f}%  {s['std']*100:>6.2f}%"
              f"  {s['cv_pct']:>5.2f}%  {ci}")


def print_fold_table(fold_stats):
    cols = ['dice', 'sensitivity', 'ppv', 'npv', 'accuracy']
    print(f"  {'Fold':<6}" + "".join(f"  {c[:5]:>15}" for c in cols))
    print("  " + "-" * 86)
    for fold, s in fold_stats.items():
        row = f"  {fold:<6}"
        for c in cols:
            row += f"  {s[c]['mean']*100:.1f}+-{s[c]['std']*100:.1f}%"
        print(row)


def check_cv(summary):
    print("\n  CV Stability (threshold: CV < 10%):")
    for k in ['dice', 'sensitivity', 'ppv']:
        cv = summary[k]['cv_pct']
        print(f"    {k:<14}: {cv:.2f}%  {'OK' if cv < 10 else 'HIGH'}")


# ═══════════════════════════════════════════════════════════════════════════
# Repeated CV
# ═══════════════════════════════════════════════════════════════════════════

def run_repeated_cv():
    sep(f"REPEATED 5-FOLD CV  ({N_CV_REPS} reps x {N_FOLDS} folds = {N_CV_REPS*N_FOLDS} evals)")
    all_results = []

    for rep in range(1, N_CV_REPS + 1):
        seed = SEEDS[rep - 1]
        print(f"\n  Rep {rep:>2}/{N_CV_REPS}  seed={seed}")
        for fold in range(1, N_FOLDS + 1):
            print(f"    fold={fold}  train...", end=' ', flush=True)
            train_fold(fold=fold, seed=seed, rep=rep)
            print("test...", end=' ', flush=True)
            m = test_fold(fold=fold, rep=rep, seed=seed)
            all_results.append(m)
            print(f"Dice={m['dice']*100:.2f}%  Sens={m['sensitivity']*100:.2f}%")

    overall    = compute_stats(all_results)
    fold_stats = {fold: compute_stats([r for r in all_results if r['fold'] == fold])
                  for fold in range(1, N_FOLDS + 1)}

    sep("REPEATED CV — OVERALL RESULTS")
    print_stats(overall)
    check_cv(overall)

    sep("REPEATED CV — PER-FOLD")
    print_fold_table(fold_stats)

    return dict(all_results=all_results, overall_summary=overall,
                per_fold_summary={str(k): v for k, v in fold_stats.items()})


# ═══════════════════════════════════════════════════════════════════════════
# Ablation Study
# ═══════════════════════════════════════════════════════════════════════════

def run_ablation_study():
    total = N_ABLATION_REPS * N_FOLDS
    sep(f"ABLATION STUDY  ({N_ABLATION_REPS} reps x {N_FOLDS} folds = {total} evals PER CONFIG)")
    ablation_results = {}
    seeds_used = SEEDS[:N_ABLATION_REPS]

    for config in ABLATION_CONFIGS:
        name = config["name"]
        tag  = config["tag"]
        sep(f"Config: {name}")

        cur_opt        = make_ablation_opt(opt, config)
        config_results = []

        for rep_idx, seed in enumerate(seeds_used, 1):
            print(f"\n    rep={rep_idx}/{N_ABLATION_REPS}  seed={seed}")
            for fold in range(1, N_FOLDS + 1):
                print(f"      fold={fold}  train...", end=' ', flush=True)
                train_fold(fold=fold, seed=seed, rep=rep_idx, tag=tag, cur_opt=cur_opt)
                print("test...", end=' ', flush=True)
                m = test_fold(fold=fold, rep=rep_idx, seed=seed, tag=tag, cur_opt=cur_opt)
                config_results.append(m)
                print(f"Dice={m['dice']*100:.2f}%")

        summary = compute_stats(config_results)
        ablation_results[tag] = dict(name=name, summary=summary, results=config_results)
        d = summary['dice']
        print(f"\n  {name}: Dice={d['mean']*100:.2f}% +- {d['std']*100:.2f}%  CV={d['cv_pct']:.2f}%")

    sep("ABLATION — COMPARISON TABLE")
    cols = ['dice', 'sensitivity', 'specificity', 'ppv', 'npv', 'accuracy']
    header = f"  {'Config':<32}" + "".join(f"  {c[:5]:>14}" for c in cols) + "  Delta Dice"
    print(header)
    print("  " + "-" * 118)

    prev = None
    for config in ABLATION_CONFIGS:
        tag = config["tag"]
        if tag not in ablation_results:
            continue
        s     = ablation_results[tag]["summary"]
        row   = f"  {config['name']:<32}"
        for c in cols:
            row += f"  {s[c]['mean']*100:.2f}+-{s[c]['std']*100:.2f}%"
        curr  = s['dice']['mean'] * 100
        delta = f"  {curr-prev:+.2f}%" if prev is not None else "  —"
        row  += delta
        if tag == "full":
            print("  " + "-" * 118)
        print(row)
        prev = curr

    return ablation_results


# ═══════════════════════════════════════════════════════════════════════════
# ذخیره نتایج
# ═══════════════════════════════════════════════════════════════════════════

def save_results(cv=None, abl=None):
    output = {}
    if cv:
        output["repeated_cv"] = {
            "n_reps":         N_CV_REPS,
            "n_folds":        N_FOLDS,
            "total_evals":    N_CV_REPS * N_FOLDS,
            "overall_summary":  cv["overall_summary"],
            "per_fold_summary": cv["per_fold_summary"],
            "all_results":      cv["all_results"],
        }
    if abl:
        output["ablation"] = {
            tag: {
                "name":        d["name"],
                "summary":     d["summary"],
                "n_reps":      N_ABLATION_REPS,
                "total_evals": N_ABLATION_REPS * N_FOLDS,
                "results":     d["results"],
            }
            for tag, d in abl.items()
        }
    with open('all_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    print("\n  Saved → all_results.json")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ablation-only', action='store_true')
    parser.add_argument('--cv-only',       action='store_true')
    args, _ = parser.parse_known_args()

    cv_results  = None
    abl_results = None

    if not args.ablation_only:
        cv_results = run_repeated_cv()

    if not args.cv_only:
        abl_results = run_ablation_study()

    save_results(cv_results, abl_results)

    sep("FINAL SUMMARY")
    if cv_results:
        d = cv_results["overall_summary"]["dice"]
        s = cv_results["overall_summary"]["sensitivity"]
        print(f"\n  Repeated CV ({N_CV_REPS*N_FOLDS} evals):")
        print(f"    Dice        = {d['mean']*100:.2f}% +- {d['std']*100:.2f}%"
              f"  CV={d['cv_pct']:.2f}%"
              f"  CI=[{d['ci_lo']*100:.2f}%, {d['ci_hi']*100:.2f}%]")
        print(f"    Sensitivity = {s['mean']*100:.2f}% +- {s['std']*100:.2f}%"
              f"  CV={s['cv_pct']:.2f}%")
    if abl_results:
        print(f"\n  Ablation ({N_ABLATION_REPS} rep per config):")
        prev = None
        for config in ABLATION_CONFIGS:
            tag = config["tag"]
            if tag not in abl_results:
                continue
            d    = abl_results[tag]["summary"]["dice"]
            curr = d["mean"] * 100
            delta = f"  (Δ={curr-prev:+.2f}%)" if prev is not None else ""
            print(f"    {config['name']:<32}: {curr:.2f}% +- {d['std']*100:.2f}%"
                  f"  CV={d['cv_pct']:.2f}%{delta}")
            prev = curr
