"""
Lesion-level evaluation metrics for MS lesion segmentation.

Addresses reviewer comment:
    "Specificity values consistently above 99.9% are a clear artifact of extreme
     class imbalance. Clinically meaningful metrics such as lesion detection rate
     and false positives per scan must be reported instead."

Two core functions:
    - lesion_detection_rate  (LDR / sensitivity at the lesion level)
    - false_positives_per_scan (FPPS)

Both operate on connected components, not individual pixels.
"""

import torch
import numpy as np
from scipy import ndimage


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_numpy_binary(tensor: torch.Tensor) -> np.ndarray:
    """Convert a 2-D or 3-D (B, H, W) torch tensor to a binary numpy array."""
    arr = tensor.detach().cpu().numpy()
    return (arr > 0.5).astype(np.uint8)


def _connected_components(binary_map: np.ndarray):
    """
    Label connected components in a 2-D binary map.

    Returns
    -------
    labeled : np.ndarray   integer label map (0 = background)
    n_lesions : int        number of distinct lesions found
    """
    struct = ndimage.generate_binary_structure(2, 2)   # 8-connectivity
    labeled, n_lesions = ndimage.label(binary_map, structure=struct)
    return labeled, n_lesions


def _compute_lesion_stats_single(pred_2d: np.ndarray,
                                  gt_2d: np.ndarray,
                                  iou_threshold: float = 0.1):
    """
    Compute lesion-level TP, FP, FN for one 2-D slice / scan.

    A ground-truth lesion is considered *detected* (TP) when the predicted
    mask overlaps it by at least `iou_threshold` of the GT lesion's area.
    Each predicted component that does not overlap any GT lesion is an FP.

    Parameters
    ----------
    pred_2d : np.ndarray  (H, W) binary prediction
    gt_2d   : np.ndarray  (H, W) binary ground truth
    iou_threshold : float  overlap fraction needed to count as detected
                           (default 0.1 — standard in MS lesion literature)

    Returns
    -------
    n_tp, n_fp, n_fn : int
    """
    gt_labeled,   n_gt   = _connected_components(gt_2d)
    pred_labeled, n_pred = _connected_components(pred_2d)

    detected_gt   = set()
    matched_preds = set()

    for gt_id in range(1, n_gt + 1):
        gt_mask = (gt_labeled == gt_id)
        gt_area = gt_mask.sum()
        if gt_area == 0:
            continue

        # Check every predicted lesion that overlaps this GT lesion
        overlap_pred_ids = np.unique(pred_labeled[gt_mask])
        overlap_pred_ids = overlap_pred_ids[overlap_pred_ids > 0]

        for pred_id in overlap_pred_ids:
            pred_mask   = (pred_labeled == pred_id)
            intersection = (gt_mask & pred_mask).sum()
            # Overlap fraction relative to GT lesion area
            overlap_ratio = intersection / gt_area
            if overlap_ratio >= iou_threshold:
                detected_gt.add(gt_id)
                matched_preds.add(pred_id)
                break   # this GT lesion is already counted as detected

    n_tp = len(detected_gt)
    n_fn = n_gt  - n_tp
    n_fp = n_pred - len(matched_preds)   # predicted components with no GT match
    return n_tp, n_fp, n_fn


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def lesion_detection_rate(preds: torch.Tensor,
                          targets: torch.Tensor,
                          iou_threshold: float = 0.1) -> float:
    """
    Lesion Detection Rate (LDR) — lesion-level sensitivity.

    Fraction of ground-truth lesions that are detected by the model.
    This metric is robust to class imbalance because it counts *lesions*,
    not pixels.

    Parameters
    ----------
    preds   : (B, H, W) float tensor, values in {0, 1}
    targets : (B, H, W) float tensor, values in {0, 1}
    iou_threshold : float  (default 0.1)

    Returns
    -------
    ldr : float  in [0, 1]
    """
    preds_np   = _to_numpy_binary(preds)
    targets_np = _to_numpy_binary(targets)

    total_tp = 0
    total_fn = 0

    for b in range(preds_np.shape[0]):
        tp, fp, fn = _compute_lesion_stats_single(
            preds_np[b], targets_np[b], iou_threshold)
        total_tp += tp
        total_fn += fn

    if (total_tp + total_fn) == 0:
        return 1.0   # no GT lesions → perfect by convention

    return round(total_tp / (total_tp + total_fn), 5)


def false_positives_per_scan(preds: torch.Tensor,
                              targets: torch.Tensor,
                              iou_threshold: float = 0.1) -> float:
    """
    False Positives Per Scan (FPPS).

    Average number of spurious predicted lesions per scan (image in the batch).
    Lower is better. This is the standard clinical metric used alongside LDR
    in MS lesion detection challenges (e.g. MICCAI MSSEG).

    Parameters
    ----------
    preds   : (B, H, W) float tensor, values in {0, 1}
    targets : (B, H, W) float tensor, values in {0, 1}
    iou_threshold : float  (default 0.1)

    Returns
    -------
    fpps : float  ≥ 0
    """
    preds_np   = _to_numpy_binary(preds)
    targets_np = _to_numpy_binary(targets)

    total_fp = 0
    n_scans  = preds_np.shape[0]

    for b in range(n_scans):
        tp, fp, fn = _compute_lesion_stats_single(
            preds_np[b], targets_np[b], iou_threshold)
        total_fp += fp

    return round(total_fp / n_scans, 5)


def compute_lesion_metrics(preds: torch.Tensor,
                            targets: torch.Tensor,
                            iou_threshold: float = 0.1) -> dict:
    """
    Convenience wrapper — returns both LDR and FPPS in one dict.

    Usage example (inside your test loop):

        preds  = (F.softmax(net_out, dim=1)[:, 1, :, :] > 0.5).float()
        metrics = compute_lesion_metrics(preds, targets)
        print(f"LDR: {metrics['ldr']:.4f}  FPPS: {metrics['fpps']:.4f}")
    """
    preds_np   = _to_numpy_binary(preds)
    targets_np = _to_numpy_binary(targets)

    total_tp = total_fp = total_fn = 0
    n_scans = preds_np.shape[0]

    for b in range(n_scans):
        tp, fp, fn = _compute_lesion_stats_single(
            preds_np[b], targets_np[b], iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    ldr  = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 1.0
    fpps = total_fp / n_scans

    return {
        "ldr":          round(ldr,  5),   # Lesion Detection Rate
        "fpps":         round(fpps, 5),   # False Positives Per Scan
        "lesion_tp":    total_tp,
        "lesion_fp":    total_fp,
        "lesion_fn":    total_fn,
    }
