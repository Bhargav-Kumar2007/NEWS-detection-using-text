"""Shared helpers: config, logging, seeding, device detection, metrics and plots."""
from __future__ import annotations

import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Config / paths / logging
# --------------------------------------------------------------------------- #
def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to the project root (absolute paths are kept)."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    cfg_path = resolve_path(path or "config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Iterable[str]]) -> Dict[str, Any]:
    """Apply ``dotted.key=value`` overrides (values are parsed as YAML)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Use dotted.key=value")
        key, value = item.split("=", 1)
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(value)
    return cfg


def setup_logging(log_file: str | Path | None = None, level: int = logging.INFO) -> None:
    handlers: List[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto"):
    """Detect the best device. ``preference``: auto | cpu | cuda | mps."""
    import torch

    log = logging.getLogger(__name__)
    pref = (preference or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if pref.startswith("cuda") and not torch.cuda.is_available():
        log.warning("CUDA requested but not available - falling back to CPU.")
        return torch.device("cpu")
    if pref == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        log.warning("MPS requested but not available - falling back to CPU.")
        return torch.device("cpu")
    return torch.device(pref)


def describe_device(device) -> str:
    import torch

    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        return f"cuda:{idx} ({props.name}, {props.total_memory / 1024 ** 3:.1f} GB)"
    return str(device)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Accuracy, precision, recall, F1 (positive class = label 1), macro scores and ROC-AUC."""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, roc_auc_score)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "roc_auc": auc,
        "precision_macro": float(pm),
        "recall_macro": float(rm),
        "f1_macro": float(fm),
        "threshold": float(threshold),
        "num_samples": int(len(y_true)),
    }


# --------------------------------------------------------------------------- #
# Plots (matplotlib only, headless-safe)
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_training_history(history: Dict[str, Sequence[float]], out_dir: str | Path) -> None:
    """Save accuracy_curve.png and loss_curve.png (train vs validation)."""
    plt = _plt()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    if not epochs:
        return
    for key, title, fname in (("loss", "Loss", "loss_curve.png"),
                              ("accuracy", "Accuracy", "accuracy_curve.png")):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, history[f"train_{key}"], marker="o", label=f"Train {key}")
        ax.plot(epochs, history[f"val_{key}"], marker="s", label=f"Validation {key}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"Training vs validation {key}")
        ax.grid(alpha=0.3)
        ax.legend()
        if len(epochs) <= 20:
            ax.set_xticks(epochs)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, class_names: Sequence[str], path: str | Path) -> None:
    plt = _plt()
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, title, fmt in ((axes[0], cm, "Confusion matrix (counts)", "d"),
                                 (axes[1], cm_norm, "Confusion matrix (row-normalised)", ".2%")):
        im = ax.imshow(data, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        thresh = data.max() / 2.0 if data.max() > 0 else 0.5
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                text = format(int(val), "d") if fmt == "d" else format(val, fmt)
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if val > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, path: str | Path) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve

    plt = _plt()
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    ax.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
