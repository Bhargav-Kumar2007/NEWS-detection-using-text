"""Evaluate a trained run on the official validation / test split.

Usage:
    python evaluate.py --run-dir outputs/bilstm_small --split test
    python evaluate.py --run-dir outputs/bilstm_small --split test --full-split   # whole official split

Writes to <run_dir>/evaluation/<split>_<mode>/ :
    metrics.json, classification_report.txt, confusion_matrix.png / .csv,
    roc_curve.png, predictions.csv
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from engine import predict_loader
from models import load_model_from_checkpoint
from preprocessing import Tokenizer, load_splits, make_loader
from utils import (apply_overrides, compute_metrics, describe_device, get_device, load_config,
                   plot_confusion_matrix, plot_roc_curve, resolve_path, save_json, setup_logging)

log = logging.getLogger("evaluate")


def save_evaluation_artifacts(y_true: np.ndarray, y_prob: np.ndarray, texts: Sequence[str],
                              out_dir: str | Path, class_names: Sequence[str], threshold: float,
                              split_name: str, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute every metric from real predictions and write all evaluation files."""
    from sklearn.metrics import classification_report, confusion_matrix

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_true, y_prob, threshold)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    report_txt = classification_report(y_true, y_pred, labels=[0, 1], target_names=list(class_names),
                                       digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, labels=[0, 1], target_names=list(class_names),
                                        output_dict=True, zero_division=0)

    result = {"split": split_name, **(extra or {}), "metrics": metrics,
              "confusion_matrix": cm.tolist(), "class_names": list(class_names),
              "classification_report": report_dict}
    save_json(result, out_dir / "metrics.json")
    (out_dir / "classification_report.txt").write_text(report_txt, encoding="utf-8")
    pd.DataFrame(cm, index=[f"true_{c}" for c in class_names],
                 columns=[f"pred_{c}" for c in class_names]).to_csv(out_dir / "confusion_matrix.csv")
    plot_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, y_prob, out_dir / "roc_curve.png")
    pd.DataFrame({"text": list(texts), "true_label": y_true, "prob_label_1": y_prob,
                  "pred_label": y_pred}).to_csv(out_dir / "predictions.csv", index=False)

    log.info("=== %s results (threshold %.2f, n=%d) ===", split_name, threshold, metrics["num_samples"])
    log.info("Accuracy %.4f | Precision %.4f | Recall %.4f | F1 %.4f | ROC-AUC %s",
             metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"],
             "n/a" if metrics["roc_auc"] != metrics["roc_auc"] else f"{metrics['roc_auc']:.4f}")
    log.info("Classification report:\n%s", report_txt)
    log.info("Artifacts saved to %s", out_dir)
    return result


def evaluate_texts(model, tokenizer: Tokenizer, texts: List[str], labels: np.ndarray, device,
                   out_dir: str | Path, class_names: Sequence[str], threshold: float, split_name: str,
                   batch_size: int = 512, use_amp: bool = False, num_workers: int = 0,
                   extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ids, lengths = tokenizer.encode_batch(texts, show_progress=True)
    loader = make_loader(ids, lengths, labels, batch_size, shuffle=False, num_workers=num_workers,
                         pin_memory=device.type == "cuda")
    pred = predict_loader(model, loader, device, criterion=None, use_amp=use_amp, desc=f"eval[{split_name}]")
    return save_evaluation_artifacts(pred["labels"], pred["probs"], texts, out_dir, class_names,
                                     threshold, split_name, extra)


def evaluate_run(run_dir: str | Path, split: str = "test", device_pref: Optional[str] = None,
                 batch_size: Optional[int] = None, threshold: Optional[float] = None,
                 mode: Optional[str] = None, full_split: bool = False,
                 overrides: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    run_dir = resolve_path(run_dir)
    ckpt_path = run_dir / "checkpoints" / "best_model.pt"
    tok_path = run_dir / "tokenizer.json"
    for p in (ckpt_path, tok_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found - train first (python train.py) or check --run-dir.")

    saved_cfg = run_dir / "config_used.yaml"
    cfg = load_config(saved_cfg if saved_cfg.exists() else None)
    apply_overrides(cfg, overrides)

    device = get_device(device_pref or cfg["training"].get("device", "auto"))
    log.info("Device: %s", describe_device(device))
    model, ckpt = load_model_from_checkpoint(ckpt_path, device)
    tokenizer = Tokenizer.load(tok_path)

    mode = "full" if full_split else (mode or ckpt.get("mode") or cfg["data"]["mode"])
    texts, labels = load_splits(cfg, mode, only=[split])[split]
    class_names = ckpt.get("class_names") or cfg["labels"]["class_names"]
    thr = threshold if threshold is not None else ckpt.get("threshold", cfg["labels"]["threshold"])
    use_amp = bool(cfg["training"].get("use_amp", False)) and device.type == "cuda"

    return evaluate_texts(
        model, tokenizer, texts, labels, device,
        out_dir=run_dir / "evaluation" / f"{split}_{mode}",
        class_names=class_names, threshold=float(thr), split_name=split,
        batch_size=batch_size or cfg["training"]["batch_size"], use_amp=use_amp,
        num_workers=cfg["training"].get("num_workers", 0),
        extra={"run_dir": str(run_dir), "mode": mode, "checkpoint_epoch": ckpt.get("epoch"),
               "model_type": ckpt["model_config"]["type"]})


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a trained Fakeddit run.")
    ap.add_argument("--run-dir", required=True, help="e.g. outputs/bilstm_small")
    ap.add_argument("--split", default="test", choices=["train", "validation", "test"])
    ap.add_argument("--mode", choices=["small", "full"], help="subset size to evaluate on "
                    "(default: the mode the run was trained with)")
    ap.add_argument("--full-split", action="store_true", help="evaluate on the entire official split")
    ap.add_argument("--threshold", type=float, help="decision threshold (default: from checkpoint)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--device", help="auto | cpu | cuda | mps")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", help="config overrides")
    args = ap.parse_args()

    setup_logging()
    evaluate_run(args.run_dir, args.split, args.device, args.batch_size, args.threshold,
                 args.mode, args.full_split, args.set)


if __name__ == "__main__":
    main()
