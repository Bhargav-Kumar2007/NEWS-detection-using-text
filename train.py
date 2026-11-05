"""Train an Embedding+BiLSTM or Embedding+TextCNN fake-news classifier on Fakeddit.

Examples:
    python train.py --mode small                 # quick run on a stratified subset
    python train.py --mode full --model textcnn  # whole official splits
    python train.py --mode full --resume         # continue an interrupted run
    python train.py --set training.epochs=5 training.batch_size=128

Everything is written to outputs/<model>_<mode>/ (see README for the layout).
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import yaml

from engine import predict_loader, train_one_epoch
from evaluate import evaluate_texts
from models import build_model, count_parameters, load_model_from_checkpoint, model_config_from_cfg
from preprocessing import Tokenizer, load_splits, make_loader, tokenizer_from_config
from utils import (apply_overrides, compute_metrics, describe_device, get_device, load_config,
                   plot_training_history, resolve_path, save_json, set_seed, setup_logging)

log = logging.getLogger("train")

HISTORY_KEYS = ["train_loss", "train_accuracy", "val_loss", "val_accuracy", "val_precision",
                "val_recall", "val_f1", "val_roc_auc", "lr", "epoch_seconds"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train a Fakeddit fake-news classifier.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["small", "full"], help="small = stratified subset, full = whole splits")
    ap.add_argument("--model", choices=["bilstm", "textcnn"], help="architecture (overrides config)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--device", help="auto | cpu | cuda | mps")
    ap.add_argument("--run-name", help="output folder name (default: <model>_<mode>)")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/last.pt")
    ap.add_argument("--skip-test", action="store_true", help="do not evaluate on the test split at the end")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", help="config overrides, e.g. model.embed_dim=300")
    return ap.parse_args()


def monitor_score(monitor: str, val_loss: float, val_metrics: Dict[str, float]) -> float:
    """Higher is always better."""
    if monitor == "val_loss":
        return -val_loss
    key = monitor.replace("val_", "")
    if key not in val_metrics:
        raise ValueError(f"Unknown training.monitor '{monitor}'")
    value = val_metrics[key]
    return -1e9 if value != value else value  # NaN-safe


def make_checkpoint(model, optimizer, scheduler, epoch, history, best_score, no_improve,
                    cfg, mode, val_metrics) -> Dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "model_config": model.config,
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "history": history,
        "best_score": best_score,
        "epochs_no_improve": no_improve,
        "mode": mode,
        "max_len": cfg["preprocessing"]["max_len"],
        "threshold": cfg["labels"]["threshold"],
        "class_names": cfg["labels"]["class_names"],
        "text_column": cfg["data"]["text_column"],
        "label_column": cfg["data"]["label_column"],
        "val_metrics": val_metrics,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["type"] = args.model
    if args.mode:
        cfg["data"]["mode"] = args.mode
    if args.epochs:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.device:
        cfg["training"]["device"] = args.device
    apply_overrides(cfg, args.set)

    mode, tcfg = cfg["data"]["mode"], cfg["training"]
    run_name = args.run_name or f"{cfg['model']['type']}_{mode}"
    run_dir = resolve_path(cfg["paths"]["output_dir"]) / run_name
    ckpt_dir = run_dir / "checkpoints"
    plots_dir = run_dir / "plots"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "train.log")
    with open(run_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    set_seed(cfg["seed"])
    device = get_device(tcfg.get("device", "auto"))
    use_amp = bool(tcfg.get("use_amp", False)) and device.type == "cuda"
    log.info("Run: %s | mode=%s | model=%s", run_name, mode, cfg["model"]["type"])
    log.info("Device: %s | mixed precision: %s", describe_device(device), use_amp)

    # ------------------------------------------------------------------ data
    splits = load_splits(cfg, mode)
    class_names = cfg["labels"]["class_names"]
    last_path = ckpt_dir / "last.pt"
    resume = args.resume and last_path.exists()
    if args.resume and not resume:
        log.warning("--resume given but %s does not exist; starting from scratch.", last_path)

    tok_path = run_dir / "tokenizer.json"
    if resume and tok_path.exists():
        tokenizer = Tokenizer.load(tok_path)
        log.info("Loaded tokenizer from %s (vocab=%d)", tok_path, tokenizer.vocab_size)
    else:
        tokenizer = tokenizer_from_config(cfg).fit(splits["train"][0])  # vocabulary from TRAIN only
        tokenizer.save(tok_path)
        log.info("Tokenizer saved to %s", tok_path)

    loaders = {}
    for name in ("train", "validation"):
        texts, labels = splits[name]
        ids, lengths = tokenizer.encode_batch(texts, show_progress=True)
        loaders[name] = make_loader(ids, lengths, labels, tcfg["batch_size"], shuffle=(name == "train"),
                                    num_workers=tcfg.get("num_workers", 0), pin_memory=device.type == "cuda")

    # ----------------------------------------------------------------- model
    if resume:
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model = build_model(ckpt["model_config"])
        model.load_state_dict(ckpt["model_state"])
    else:
        ckpt = None
        model = build_model(model_config_from_cfg(cfg, tokenizer.vocab_size))
    model.to(device)
    log.info("Model: %s | trainable parameters: %s", model.config["type"], f"{count_parameters(model):,}")

    train_labels = splits["train"][1]
    pos_weight = None
    if tcfg.get("use_pos_weight", False):
        n_pos = max(int(train_labels.sum()), 1)
        n_neg = max(int(len(train_labels) - n_pos), 1)
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
        log.info("Using pos_weight=%.3f", pos_weight.item())
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg["learning_rate"],
                                  weight_decay=tcfg.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=tcfg.get("lr_scheduler_factor", 0.5),
        patience=tcfg.get("lr_scheduler_patience", 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: Dict[str, list] = {k: [] for k in HISTORY_KEYS}
    start_epoch, best_score, no_improve, best_epoch = 1, -float("inf"), 0, 0
    if ckpt is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        best_score, no_improve = ckpt["best_score"], ckpt["epochs_no_improve"]
        best_epoch = int(ckpt.get("best_epoch", ckpt["epoch"]))
        log.info("Resumed from epoch %d (best %s so far: %.4f)", ckpt["epoch"], tcfg["monitor"], best_score)

    # ------------------------------------------------------------- training
    epochs, patience = tcfg["epochs"], tcfg.get("early_stopping_patience", 0)
    thr = cfg["labels"]["threshold"]
    t_start = time.time()
    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, loaders["train"], criterion, optimizer, scaler, device, use_amp,
                             tcfg.get("grad_clip"), desc=f"epoch {epoch}/{epochs} train")
        va = predict_loader(model, loaders["validation"], device, criterion, use_amp,
                            desc=f"epoch {epoch}/{epochs} val")
        vm = compute_metrics(va["labels"], va["probs"], thr)
        scheduler.step(va["loss"])

        row = {"train_loss": tr["loss"], "train_accuracy": tr["accuracy"], "val_loss": va["loss"],
               "val_accuracy": vm["accuracy"], "val_precision": vm["precision"], "val_recall": vm["recall"],
               "val_f1": vm["f1"], "val_roc_auc": vm["roc_auc"], "lr": optimizer.param_groups[0]["lr"],
               "epoch_seconds": time.time() - t0}
        for k, v in row.items():
            history[k].append(float(v))
        log.info("Epoch %02d/%d | train loss %.4f acc %.4f | val loss %.4f acc %.4f P %.4f R %.4f "
                 "F1 %.4f AUC %.4f | %.0fs", epoch, epochs, tr["loss"], tr["accuracy"], va["loss"],
                 vm["accuracy"], vm["precision"], vm["recall"], vm["f1"], vm["roc_auc"], row["epoch_seconds"])

        score = monitor_score(tcfg["monitor"], va["loss"], vm)
        if score > best_score + tcfg.get("min_delta", 0.0):
            best_score, no_improve, best_epoch = score, 0, epoch
            ck = make_checkpoint(model, optimizer, scheduler, epoch, history, best_score, no_improve,
                                 cfg, mode, vm)
            ck.pop("optimizer_state"), ck.pop("scheduler_state")  # best_model.pt is inference-only
            torch.save(ck, ckpt_dir / "best_model.pt")
            log.info("  -> new best (%s=%.4f), saved checkpoints/best_model.pt", tcfg["monitor"], score)
        else:
            no_improve += 1

        ck_last = make_checkpoint(model, optimizer, scheduler, epoch, history, best_score, no_improve,
                                  cfg, mode, vm)
        ck_last["best_epoch"] = best_epoch
        torch.save(ck_last, last_path)
        if tcfg.get("save_every_epoch", False):
            torch.save(ck_last, ckpt_dir / f"epoch_{epoch:02d}.pt")
        save_json(history, run_dir / "history.json")
        plot_training_history(history, plots_dir)

        if patience and no_improve >= patience:
            log.info("Early stopping: no improvement in %s for %d epochs.", tcfg["monitor"], patience)
            break

    train_seconds = time.time() - t_start
    if not (ckpt_dir / "best_model.pt").exists():
        log.error("No best checkpoint was written (no epochs ran?). Nothing to evaluate.")
        return

    # --------------------------------------------------- final evaluation
    best_model, best_ckpt = load_model_from_checkpoint(ckpt_dir / "best_model.pt", device)
    summary: Dict[str, Any] = {
        "run_name": run_name, "mode": mode, "model_type": best_model.config["type"],
        "device": describe_device(device), "best_epoch": best_ckpt["epoch"],
        "monitor": tcfg["monitor"], "epochs_run": len(history["train_loss"]),
        "train_seconds_this_session": train_seconds, "vocab_size": tokenizer.vocab_size,
        "trainable_parameters": count_parameters(best_model),
        "num_samples": {k: len(v[1]) for k, v in splits.items()},
    }
    common = dict(class_names=class_names, threshold=thr, batch_size=cfg["inference"]["batch_size"],
                  use_amp=use_amp, num_workers=tcfg.get("num_workers", 0),
                  extra={"run_dir": str(run_dir), "mode": mode, "checkpoint_epoch": best_ckpt["epoch"],
                         "model_type": best_model.config["type"]})
    eval_splits = ["validation"] + ([] if args.skip_test else ["test"])
    for split in eval_splits:
        texts, labels = splits[split]
        res = evaluate_texts(best_model, tokenizer, texts, labels, device,
                             out_dir=run_dir / "evaluation" / f"{split}_{mode}", split_name=split, **common)
        summary[f"{split}_metrics"] = res["metrics"]
    save_json(summary, run_dir / "training_summary.json")
    log.info("Done. All artifacts are in %s", run_dir)


if __name__ == "__main__":
    main()
