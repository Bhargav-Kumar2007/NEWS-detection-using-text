"""Training / inference loops shared by train.py, evaluate.py and predict.py."""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

import numpy as np
import torch
from tqdm.auto import tqdm


def autocast_ctx(device: torch.device, enabled: bool):
    """Mixed-precision context on CUDA; a no-op everywhere else."""
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp: bool = False,
                    grad_clip: Optional[float] = None, desc: str = "train") -> Dict[str, float]:
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for ids, lengths, labels in pbar:
        ids = ids.to(device, non_blocking=True).long()
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, use_amp):
            logits = model(ids, lengths)
        loss = criterion(logits.float(), labels)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        n = labels.size(0)
        total_loss += loss.item() * n
        correct += ((logits.detach() > 0).float() == labels).sum().item()
        seen += n
        pbar.set_postfix(loss=f"{total_loss / seen:.4f}", acc=f"{correct / seen:.4f}")
    return {"loss": total_loss / max(seen, 1), "accuracy": correct / max(seen, 1)}


@torch.no_grad()
def predict_loader(model, loader, device, criterion=None, use_amp: bool = False,
                   desc: str = "eval") -> Dict[str, Any]:
    """Run the model over a loader -> {'loss', 'probs' (P[label=1]), 'labels'}."""
    model.eval()
    probs, labels_all = [], []
    total_loss, seen = 0.0, 0
    for ids, lengths, labels in tqdm(loader, desc=desc, leave=False):
        ids = ids.to(device, non_blocking=True).long()
        labels_dev = labels.to(device, non_blocking=True)
        with autocast_ctx(device, use_amp):
            logits = model(ids, lengths)
        logits = logits.float()
        if criterion is not None:
            total_loss += criterion(logits, labels_dev).item() * labels.size(0)
        seen += labels.size(0)
        probs.append(torch.sigmoid(logits).cpu().numpy())
        labels_all.append(labels.numpy())
    return {
        "loss": total_loss / max(seen, 1) if criterion is not None else float("nan"),
        "probs": np.concatenate(probs) if probs else np.array([]),
        "labels": np.concatenate(labels_all).astype(int) if labels_all else np.array([], dtype=int),
    }


@torch.no_grad()
def forward_features(model, ids: np.ndarray, lengths: np.ndarray, device, batch_size: int = 512,
                     return_probs: bool = False):
    """Batched inference on pre-encoded arrays -> (features [N, D], probs [N]) as numpy."""
    model.eval()
    feats, probs = [], []
    for start in range(0, len(ids), batch_size):
        b_ids = torch.from_numpy(ids[start:start + batch_size]).to(device).long()
        b_len = torch.from_numpy(lengths[start:start + batch_size])
        f = model.extract_features(b_ids, b_len)
        feats.append(f.float().cpu().numpy())
        if return_probs:
            logits = model.out(f).squeeze(-1)
            probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    features = np.concatenate(feats) if feats else np.zeros((0, model.feature_dim), dtype=np.float32)
    if return_probs:
        return features, (np.concatenate(probs) if probs else np.array([]))
    return features
