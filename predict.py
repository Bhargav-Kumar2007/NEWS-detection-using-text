"""Reusable prediction + feature extraction for arbitrary input strings.

Python usage:
    from predict import predict_text, extract_features, FakeNewsPredictor

    predict_text("Scientists confirm the moon is made of cheese")           # -> dict
    predict_text(["headline one", "headline two"], run_dir="outputs/bilstm_full")  # -> list[dict]
    extract_features("some headline")                                       # -> np.ndarray [fc_dim]
    extract_features(["a", "b"])                                            # -> np.ndarray [2, fc_dim]

CLI usage:
    python predict.py --text "some headline" [--run-dir outputs/bilstm_small] [--features]
    python predict.py --file headlines.txt          # one text per line
    python predict.py --interactive
"""
from __future__ import annotations

import argparse
import functools
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from engine import forward_features
from models import load_model_from_checkpoint
from preprocessing import Tokenizer, clean_text
from utils import get_device, load_config, resolve_path

log = logging.getLogger("predict")
TextInput = Union[str, Sequence[str]]


def find_latest_run(output_dir: str | Path | None = None, default_run: Optional[str] = None) -> Path:
    """Most recently trained run that has a best checkpoint (or ``default_run`` if given)."""
    cfg = load_config()
    out = resolve_path(output_dir or cfg["paths"]["output_dir"])
    name = default_run or cfg.get("inference", {}).get("default_run")
    if name:
        return out / name
    runs = [p.parent.parent for p in out.glob("*/checkpoints/best_model.pt")] if out.exists() else []
    if not runs:
        raise FileNotFoundError(f"No trained runs found in {out}. Train first: python train.py --mode small")
    return max(runs, key=lambda r: (r / "checkpoints" / "best_model.pt").stat().st_mtime)


class FakeNewsPredictor:
    """Loads a trained run once; then predicts / extracts features for any strings."""

    def __init__(self, run_dir: str | Path | None = None, device: str = "auto", batch_size: int = 512):
        self.run_dir = resolve_path(run_dir) if run_dir else find_latest_run()
        ckpt_path = self.run_dir / "checkpoints" / "best_model.pt"
        tok_path = self.run_dir / "tokenizer.json"
        for p in (ckpt_path, tok_path):
            if not p.exists():
                raise FileNotFoundError(f"{p} not found. Is --run-dir a trained run folder?")
        self.device = get_device(device)
        self.model, self.ckpt = load_model_from_checkpoint(ckpt_path, self.device)
        self.tokenizer = Tokenizer.load(tok_path)
        self.class_names: List[str] = list(self.ckpt.get("class_names", ["label_0", "label_1"]))
        self.threshold: float = float(self.ckpt.get("threshold", 0.5))
        self.batch_size = batch_size

    @staticmethod
    def _as_list(texts: TextInput) -> List[str]:
        if isinstance(texts, str):
            return [texts]
        return ["" if t is None else str(t) for t in texts]

    def predict(self, texts: TextInput, threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        thr = self.threshold if threshold is None else float(threshold)
        items = self._as_list(texts)
        if not items:
            return []
        ids, lengths = self.tokenizer.encode_batch(items)
        _, probs = forward_features(self.model, ids, lengths, self.device, self.batch_size, return_probs=True)
        results = []
        for text, p in zip(items, probs):
            label = int(p >= thr)
            results.append({
                "text": text,
                "clean_text": clean_text(text, self.tokenizer.lowercase, self.tokenizer.replace_urls),
                "label": label,
                "label_name": self.class_names[label],
                "prob_label_1": float(p),
                "confidence": float(p if label == 1 else 1.0 - p),
            })
        return results

    def extract_features(self, texts: TextInput) -> np.ndarray:
        """Dense representation from the layer before the output unit: [N, fc_dim] (or [fc_dim] for a str)."""
        items = self._as_list(texts)
        ids, lengths = self.tokenizer.encode_batch(items)
        feats = forward_features(self.model, ids, lengths, self.device, self.batch_size)
        return feats[0] if isinstance(texts, str) else feats


@functools.lru_cache(maxsize=4)
def _cached_predictor(run_dir: Optional[str], device: str) -> FakeNewsPredictor:
    return FakeNewsPredictor(run_dir, device)


def predict_text(text: TextInput, run_dir: str | Path | None = None, threshold: Optional[float] = None,
                 device: str = "auto") -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """Classify one string (returns a dict) or a list of strings (returns a list of dicts)."""
    predictor = _cached_predictor(str(run_dir) if run_dir else None, device)
    results = predictor.predict(text, threshold)
    return results[0] if isinstance(text, str) else results


def extract_features(text: TextInput, run_dir: str | Path | None = None, device: str = "auto") -> np.ndarray:
    """Feature vector(s) for arbitrary strings (see FakeNewsPredictor.extract_features)."""
    return _cached_predictor(str(run_dir) if run_dir else None, device).extract_features(text)


def _print(res: Dict[str, Any]) -> None:
    print(f"[{res['label_name']}] (label {res['label']}, P(label 1)={res['prob_label_1']:.4f}, "
          f"confidence={res['confidence']:.4f})  {res['text']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict with a trained Fakeddit model.")
    ap.add_argument("--run-dir", help="e.g. outputs/bilstm_small (default: latest trained run)")
    ap.add_argument("--text", action="append", help="text to classify (repeatable)")
    ap.add_argument("--file", help="UTF-8 text file with one input per line")
    ap.add_argument("--interactive", action="store_true", help="type texts interactively")
    ap.add_argument("--features", action="store_true", help="also print the feature-vector shape / head")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING)
    predictor = FakeNewsPredictor(args.run_dir, args.device)
    print(f"Loaded run: {predictor.run_dir} (device: {predictor.device})")

    texts: List[str] = list(args.text or [])
    if args.file:
        texts += [ln.rstrip("\n") for ln in Path(args.file).read_text(encoding="utf-8").splitlines() if ln.strip()]

    def handle(batch: List[str]) -> None:
        results = predictor.predict(batch, args.threshold)
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            for r in results:
                _print(r)
        if args.features:
            feats = predictor.extract_features(batch)
            print(f"features: shape={feats.shape}, first values of row 0: {np.round(feats[0][:8], 4)}")

    if texts:
        handle(texts)
    if args.interactive:
        print("Enter a headline (empty line to quit):")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            handle([line])
    if not texts and not args.interactive:
        ap.error("provide --text, --file or --interactive")


if __name__ == "__main__":
    main()
