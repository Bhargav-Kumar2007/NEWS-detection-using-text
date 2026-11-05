"""Text cleaning, tokenisation, vocabulary building, padding and data loading.

Only the standard library + numpy/pandas are needed for cleaning and tokenising;
torch is imported lazily by the DataLoader helpers.
"""
from __future__ import annotations

import csv
import html
import json
import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

log = logging.getLogger(__name__)

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
PAD_IDX, UNK_IDX = 0, 1

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# words made of letters/digits (any language), optionally joined by apostrophes: don't, it's
_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


# --------------------------------------------------------------------------- #
# Cleaning + tokenisation
# --------------------------------------------------------------------------- #
def clean_text(text: Any, lowercase: bool = True, replace_urls: bool = True) -> str:
    """Normalise one string: unescape HTML, strip URLs/punctuation, lowercase, collapse spaces.

    Accepts any input (None / NaN / numbers are handled). The function is idempotent,
    so it is safe to apply on the already-filtered ``clean_title`` column.
    """
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("`", "'")
    if replace_urls:
        text = _URL_RE.sub(" ", text)
    if lowercase:
        text = text.lower()
    return " ".join(_TOKEN_RE.findall(text))


class Tokenizer:
    """Word-level tokenizer: cleaning -> whitespace tokens -> vocabulary ids -> padding."""

    def __init__(self, max_vocab_size: int = 50000, min_freq: int = 2, max_len: int = 50,
                 lowercase: bool = True, replace_urls: bool = True):
        self.max_vocab_size = int(max_vocab_size)
        self.min_freq = int(min_freq)
        self.max_len = int(max_len)
        self.lowercase = bool(lowercase)
        self.replace_urls = bool(replace_urls)
        self.itos: List[str] = [PAD_TOKEN, UNK_TOKEN]
        self.stoi: Dict[str, int] = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}

    # -- vocabulary ---------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def tokenize(self, text: Any) -> List[str]:
        return clean_text(text, self.lowercase, self.replace_urls).split()

    def fit(self, texts: Iterable[Any]) -> "Tokenizer":
        """Build the vocabulary from (training) texts only."""
        counter: Counter = Counter()
        for t in tqdm(texts, desc="Building vocabulary", leave=False):
            counter.update(self.tokenize(t))
        self.itos = [PAD_TOKEN, UNK_TOKEN]
        budget = max(self.max_vocab_size - len(self.itos), 0)
        for word, freq in counter.most_common():
            if len(self.itos) - 2 >= budget:
                break
            if freq < self.min_freq:
                break
            self.itos.append(word)
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        log.info("Vocabulary: %d tokens kept (from %d distinct words, min_freq=%d)",
                 self.vocab_size, len(counter), self.min_freq)
        return self

    # -- encoding -----------------------------------------------------------
    def encode(self, text: Any, max_len: Optional[int] = None) -> List[int]:
        """Unpadded list of token ids (truncated to max_len; empty text -> [<unk>])."""
        limit = max_len or self.max_len
        ids = [self.stoi.get(w, UNK_IDX) for w in self.tokenize(text)[:limit]]
        return ids or [UNK_IDX]

    def encode_batch(self, texts: Sequence[Any], max_len: Optional[int] = None,
                     show_progress: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ids[int32, N x L] right-padded with <pad>=0, lengths[int64, N])."""
        limit = max_len or self.max_len
        n = len(texts)
        ids = np.zeros((n, limit), dtype=np.int32)
        lengths = np.ones(n, dtype=np.int64)
        iterator = tqdm(range(n), desc="Encoding", leave=False) if show_progress else range(n)
        for i in iterator:
            row = self.encode(texts[i], limit)
            ids[i, :len(row)] = row
            lengths[i] = len(row)
        return ids, lengths

    def decode(self, ids: Iterable[int]) -> str:
        return " ".join(self.itos[i] for i in ids if i != PAD_IDX and i < len(self.itos))

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_vocab_size": self.max_vocab_size, "min_freq": self.min_freq,
            "max_len": self.max_len, "lowercase": self.lowercase,
            "replace_urls": self.replace_urls, "itos": self.itos,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        tok = cls(d["max_vocab_size"], d["min_freq"], d["max_len"], d["lowercase"], d["replace_urls"])
        tok.itos = list(d["itos"])
        tok.stoi = {w: i for i, w in enumerate(tok.itos)}
        return tok


def tokenizer_from_config(cfg: Dict[str, Any]) -> Tokenizer:
    p = cfg["preprocessing"]
    return Tokenizer(p["max_vocab_size"], p["min_freq"], p["max_len"], p["lowercase"], p["replace_urls"])


# --------------------------------------------------------------------------- #
# Locating and reading the dataset files
# --------------------------------------------------------------------------- #
SPLITS = ("train", "validation", "test")


def find_split_files(data_dir: str | Path, variant: str = "all",
                     explicit: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, Path]:
    """Locate the official train / validation / test TSV files under ``data_dir``.

    variant="all"             -> full text dataset (files not inside / named "multimodal")
    variant="multimodal_only" -> only samples that have both text and image
    """
    data_dir = Path(data_dir)
    found: Dict[str, Path] = {}
    explicit = explicit or {}
    all_tsv = sorted(p for p in data_dir.rglob("*.tsv") if "comment" not in p.name.lower())

    def is_multimodal(p: Path) -> bool:
        # The official text folder is named "all_samples (also includes non multimodal)";
        # match the multimodal variant's actual directory/file prefix instead of the substring.
        return any(part.lower().startswith("multimodal_")
                   for part in p.relative_to(data_dir).parts)

    pool = [p for p in all_tsv if is_multimodal(p)] if variant == "multimodal_only" \
        else [p for p in all_tsv if not is_multimodal(p)]

    keywords = {"train": ("train",), "validation": ("validate", "validation", "val"), "test": ("test",)}
    for split in SPLITS:
        if explicit.get(split):
            path = data_dir / explicit[split]
            if not path.exists():
                raise FileNotFoundError(f"Configured file for '{split}' not found: {path}")
            found[split] = path
            continue
        cands = [p for p in pool if any(k in p.name.lower() for k in keywords[split])]
        if split == "test":
            cands = sorted(cands, key=lambda p: (0 if "public" in p.name.lower() else 1, p.name))
        if cands:
            found[split] = cands[0]

    missing = [s for s in SPLITS if s not in found]
    if missing:
        listing = "\n  ".join(str(p.relative_to(data_dir)) for p in all_tsv) or "(no .tsv files found)"
        raise FileNotFoundError(
            f"Could not find {missing} files for variant '{variant}' in {data_dir}.\n"
            f"TSV files present:\n  {listing}\n"
            "Run `python download_data.py`, or copy the files there manually, or set "
            "data.files.* in config.yaml.")
    return found


def load_split(path: str | Path, text_col: str, label_col: str) -> pd.DataFrame:
    """Read one TSV, keeping only the text and label columns."""
    path = Path(path)
    kwargs = dict(sep="\t", usecols=[text_col, label_col], dtype={text_col: str})
    try:
        df = pd.read_csv(path, **kwargs)
    except ValueError:
        header = list(pd.read_csv(path, sep="\t", nrows=0).columns)
        raise ValueError(f"{path.name}: columns '{text_col}' / '{label_col}' not found. "
                         f"Available columns: {header}") from None
    except pd.errors.ParserError:
        log.warning("%s: default quoting failed; re-reading with quoting disabled", path.name)
        df = pd.read_csv(path, quoting=csv.QUOTE_NONE, **kwargs)

    df = df.dropna(subset=[label_col]).copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    df[label_col] = df[label_col].astype(int)
    bad = sorted(set(df[label_col].unique()) - {0, 1})
    if bad:
        raise ValueError(f"{path.name}: expected binary labels {{0,1}} in '{label_col}', found {bad}")
    return df.reset_index(drop=True)


def stratified_subsample(df: pd.DataFrame, n: Optional[int], label_col: str, seed: int) -> pd.DataFrame:
    """Stratified random subset of ``n`` rows (returns df unchanged if n is None / too large)."""
    if n is None or n >= len(df):
        return df
    from sklearn.model_selection import train_test_split

    sub, _ = train_test_split(df, train_size=int(n), stratify=df[label_col], random_state=seed)
    return sub.reset_index(drop=True)


def load_splits(cfg: Dict[str, Any], mode: str, only: Optional[Sequence[str]] = None
                ) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """Load official splits -> {split: (texts, labels)}; ``mode`` is 'small' or 'full'."""
    from utils import resolve_path

    d = cfg["data"]
    if mode not in d["subset_sizes"]:
        raise ValueError(f"Unknown mode '{mode}'. Choose from {list(d['subset_sizes'])}")
    data_dir = resolve_path(cfg["paths"]["data_dir"])
    if not data_dir.exists():
        raise FileNotFoundError(f"{data_dir} does not exist. Run `python download_data.py` first.")
    files = find_split_files(data_dir, d["variant"], d.get("files"))
    out: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for split in (only or SPLITS):
        df = load_split(files[split], d["text_column"], d["label_column"])
        n_full = len(df)
        df = stratified_subsample(df, d["subset_sizes"][mode].get(split), d["label_column"],
                                  cfg.get("seed", 42))
        labels = df[d["label_column"]].to_numpy(dtype=np.int64)
        counts = np.bincount(labels, minlength=2)
        log.info("[%s] %s -> %d/%d rows | label 0: %d, label 1: %d",
                 split, files[split].name, len(df), n_full, counts[0], counts[1])
        out[split] = (df[d["text_column"]].tolist(), labels)
    return out


# --------------------------------------------------------------------------- #
# Torch dataset / loader (batched indexing for speed)
# --------------------------------------------------------------------------- #
def make_loader(ids: np.ndarray, lengths: np.ndarray, labels: np.ndarray, batch_size: int,
                shuffle: bool, num_workers: int = 0, pin_memory: bool = False):
    """DataLoader yielding (ids[B,L] int32, lengths[B] int64, labels[B] float32)."""
    import torch
    from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

    class ArrayDataset(Dataset):
        def __init__(self):
            self.ids = torch.from_numpy(ids)
            self.lengths = torch.from_numpy(lengths)
            self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):  # idx is a list of indices (batched)
            return self.ids[idx], self.lengths[idx], self.labels[idx]

    ds = ArrayDataset()
    base = RandomSampler(ds) if shuffle else SequentialSampler(ds)
    batches = BatchSampler(base, batch_size=batch_size, drop_last=False)
    return DataLoader(ds, sampler=batches, batch_size=None, num_workers=num_workers,
                      pin_memory=pin_memory)
