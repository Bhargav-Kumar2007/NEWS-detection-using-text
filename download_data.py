"""Download the Fakeddit v2.0 text/metadata files and verify the columns we need.

Source (official): https://github.com/entitize/Fakeddit  ->  "Download text and metadata" (Google Drive folder)

Usage:
    python download_data.py            # download (skipped if the split files already exist)
    python download_data.py --force    # download again
    python download_data.py --check    # only verify files already placed in data/raw/
"""
from __future__ import annotations

import argparse
import gzip
import logging
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

from preprocessing import find_split_files, load_split
from utils import apply_overrides, load_config, resolve_path, save_json, setup_logging

log = logging.getLogger("download")

MANUAL_HELP = """
Automatic download failed. Manual fallback:
  1. Open https://github.com/entitize/Fakeddit and follow the v2.0 "Download text and metadata" link
     (Google Drive folder).
  2. Download the train / validate / test_public .tsv files (the 'all_*' files for the full dataset,
     or the ones in multimodal_only_samples/ if you set data.variant: multimodal_only).
  3. Put them anywhere inside: {data_dir}
  4. Run: python download_data.py --check
Common cause: Google Drive's download quota ("Too many users have viewed or downloaded this file").
Wait some hours, try another network, or use the manual route above.
"""


def extract_archives(data_dir: Path) -> None:
    """Unpack any .zip / .tar(.gz) / .tsv.gz that the Drive download produced."""
    for path in list(data_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        try:
            if name.endswith(".zip"):
                log.info("Extracting %s", path.name)
                with zipfile.ZipFile(path) as z:
                    z.extractall(path.parent)
            elif name.endswith((".tar.gz", ".tgz", ".tar")):
                log.info("Extracting %s", path.name)
                with tarfile.open(path) as t:
                    t.extractall(path.parent)
            elif name.endswith(".tsv.gz"):
                target = path.with_suffix("")  # strips .gz
                if not target.exists():
                    log.info("Decompressing %s", path.name)
                    with gzip.open(path, "rb") as fi, open(target, "wb") as fo:
                        shutil.copyfileobj(fi, fo)
        except (zipfile.BadZipFile, tarfile.TarError, OSError) as e:
            log.warning("Could not extract %s: %s", path.name, e)


def download_folder(folder_id: str, data_dir: Path) -> None:
    try:
        import gdown
    except ImportError:
        sys.exit("gdown is not installed. Run: pip install -r requirements.txt")
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    log.info("Downloading Fakeddit v2.0 folder from %s", url)
    data_dir.mkdir(parents=True, exist_ok=True)
    result = gdown.download_folder(url=url, output=str(data_dir), quiet=False,
                                   use_cookies=False)
    if not result:
        raise RuntimeError("gdown returned no files")


def verify(data_dir: Path, cfg: dict) -> bool:
    d = cfg["data"]
    try:
        files = find_split_files(data_dir, d["variant"], d.get("files"))
    except FileNotFoundError as e:
        log.error("%s", e)
        return False

    ok, summary = True, {"variant": d["variant"], "files": {}}
    for split, path in files.items():
        header = list(pd.read_csv(path, sep="\t", nrows=0).columns)
        missing = [c for c in (d["text_column"], d["label_column"]) if c not in header]
        if missing:
            log.error("%s is missing columns %s (available: %s)", path.name, missing, header)
            ok = False
            continue
        df = load_split(path, d["text_column"], d["label_column"])  # same reader training uses
        counts = df[d["label_column"]].astype(int).value_counts().sort_index().to_dict()
        log.info("%-10s %-32s rows=%-8d labels=%s", split, path.name, len(df), counts)
        summary["files"][split] = {"path": str(path), "rows": int(len(df)),
                                   "label_counts": {str(k): int(v) for k, v in counts.items()}}
    if ok:
        save_json(summary, data_dir / "dataset_summary.json")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Download and verify the Fakeddit dataset.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    ap.add_argument("--check", action="store_true", help="only verify existing files")
    ap.add_argument("--set", nargs="+", metavar="KEY=VALUE", help="config overrides")
    args = ap.parse_args()

    setup_logging()
    cfg = load_config(args.config)
    apply_overrides(cfg, args.set)
    data_dir = resolve_path(cfg["paths"]["data_dir"])

    if not args.check:
        have_files = True
        try:
            find_split_files(data_dir, cfg["data"]["variant"], cfg["data"].get("files"))
        except FileNotFoundError:
            have_files = False
        if have_files and not args.force:
            log.info("Split files already present in %s (use --force to re-download).", data_dir)
        else:
            try:
                download_folder(cfg["data"]["gdrive_folder_id"], data_dir)
                extract_archives(data_dir)
            except Exception as e:  # network / quota / gdown errors
                log.error("Download failed: %s", e)
                print(MANUAL_HELP.format(data_dir=data_dir))
                sys.exit(1)

    if verify(data_dir, cfg):
        log.info("Dataset OK. Next: python train.py --mode small")
    else:
        print(MANUAL_HELP.format(data_dir=data_dir))
        sys.exit(1)


if __name__ == "__main__":
    main()
