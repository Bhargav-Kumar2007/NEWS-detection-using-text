"""Installation smoke test. No dataset needed and NO training happens.

Run:  python tests/test_smoke.py     (or: pytest tests)
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from preprocessing import PAD_IDX, UNK_IDX, Tokenizer, clean_text, find_split_files


def test_clean_text():
    assert clean_text("My Cat's  FIRST day!!! https://x.com/a?b=1") == "my cat's first day"
    assert clean_text("Tom &amp; Jerry") == "tom jerry"
    assert clean_text(None) == "" and clean_text(float("nan")) == "" and clean_text(123) == "123"
    assert clean_text(clean_text("Hello, World!")) == "hello world"  # idempotent


def test_tokenizer_roundtrip():
    texts = ["the cat sat", "the dog sat", "a cat ran", "the end"]
    tok = Tokenizer(max_vocab_size=50, min_freq=2, max_len=5).fit(texts)
    assert tok.stoi["<pad>"] == PAD_IDX and tok.stoi["<unk>"] == UNK_IDX
    assert "the" in tok.stoi and "ran" not in tok.stoi          # min_freq filtering
    ids, lengths = tok.encode_batch(["the cat sat on the mat today", "", "the"], max_len=5)
    assert ids.shape == (3, 5) and ids.dtype == np.int32
    assert lengths.tolist() == [5, 1, 1]                        # truncated / empty -> <unk> / short
    assert ids[2, 1:].sum() == 0                                # right padding with 0
    with tempfile.TemporaryDirectory() as d:
        tok.save(Path(d) / "tok.json")
        tok2 = Tokenizer.load(Path(d) / "tok.json")
    assert tok2.itos == tok.itos and tok2.encode("the cat") == tok.encode("the cat")


def test_find_split_files():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "multimodal_only_samples").mkdir()
        for n in ("all_train.tsv", "all_validate.tsv", "all_test_public.tsv", "comments.tsv",
                  "multimodal_only_samples/multimodal_train.tsv",
                  "multimodal_only_samples/multimodal_validate.tsv",
                  "multimodal_only_samples/multimodal_test_public.tsv"):
            (d / n).write_text("x")
        full = find_split_files(d, "all")
        assert {k: v.name for k, v in full.items()} == {
            "train": "all_train.tsv", "validation": "all_validate.tsv", "test": "all_test_public.tsv"}
        mm = find_split_files(d, "multimodal_only")
        assert mm["train"].name == "multimodal_train.tsv"


def test_models_forward():
    try:
        import torch
    except ImportError:
        print("torch not installed - skipping model test")
        return
    from models import build_model, model_config_from_cfg
    from utils import load_config

    cfg = load_config()
    ids = torch.randint(2, 100, (4, 12))
    lengths = torch.tensor([12, 7, 3, 1])
    ids[1, 7:] = 0
    ids[2, 3:] = 0
    ids[3, 1:] = 0
    for mtype in ("bilstm", "textcnn"):
        cfg["model"]["type"] = mtype
        model = build_model(model_config_from_cfg(cfg, vocab_size=100)).eval()
        logits = model(ids, lengths)
        feats = model.extract_features(ids, lengths)
        assert logits.shape == (4,) and feats.shape == (4, cfg["model"]["fc_dim"])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok:", name)
    print("All smoke tests passed.")
