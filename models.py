"""Embedding + BiLSTM and Embedding + TextCNN binary classifiers.

Both models expose:
    forward(ids, lengths)           -> logits  [B]
    extract_features(ids, lengths)  -> dense sentence representation [B, fc_dim]
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

PAD_IDX = 0


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 200, hidden_dim: int = 128,
                 num_layers: int = 1, dropout: float = 0.4, fc_dim: int = 128, pad_idx: int = PAD_IDX):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        # final hidden states (fwd+bwd) concatenated with masked max-pool over time -> 4*hidden
        self.fc = nn.Linear(hidden_dim * 4, fc_dim)
        self.out = nn.Linear(fc_dim, 1)
        self.feature_dim = fc_dim

    def extract_features(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        seq_len = ids.size(1)
        lengths = lengths.detach().clamp(min=1, max=seq_len)
        emb = self.embed_dropout(self.embedding(ids))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq_len)

        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)                       # [B, 2H]
        mask = torch.arange(seq_len, device=ids.device)[None, :] < lengths.to(ids.device)[:, None]
        pooled = out.masked_fill(~mask.unsqueeze(-1), -1e4).max(dim=1).values  # [B, 2H]
        return F.relu(self.fc(self.dropout(torch.cat([h_last, pooled], dim=1))))

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.out(self.dropout(self.extract_features(ids, lengths))).squeeze(-1)


class TextCNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 200, num_filters: int = 128,
                 kernel_sizes: Sequence[int] = (2, 3, 4, 5), dropout: float = 0.4,
                 fc_dim: int = 128, pad_idx: int = PAD_IDX):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        self.kernel_sizes = list(kernel_sizes)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in self.kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(self.kernel_sizes), fc_dim)
        self.out = nn.Linear(fc_dim, 1)
        self.feature_dim = fc_dim

    def extract_features(self, ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        x = self.embed_dropout(self.embedding(ids)).transpose(1, 2)        # [B, E, T]
        min_len = max(self.kernel_sizes)
        if x.size(2) < min_len:                                            # very short max_len
            x = F.pad(x, (0, min_len - x.size(2)))
        pooled = [F.relu(conv(x)).max(dim=2).values for conv in self.convs]
        return F.relu(self.fc(self.dropout(torch.cat(pooled, dim=1))))

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        return self.out(self.dropout(self.extract_features(ids, lengths))).squeeze(-1)


# --------------------------------------------------------------------------- #
# Factory / checkpoint helpers
# --------------------------------------------------------------------------- #
def model_config_from_cfg(cfg: Dict[str, Any], vocab_size: int) -> Dict[str, Any]:
    m = cfg["model"]
    mtype = m["type"].lower()
    base = {"type": mtype, "vocab_size": int(vocab_size), "embed_dim": m["embed_dim"],
            "dropout": m["dropout"], "fc_dim": m["fc_dim"]}
    if mtype == "bilstm":
        base.update(hidden_dim=m["bilstm"]["hidden_dim"], num_layers=m["bilstm"]["num_layers"])
    elif mtype == "textcnn":
        base.update(num_filters=m["textcnn"]["num_filters"], kernel_sizes=list(m["textcnn"]["kernel_sizes"]))
    else:
        raise ValueError(f"Unknown model.type '{m['type']}'. Use 'bilstm' or 'textcnn'.")
    return base


def build_model(model_config: Dict[str, Any]) -> nn.Module:
    c = dict(model_config)
    mtype = c.pop("type")
    if mtype == "bilstm":
        model = BiLSTMClassifier(**c)
    elif mtype == "textcnn":
        model = TextCNNClassifier(**c)
    else:
        raise ValueError(f"Unknown model type '{mtype}'")
    model.config = dict(model_config)
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model_from_checkpoint(path: str | Path, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """Rebuild the architecture stored in a checkpoint and load its weights (eval mode)."""
    # weights_only=False: these files are produced by this project (they hold plain dicts + tensors).
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt
