"""SASRec (Kang and McAuley, 2018) with a softmax cross-entropy objective, used as the first-stage retriever."""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .retrievers import Retriever


class SASRec(nn.Module):
    def __init__(self, n_items: int, d: int = 64, max_len: int = 50, n_blocks: int = 2, n_heads: int = 1,
                 dropout: float = 0.2):
        super().__init__()
        self.n_items, self.max_len = n_items, max_len
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)   # index 0 is padding, item i -> i + 1
        self.pos_emb = nn.Embedding(max_len, d)
        self.drop = nn.Dropout(dropout)
        self.attn = nn.ModuleList([nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
                                   for _ in range(n_blocks)])
        self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_blocks)])
        self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(n_blocks)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout),
                                               nn.Linear(d, d), nn.Dropout(dropout))
                                 for _ in range(n_blocks)])
        self.ln_out = nn.LayerNorm(d)
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, L) item indices shifted by one, left padded with 0. Returns hidden states (B, L, d)."""
        B, L = seq.shape
        keep = (seq != 0).unsqueeze(-1).float()
        h = self.embed(seq) * (self.item_emb.embedding_dim ** 0.5)
        h = self.drop(h + self.pos_emb(torch.arange(L, device=seq.device))[None]) * keep
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1)
        for attn, ln1, ln2, ff in zip(self.attn, self.ln1, self.ln2, self.ff):
            q = ln1(h)
            a, _ = attn(q, q, q, attn_mask=causal, need_weights=False)
            h = (h + a) * keep
            h = (h + ff(ln2(h))) * keep
        return self.ln_out(h)

    def embed(self, seq: torch.Tensor) -> torch.Tensor:
        """Input representation of shifted item indices. Subclasses add side information here."""
        return self.item_emb(seq)

    def item_matrix(self) -> torch.Tensor:
        """Output representation of every item, row i for item i."""
        return self.item_emb.weight[1: self.n_items + 1]

    def item_logits(self, h: torch.Tensor, cand: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = self.item_matrix()
        return h @ (w if cand is None else w[cand]).T


def _pad(seqs: list[list[int]], max_len: int) -> torch.Tensor:
    out = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for r, s in enumerate(seqs):
        s = s[-max_len:]
        if s:
            out[r, -len(s):] = torch.tensor(s, dtype=torch.long) + 1
    return out


@torch.no_grad()
def _topk(model: SASRec, histories: list[list[int]], k: int, device, batch: int = 512):
    model.eval()
    idx, val = [], []
    for s in range(0, len(histories), batch):
        hs = histories[s: s + batch]
        seq = _pad(hs, model.max_len).to(device)
        logits = model.item_logits(model(seq)[:, -1]).float()
        for r, h in enumerate(hs):
            if h:
                logits[r, torch.tensor(h, device=device)] = float("-inf")
        v, i = torch.topk(logits, min(k, logits.shape[1]), dim=1)
        idx.append(i.cpu().numpy())
        val.append(v.cpu().numpy())
    return np.concatenate(idx), np.concatenate(val)


def _ndcg10(model: SASRec, histories, targets, device) -> float:
    idx, _ = _topk(model, histories, 10, device)
    hit = idx == np.asarray(targets)[:, None]
    gain = 1.0 / np.log2(hit.argmax(1) + 2.0)
    return float(np.mean(np.where(hit.any(1), gain, 0.0)))


def train_sasrec(seqs: list, n_items: int, out_dir: str, d: int = 64, max_len: int = 50, n_blocks: int = 2,
                 n_heads: int = 1, dropout: float = 0.2, lr: float = 1e-3, batch_size: int = 256,
                 epochs: int = 300, patience: int = 10, n_sampled: int = 0, max_full_softmax: int = 60000,
                 seed: int = 13, device: Optional[str] = None, eval_users: int = 10000) -> dict:
    """Train on each user's sequence minus the last two items, early-stop on the validation item."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    train = [[int(x[0]) for x in s[:-2]] for s in seqs]
    users = [u for u, s in enumerate(train) if len(s) >= 2]
    val_users = [u for u in range(len(seqs)) if len(seqs[u]) >= 3]
    if len(val_users) > eval_users:
        val_users = rng.choice(val_users, eval_users, replace=False).tolist()
    val_hist = [train[u] for u in val_users]
    val_tgt = [int(seqs[u][-2][0]) for u in val_users]
    if n_sampled == 0 and n_items > max_full_softmax:
        n_sampled = 4096
    model = SASRec(n_items, d, max_len, n_blocks, n_heads, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_ep, bad = -1.0, 0, 0
    os.makedirs(out_dir, exist_ok=True)
    cfg = {"n_items": n_items, "d": d, "max_len": max_len, "n_blocks": n_blocks, "n_heads": n_heads,
           "dropout": dropout}
    for ep in range(1, epochs + 1):
        model.train()
        order = rng.permutation(users)
        total, steps = 0.0, 0
        for s in range(0, len(order), batch_size):
            batch = [train[u][-(max_len + 1):] for u in order[s: s + batch_size]]
            inp = _pad([b[:-1] for b in batch], max_len).to(device)
            tgt = _pad([b[1:] for b in batch], max_len).to(device)          # shifted by one, 0 = pad
            h = model(inp)
            mask = tgt != 0
            hm, tm = h[mask], tgt[mask] - 1
            if n_sampled:
                neg = torch.from_numpy(rng.integers(0, n_items, n_sampled)).to(device)
                cand, inv = torch.unique(torch.cat([tm, neg]).cpu(), return_inverse=True)
                cand, inv = cand.to(device), inv.to(device)
                loss = F.cross_entropy(model.item_logits(hm, cand), inv[: len(tm)])
            else:
                loss = F.cross_entropy(model.item_logits(hm), tm)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, steps = total + loss.item(), steps + 1
        score = _ndcg10(model, val_hist, val_tgt, device)
        print(f"[sasrec] epoch {ep:3d} loss {total / max(steps, 1):.4f} valid NDCG@10 {score:.4f}", flush=True)
        if score > best + 1e-5:
            best, best_ep, bad = score, ep, 0
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
        else:
            bad += 1
            if bad >= patience:
                break
    info = {"config": cfg, "best_valid_ndcg10": best, "best_epoch": best_ep, "sampled_softmax": n_sampled}
    with open(os.path.join(out_dir, "sasrec.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


class SASRecRetriever(Retriever):
    name = "sasrec"

    def __init__(self, path: str, device: Optional[str] = None):
        with open(os.path.join(path, "sasrec.json")) as f:
            cfg = json.load(f)["config"]
        if device is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.model = SASRec(**cfg).to(device)
        self.model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=device))
        self.model.eval()

    def topk(self, histories: list[list[int]], k: int):
        return _topk(self.model, histories, k, self.device)

    @torch.no_grad()
    def score_items(self, histories: list[list[int]], items: list[list[int]], batch: int = 256) -> list[np.ndarray]:
        """SASRec scores of the given items for each history (the logits that `topk` ranks)."""
        out = []
        w = self.model.item_matrix()
        for s in range(0, len(histories), batch):
            seq = _pad(histories[s: s + batch], self.model.max_len).to(self.device)
            h = self.model(seq)[:, -1]
            for r, cand in enumerate(items[s: s + batch]):
                c = torch.tensor(cand, dtype=torch.long, device=self.device)
                out.append((h[r] @ w[c].T).float().cpu().numpy())
        return out
