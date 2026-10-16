"""Trained recommendation baselines, all evaluated by full-catalogue ranking under one protocol.

    pop, itemknn   training-free
    bprmf          matrix factorisation with the BPR loss (Rendle et al., 2009)
    lightgcn       graph collaborative filtering (He et al., 2020)
    gru4rec        recurrent sequential model with a softmax loss (Hidasi et al., 2016)
    sasrec         self-attentive sequential model (Kang and McAuley, 2018)
    sasrecf        SASRec with knowledge-graph attribute embeddings added to the item representation
    sasrecf_dm     SASRecF whose attributes also include typed item facets answered by a decision model
    sasrec_dm      the same with the decision-model facets only, without the raw metadata attributes
    sasrecf_emb    SASRecF with facets assigned by text-embedding similarity instead (control for sasrecf_dm)
    bert4rec       bidirectional masked-item model (Sun et al., 2019)

Every model is trained on each user's sequence minus the last two items, early-stopped on the
second-to-last item, and tested on the last item with the validation item added to the input of the
sequential models. Hyperparameters are chosen on validation NDCG@10 from a small grid. Models and
configurations are trained strictly one after the other. Budget: at most 300 epochs with early
stopping after 10 validations without improvement (the RecBole defaults); LightGCN may run up to
1,000 epochs as in its paper.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.schema import Dataset
from .retrievers import (ContentKNNRetriever, ItemKNNRetriever, PopRetriever, SeqRulesRetriever,
                         TransTextRetriever)
from .sasrec import SASRec, _pad

# Facet models: name -> (facet source in ds.extra["facet_attrs"], keep the raw metadata attributes)
FACET_MODELS = {"sasrecf_dm": ("dm", True), "sasrec_dm": ("dm", False), "sasrecf_emb": ("emb", True)}
SEQUENTIAL = ("gru4rec", "sasrec", "sasrecf", "bert4rec") + tuple(FACET_MODELS)
COLLABORATIVE = ("bprmf", "lightgcn")
TRAINING_FREE = ("pop", "itemknn", "seqrules", "contentknn", "transtext")
TEXT_MODELS = ("contentknn", "transtext")      # need item-text embeddings in ds.extra["item_text_matrix"]
MODELS = ("pop", "itemknn") + COLLABORATIVE + tuple(m for m in SEQUENTIAL if m not in FACET_MODELS)   # need no extras

DEFAULT_GRID = {
    "pop": [{}], "itemknn": [{}],
    "seqrules": [{"recent": 1}, {"recent": 3}],
    "contentknn": [{"recent": 1}, {"recent": 3}],
    "transtext": [{"first": 50, "second": 10}, {"first": 20, "second": 20}],
    "bprmf": [{"d": 64, "lr": 1e-3, "reg": 1e-5}, {"d": 128, "lr": 1e-3, "reg": 1e-4}],
    "lightgcn": [{"d": 64, "lr": 1e-3, "layers": 3, "reg": 1e-4, "epochs": 1000},
                 {"d": 64, "lr": 1e-3, "layers": 2, "reg": 1e-4, "epochs": 1000}],
    "gru4rec": [{"d": 64, "lr": 1e-3, "dropout": 0.2}, {"d": 128, "lr": 1e-3, "dropout": 0.3}],
    "sasrec": [{"d": 64, "lr": 1e-3, "dropout": 0.2, "n_blocks": 2}, {"d": 128, "lr": 1e-3, "dropout": 0.3, "n_blocks": 2}],
    "sasrecf": [{"d": 64, "lr": 1e-3, "dropout": 0.2, "n_blocks": 2}, {"d": 128, "lr": 1e-3, "dropout": 0.3, "n_blocks": 2}],
    **{m: [{"d": 64, "lr": 1e-3, "dropout": 0.2, "n_blocks": 2}, {"d": 128, "lr": 1e-3, "dropout": 0.3, "n_blocks": 2}]
       for m in FACET_MODELS},
    "bert4rec": [{"d": 64, "lr": 1e-3, "dropout": 0.2, "n_blocks": 2, "mask_prob": 0.3},
                 {"d": 128, "lr": 1e-3, "dropout": 0.2, "n_blocks": 2, "mask_prob": 0.5}],
}


def auto_device(device: Optional[str] = None) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


# ---- models ----------------------------------------------------------------

class GRU4Rec(nn.Module):
    def __init__(self, n_items: int, d: int = 64, max_len: int = 50, dropout: float = 0.2, **_):
        super().__init__()
        self.n_items, self.max_len = n_items, max_len
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.gru = nn.GRU(d, d, batch_first=True)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.item_emb.weight, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.drop(self.item_emb(seq)))
        return self.drop(h)

    def item_logits(self, h: torch.Tensor, cand: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = self.item_emb.weight[1: self.n_items + 1]
        return h @ (w if cand is None else w[cand]).T


class SASRecF(SASRec):
    """SASRec whose item representation is the item embedding plus the mean of its attribute embeddings."""

    def __init__(self, n_items: int, attr_table: torch.Tensor, n_attrs: int, **kw):
        super().__init__(n_items, **kw)
        self.register_buffer("attr_table", attr_table)                 # (n_items + 1, A), 0 = no attribute
        self.attr_emb = nn.Embedding(n_attrs + 1, self.item_emb.embedding_dim, padding_idx=0)
        nn.init.normal_(self.attr_emb.weight, std=0.02)

    def embed(self, seq: torch.Tensor) -> torch.Tensor:
        ids = self.attr_table[seq]
        mask = (ids != 0).float().unsqueeze(-1)
        return self.item_emb(seq) + (self.attr_emb(ids) * mask).sum(-2) / mask.sum(-2).clamp(min=1.0)

    def item_matrix(self) -> torch.Tensor:
        return self.embed(torch.arange(1, self.n_items + 1, device=self.attr_table.device))


class BERT4Rec(nn.Module):
    """Bidirectional transformer trained to recover masked items. Index n_items + 1 is the mask token."""

    def __init__(self, n_items: int, d: int = 64, max_len: int = 50, n_blocks: int = 2, n_heads: int = 2,
                 dropout: float = 0.2, **_):
        super().__init__()
        self.n_items, self.max_len, self.mask_id = n_items, max_len, n_items + 1
        self.item_emb = nn.Embedding(n_items + 2, d, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_blocks, enable_nested_tensor=False)
        self.ln, self.drop = nn.LayerNorm(d), nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.zeros(n_items))
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        L = seq.shape[1]
        h = self.drop(self.item_emb(seq) + self.pos_emb(torch.arange(L, device=seq.device))[None])
        pad = seq == 0
        pad[:, -1] = False                       # keep one visible position so no row is fully masked
        return self.ln(self.encoder(h, src_key_padding_mask=pad))

    def item_logits(self, h: torch.Tensor, cand: Optional[torch.Tensor] = None) -> torch.Tensor:
        w = self.item_emb.weight[1: self.n_items + 1]
        if cand is None:
            return h @ w.T + self.bias
        return h @ w[cand].T + self.bias[cand]


class BPRMF(nn.Module):
    def __init__(self, n_users: int, n_items: int, d: int = 64, **_):
        super().__init__()
        self.user = nn.Embedding(n_users, d)
        self.item = nn.Embedding(n_items, d)
        nn.init.normal_(self.user.weight, std=0.05)
        nn.init.normal_(self.item.weight, std=0.05)

    def embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.user.weight, self.item.weight


class LightGCN(BPRMF):
    """`graph` is (rows, cols, values) of the symmetrically normalised user-item adjacency. Propagation
    uses a sparse product on CUDA and CPU, and an equivalent gather and index-add on Apple MPS, where
    sparse tensors are not supported."""

    def __init__(self, n_users: int, n_items: int, graph: tuple, d: int = 64, layers: int = 2, **_):
        super().__init__(n_users, n_items, d)
        self.layers, self.n_users, self.n_nodes = layers, n_users, n_users + n_items
        rows, cols, vals = graph
        self.register_buffer("rows", rows, persistent=False)
        self.register_buffer("cols", cols, persistent=False)
        self.register_buffer("vals", vals, persistent=False)
        self._adj = None

    def _propagate(self, e: torch.Tensor) -> torch.Tensor:
        if e.device.type == "mps":
            out = torch.zeros_like(e)
            return out.index_add_(0, self.rows, e[self.cols] * self.vals.unsqueeze(1))
        if self._adj is None or self._adj.device != e.device:
            self._adj = torch.sparse_coo_tensor(torch.stack([self.rows, self.cols]), self.vals,
                                                (self.n_nodes, self.n_nodes), check_invariants=False).coalesce()
        return torch.sparse.mm(self._adj, e)

    def embeddings(self) -> tuple[torch.Tensor, torch.Tensor]:
        e = torch.cat([self.user.weight, self.item.weight], 0)
        acc = e
        for _ in range(self.layers):
            e = self._propagate(e)
            acc = acc + e
        acc = acc / (self.layers + 1)
        return acc[: self.n_users], acc[self.n_users:]


def _attr_table(ds: Dataset, max_attrs: int = 12, facets: Optional[list] = None,
                raw: bool = True) -> tuple[torch.Tensor, int]:
    """Attribute ids per item (row i + 1 for item i, 0 = none). Without facets the table is exactly the one of
    the trained SASRecF checkpoints, whose state holds it. Facets come first and are not cut by `max_attrs`,
    which limits the raw metadata attributes only."""
    raw_rows = [[tuple(a) for a in it.get("attrs", [])] if raw else [] for it in ds.items]
    fac_rows = [[tuple(a) for a in facets[i]] if facets is not None else [] for i in range(ds.n_items)]
    vocab = {key: n + 1 for n, key in enumerate(sorted({a for row in raw_rows + fac_rows for a in row}))}
    width = max(1, max((len(r) for r in fac_rows), default=0) + (max_attrs if raw else 0))
    table = torch.zeros((ds.n_items + 1, width), dtype=torch.long)
    for i in range(ds.n_items):
        ids = [vocab[a] for a in fac_rows[i]] + [vocab[a] for a in raw_rows[i]][:max_attrs]
        if ids:
            table[i + 1, : len(ids)] = torch.tensor(ids)
    return table, len(vocab)


def _adjacency(ds: Dataset) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Edges of the bipartite training graph in both directions, with 1 / sqrt(deg_u * deg_i) weights."""
    users = np.concatenate([np.full(max(len(s) - 2, 0), u, dtype=np.int64) for u, s in enumerate(ds.seqs)])
    items = np.concatenate([np.array([x[0] for x in s[:-2]], dtype=np.int64) for s in ds.seqs]) + ds.n_users
    r, c = np.concatenate([users, items]), np.concatenate([items, users])
    deg = np.bincount(r, minlength=ds.n_users + ds.n_items).astype(np.float32)
    val = 1.0 / np.sqrt(np.maximum(deg[r], 1.0) * np.maximum(deg[c], 1.0))
    return torch.from_numpy(r), torch.from_numpy(c), torch.from_numpy(val.astype(np.float32))


def build_model(name: str, ds: Dataset, hp: dict, device: str) -> nn.Module:
    hp = {k: v for k, v in hp.items() if k not in ("lr", "reg", "mask_prob", "batch_size", "epochs", "patience")}
    if name == "gru4rec":
        return GRU4Rec(ds.n_items, **hp).to(device)
    if name == "sasrec":
        return SASRec(ds.n_items, **hp).to(device)
    if name == "sasrecf":
        table, n_attrs = _attr_table(ds)
        return SASRecF(ds.n_items, table, n_attrs, **hp).to(device)
    if name in FACET_MODELS:
        source, raw = FACET_MODELS[name]
        facets = ds.extra.get("facet_attrs", {}).get(source)
        if facets is None:
            raise SystemExit(f"{name} needs item facets from source {source!r}: set `facets:` in the config "
                             "and run the facet step first")
        table, n_attrs = _attr_table(ds, facets=facets, raw=raw)
        return SASRecF(ds.n_items, table, n_attrs, **hp).to(device)
    if name == "bert4rec":
        return BERT4Rec(ds.n_items, **hp).to(device)
    if name == "bprmf":
        return BPRMF(ds.n_users, ds.n_items, **hp).to(device)
    if name == "lightgcn":
        return LightGCN(ds.n_users, ds.n_items, _adjacency(ds), **hp).to(device)
    raise ValueError(f"unknown model {name!r}")


# ---- scoring ---------------------------------------------------------------

class Scorer:
    """Scores every item for a batch of users. History items are masked by the caller."""

    def __init__(self, name: str, ds: Dataset, model=None, device: str = "cpu", hp: Optional[dict] = None):
        self.name, self.ds, self.model, self.device = name, ds, model, device
        hp = dict(hp or {})
        if name in TEXT_MODELS and ds.extra.get("item_text_matrix") is None:
            raise SystemExit(f"{name} needs item-text embeddings: set `item_text:` (embedder, cache) in the config")
        if name == "pop":
            self.free = PopRetriever(ds.seqs, ds.n_items)
        elif name == "itemknn":
            self.free = ItemKNNRetriever(ds.seqs, ds.n_items)
        elif name == "seqrules":
            self.free = SeqRulesRetriever(ds.seqs, ds.n_items, **hp)
        elif name == "contentknn":
            self.free = ContentKNNRetriever(ds.extra["item_text_matrix"], PopRetriever(ds.seqs, ds.n_items).pop, **hp)
        elif name == "transtext":
            self.free = TransTextRetriever(ds.seqs, ds.n_items, ds.extra["item_text_matrix"], **hp)

    @torch.no_grad()
    def scores(self, users: list[int], histories: list[list[int]]) -> torch.Tensor:
        """(batch, n_items) scores, left on the model's device so that large catalogues are not copied."""
        if self.name in TRAINING_FREE:
            return torch.from_numpy(np.stack([self.free.scores(h) for h in histories])).float()
        self.model.eval()
        if self.name in COLLABORATIVE:
            ue, ie = self.model.embeddings()
            return (ue[torch.tensor(users, device=self.device)] @ ie.T).float()
        if self.name == "bert4rec":
            m = self.model
            seq = _pad([h[-(m.max_len - 1):] for h in histories], m.max_len - 1).to(self.device)
            seq = torch.cat([seq, torch.full((len(histories), 1), m.mask_id, device=self.device)], 1)
            return m.item_logits(m(seq)[:, -1]).float()
        seq = _pad(histories, self.model.max_len).to(self.device)
        return self.model.item_logits(self.model(seq)[:, -1]).float()

    @torch.no_grad()
    def ranks(self, users: list[int], histories: list[list[int]], targets: list[int], batch: int = 256) -> np.ndarray:
        """1-based rank of each target over the whole catalogue, history excluded, ties counted against it."""
        out = []
        for s in range(0, len(users), batch):
            hs = histories[s: s + batch]
            sc = self.scores(users[s: s + batch], hs)
            rows = torch.tensor([r for r, h in enumerate(hs) for _ in h], dtype=torch.long, device=sc.device)
            cols = torch.tensor([i for h in hs for i in h], dtype=torch.long, device=sc.device)
            if len(rows):
                sc[rows, cols] = float("-inf")
            t = torch.tensor(targets[s: s + batch], device=sc.device)
            ts = sc.gather(1, t.unsqueeze(1))
            out.append(((sc > ts).sum(1) + (sc == ts).sum(1)).cpu().numpy())
        return np.concatenate(out).astype(float)


def _ndcg10(ranks: np.ndarray) -> float:
    return float(np.mean(np.where(ranks <= 10, 1.0 / np.log2(ranks + 1.0), 0.0)))


# ---- training --------------------------------------------------------------

def _valid_set(ds: Dataset, rng: np.random.Generator, n: int):
    users = [u for u in range(ds.n_users) if len(ds.seqs[u]) >= 3]
    if len(users) > n:
        users = sorted(rng.choice(users, n, replace=False).tolist())
    return users, [[int(x[0]) for x in ds.seqs[u][:-2]] for u in users], [int(ds.seqs[u][-2][0]) for u in users]


def fit(name: str, ds: Dataset, out_dir: str, hp: Optional[dict] = None, epochs: int = 300, patience: int = 10,
        seed: int = 13, device: Optional[str] = None, eval_users: int = 10000, n_sampled: int = 0,
        max_full_softmax: int = 60000) -> dict:
    """Train one configuration, keep the checkpoint with the best validation NDCG@10.

    The training budget follows the convention of the RecBole library, which most recent recommender
    papers adopt: at most 300 epochs, Adam, validation after every epoch, and early stopping after
    10 validations without improvement. Models are trained one at a time."""
    hp = dict(hp or {})
    device = auto_device(device)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    vu, vh, vt = _valid_set(ds, rng, eval_users)
    if name in TRAINING_FREE:
        info = {"model": name, "hp": hp, "best_valid_ndcg10": _ndcg10(Scorer(name, ds, hp=hp).ranks(vu, vh, vt)),
                "best_epoch": 0}
        with open(os.path.join(out_dir, "info.json"), "w") as f:
            json.dump(info, f, indent=1)
        return info
    model = build_model(name, ds, hp, device)
    scorer = Scorer(name, ds, model, device)
    lr, reg = float(hp.get("lr", 1e-3)), float(hp.get("reg", 0.0))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    epochs, patience = int(hp.get("epochs", epochs)), int(hp.get("patience", patience))
    train = [[int(x[0]) for x in s[:-2]] for s in ds.seqs]
    if n_sampled == 0 and ds.n_items > max_full_softmax:
        n_sampled = 4096
    if name in COLLABORATIVE:
        pairs = np.array([(u, i) for u, s in enumerate(train) for i in s], dtype=np.int64)
        batch_size, check = int(hp.get("batch_size", 2048)), 1
    else:
        users = [u for u, s in enumerate(train) if len(s) >= 2]
        batch_size, check = int(hp.get("batch_size", 256)), 1
        max_len = model.max_len
    best, best_ep, bad = -1.0, 0, 0
    for ep in range(1, epochs + 1):
        model.train()
        total, steps = 0.0, 0
        if name in COLLABORATIVE:
            order = rng.permutation(len(pairs))
            for s in range(0, len(order), batch_size):
                b = pairs[order[s: s + batch_size]]
                u = torch.from_numpy(b[:, 0]).to(device)
                pos = torch.from_numpy(b[:, 1]).to(device)
                neg = torch.from_numpy(rng.integers(0, ds.n_items, len(b))).to(device)
                ue, ie = model.embeddings()
                diff = (ue[u] * ie[pos]).sum(1) - (ue[u] * ie[neg]).sum(1)
                l2 = (model.user(u) ** 2).sum() + (model.item(pos) ** 2).sum() + (model.item(neg) ** 2).sum()
                loss = -F.logsigmoid(diff).mean() + reg * l2 / len(b)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, steps = total + loss.item(), steps + 1
        else:
            order = rng.permutation(users)
            for s in range(0, len(order), batch_size):
                seqs = [train[u] for u in order[s: s + batch_size]]
                if name == "bert4rec":
                    full = _pad([q[-max_len:] for q in seqs], max_len).to(device)
                    real = full != 0
                    pick = (torch.rand(full.shape, device=device) < float(hp.get("mask_prob", 0.3))) & real
                    last = torch.rand(len(seqs), device=device) < 0.2      # some rows mask only the last item,
                    pick[last] = False                                     # which is what inference asks for
                    pick[:, -1] = pick[:, -1] | last | ~pick.any(1)
                    inp = full.masked_fill(pick, model.mask_id)
                    hm, tm = model(inp)[pick], full[pick] - 1
                else:
                    cut = [q[-(max_len + 1):] for q in seqs]
                    inp = _pad([q[:-1] for q in cut], max_len).to(device)
                    tgt = _pad([q[1:] for q in cut], max_len).to(device)
                    mask = tgt != 0
                    hm, tm = model(inp)[mask], tgt[mask] - 1
                if n_sampled:
                    neg = torch.from_numpy(rng.integers(0, ds.n_items, n_sampled)).to(device)
                    cand, inv = torch.unique(torch.cat([tm, neg]).cpu(), return_inverse=True)
                    cand, inv = cand.to(device), inv.to(device)
                    loss = F.cross_entropy(model.item_logits(hm, cand), inv[: len(tm)])
                else:
                    loss = F.cross_entropy(model.item_logits(hm), tm)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total, steps = total + loss.item(), steps + 1
        if ep % check:
            continue
        score = _ndcg10(scorer.ranks(vu, vh, vt))
        print(f"[{name}] epoch {ep:3d} loss {total / max(steps, 1):.4f} valid NDCG@10 {score:.4f}", flush=True)
        if score > best + 1e-5:
            best, best_ep, bad = score, ep, 0
            torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
        else:
            bad += 1
            if bad >= patience:
                break
    info = {"model": name, "hp": hp, "best_valid_ndcg10": best, "best_epoch": best_ep, "sampled_softmax": n_sampled}
    with open(os.path.join(out_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def load(name: str, out_dir: str, ds: Dataset, device: Optional[str] = None) -> Scorer:
    device = auto_device(device)
    if name in TRAINING_FREE:
        hp = {}
        if os.path.exists(os.path.join(out_dir, "info.json")):
            with open(os.path.join(out_dir, "info.json")) as f:
                hp = json.load(f).get("hp") or {}
        return Scorer(name, ds, hp=hp)
    with open(os.path.join(out_dir, "info.json")) as f:
        hp = json.load(f)["hp"]
    model = build_model(name, ds, hp, device)
    model.load_state_dict(torch.load(os.path.join(out_dir, "model.pt"), map_location=device))
    return Scorer(name, ds, model, device)


def tune(name: str, ds: Dataset, out_root: str, grid: Optional[list[dict]] = None,
         common: Optional[dict] = None, **kw) -> dict:
    """Train every configuration of the grid and return the one with the best validation NDCG@10.
    `common` is merged into every configuration of the trained models (for example a longer max_len)."""
    best = None
    for n, hp in enumerate(grid or DEFAULT_GRID[name]):
        if name not in TRAINING_FREE and common:
            hp = {**common, **hp}
        path = os.path.join(out_root, name, f"cfg{n}")
        info_path = os.path.join(path, "info.json")
        if os.path.exists(info_path):                       # resumable
            with open(info_path) as f:
                info = json.load(f)
        else:
            info = fit(name, ds, path, hp, **kw)
        info["dir"] = path
        if best is None or info["best_valid_ndcg10"] > best["best_valid_ndcg10"]:
            best = info
    return best
