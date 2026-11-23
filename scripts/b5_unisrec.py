"""B5 (docs/18): UniSRec (Hou et al., KDD 2022) as the text-transfer baseline, on our users and with our metric.

Official model class, hyperparameters and pretrained weights (third_party/UniSRec at 05aa5cba: props/UniSRec.yaml and
props/finetune.yaml; UniSRec-FHCKM-300.pth; encoder and positions frozen, the README's fine-tuning default). Our data loop
replaces RecBole's loaders so that training targets, model selection and ranking are those of our sequence baselines
(decision4rec.models.baselines.fit and Scorer.ranks):
- train: each user's sequence without its last two items; every prefix of at least one item predicts the next item, history
  cut to the last 50 (the checkpoint's MAX_ITEM_LIST_LENGTH), as UniSRec's own atomic files;
- validation and early stopping: the 10,000 validation users of baselines.fit (seed 13), NDCG@10 over the whole catalogue,
  history excluded, ties counted against, patience 10 (finetune.yaml), at most 300 epochs, Adam lr 1e-3, batch 2048
  (RecBole defaults);
- test: the 5,000 test cases of <c>_rtm, same rank;
- item text: Dataset.item_text, CLS of google-bert/bert-base-uncased (512 tokens), as UniSRec's feat1CLS.

  python scripts/b5_unisrec.py export <c> [<c> ...]     on .34 (main .venv): runs/small/<c>/unisrec/{data.npz,feat1CLS.npy}
  python scripts/b5_unisrec.py fit <dir> <out> [--stage transductive_ft|inductive_ft] [--scratch] [--smoke]
        needs torch, recbole and the UniSRec code (UNISREC_DIR) and weights (UNISREC_WEIGHTS); writes model.pt,
        info.json, test_ranks.json
  python scripts/b5_unisrec.py selfcheck                batch builder and rank rule against naive versions
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
UNI = os.environ.get("UNISREC_DIR", os.path.join(HERE, "..", "third_party", "UniSRec"))
WEIGHTS = os.environ.get("UNISREC_WEIGHTS", os.path.join(HERE, "..", "data", "raw", "unisrec", "UniSRec-FHCKM-300.pth"))
BERT = "google-bert/bert-base-uncased"
L = 50
CAP_ABOVE, CAP = 5_000_000, 50   # shortcut: compute cap, only ML-32M (15.6M prefixes) exceeds it; all prefixes elsewhere


def flat(seqs):
    off = np.zeros(len(seqs) + 1, dtype=np.int64)
    off[1:] = np.cumsum([len(s) for s in seqs])
    return np.fromiter((i for s in seqs for i in s), dtype=np.int32, count=int(off[-1])), off


def windows(fl, off, rows, ends, n=L):
    """Right-padded (len(rows), n) ids + 1 of items [max(0, end - n), end) of each row; lengths."""
    ln = np.minimum(ends, n)
    j = np.arange(n)
    ok = j[None, :] < ln[:, None]
    idx = np.where(ok, (off[rows] + ends - ln)[:, None] + j[None, :], 0)
    return np.where(ok, fl[idx].astype(np.int64) + 1, 0), ln.astype(np.int64)


def epoch_examples(ex_u, rng, cap):
    """Shuffled example indices of one epoch: all, or at most cap random prefixes per user (ex_u grouped by user)."""
    if cap is None:
        return rng.permutation(len(ex_u))
    o = np.lexsort((rng.random(len(ex_u)), ex_u))
    start = np.r_[0, np.flatnonzero(np.diff(ex_u[o])) + 1]
    pos = np.arange(len(o)) - np.repeat(start, np.diff(np.r_[start, len(o)]))
    return rng.permutation(o[pos < cap])


def rank_rows(sc, hist, tgt):
    """sc: torch (B, n_items) in our item order. Rank as Scorer.ranks: history to -inf, ties counted against the target."""
    import torch
    rows = torch.tensor([r for r, h in enumerate(hist) for _ in h], dtype=torch.long, device=sc.device)
    cols = torch.tensor([int(i) for h in hist for i in h], dtype=torch.long, device=sc.device)
    if len(rows):
        sc[rows, cols] = float("-inf")
    ts = sc.gather(1, torch.as_tensor(tgt, dtype=torch.long, device=sc.device).unsqueeze(1))
    return ((sc > ts).sum(1) + (sc == ts).sum(1)).cpu().numpy()


def ndcg10(r):
    return float(np.mean(np.where(r <= 10, 1.0 / np.log2(r + 1.0), 0.0)))


# ---- export (on .34) ---------------------------------------------------------

def export(c):
    import torch
    from transformers import AutoModel, AutoTokenizer

    from decision4rec.data.schema import Dataset, load_cases
    from decision4rec.experiments.runner import case_paths, load_config
    from decision4rec.models.baselines import _valid_set
    cfg = load_config(f"configs/small/{c}_rtm.yaml")
    ds = Dataset.load(cfg["dataset"])
    out = f"{cfg['out']}/unisrec"
    os.makedirs(out, exist_ok=True)
    test_file = case_paths(cfg)["test"]
    if os.path.exists(test_file) and not os.path.exists(f"{out}/data.npz"):   # cases exist once the baselines ran
        vu, vh, vt = _valid_set(ds, np.random.default_rng(13), 10000)      # the users and rng of baselines.fit
        cases = load_cases(test_file)
        tf, to = flat([[int(x[0]) for x in s[:-2]] for s in ds.seqs])
        vf, vo = flat(vh)
        sf, so = flat([q.history for q in cases])
        np.savez(f"{out}/data.npz", n_items=ds.n_items, train_flat=tf, train_off=to, valid_flat=vf, valid_off=vo,
                 valid_tgt=np.array(vt, dtype=np.int64), test_flat=sf, test_off=so,
                 test_tgt=np.array([q.target for q in cases], dtype=np.int64), test_cid=np.array([q.cid for q in cases]))
        info = {"scenario": c, "dataset": cfg["dataset"], "n_items": ds.n_items, "n_users": ds.n_users,
                "train_examples": int(sum(max(0, len(s) - 3) for s in ds.seqs)), "valid_users": len(vu),
                "test_cases": len(cases), "test_file": test_file}
        json.dump(info, open(f"{out}/export.json", "w"), indent=1)
        print(json.dumps(info), flush=True)
    elif not os.path.exists(test_file):
        print(f"{c}: no test cases yet ({test_file}), data.npz later", flush=True)
    if os.path.exists(f"{out}/feat1CLS.npy"):
        return
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok, bert = AutoTokenizer.from_pretrained(BERT), AutoModel.from_pretrained(BERT).eval().to(dev)
    texts = [ds.item_text(i) for i in range(ds.n_items)]
    order = np.argsort([len(t) for t in texts])
    feats = np.zeros((ds.n_items, 768), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(order), 64):
            b = order[s: s + 64]
            enc = tok([texts[i] for i in b], padding=True, truncation=True, max_length=512, return_tensors="pt").to(dev)
            feats[b] = bert(**enc).last_hidden_state[:, 0].float().cpu().numpy()
    np.save(f"{out}/feat1CLS.npy", feats)
    print(f"{c}: feat1CLS {feats.shape} from {BERT} in {time.time() - t0:.0f} s", flush=True)


# ---- fit (anywhere with torch + recbole + UniSRec code) --------------------------

def fit(a):
    import torch
    import yaml
    for k, v in (("float_", np.float64), ("complex_", np.complex128), ("unicode_", np.str_)):
        if k not in np.__dict__:
            setattr(np, k, v)        # shortcut: RecBole 1.2.1 aliases names removed in NumPy 2; drop with a fixed RecBole
    sys.path.insert(0, UNI)
    from recbole.config import Config
    from recbole.data.interaction import Interaction
    from recbole.utils import init_seed
    from recbole.utils.enum_type import EvaluatorType, InputType, ModelType
    from unisrec import UniSRec
    import config as uni_config

    D = np.load(os.path.join(a.dir, "data.npz"))
    feats = np.load(os.path.join(a.dir, "feat1CLS.npy"))
    n = int(D["n_items"])
    assert feats.shape == (n, 768)
    tf, to = D["train_flat"], D["train_off"]
    vf, vo, vt = D["valid_flat"], D["valid_off"], D["valid_tgt"]
    sf, so, st, cid = D["test_flat"], D["test_off"], D["test_tgt"], D["test_cid"]
    if a.smoke:
        vo, vt = vo[:301], vt[:300]
        so, st, cid = so[:201], st[:200], cid[:200]
    # RecBole writes CUDA_VISIBLE_DEVICES = gpu_id (default "0") whatever the caller set: on a shared machine that moved a
    # job to another user's GPU (.13, 2026-11-14). The caller's setting is passed through and checked; unset means CPU.
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    cfg = Config(model=UniSRec, dataset=os.path.basename(os.path.normpath(a.dir.replace("/unisrec", ""))),
                 config_file_list=[os.path.join(UNI, "props", "UniSRec.yaml"), os.path.join(UNI, "props", "finetune.yaml")],
                 config_dict={"train_stage": a.stage, "gpu_id": vis, "MAX_ITEM_LIST_LENGTH": L, "show_progress": False})
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == vis, (vis, os.environ.get("CUDA_VISIBLE_DEVICES"))
    dev = torch.device("cuda" if vis and torch.cuda.is_available() else "cpu")
    cfg["device"] = dev
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(0)
        print(f"device {p.name} pci bus {getattr(p, 'pci_bus_id', -1):02x} (CUDA_VISIBLE_DEVICES={vis})", flush=True)
    init_seed(cfg["seed"], cfg["reproducibility"])
    plm = torch.nn.Embedding(n + 1, 768, padding_idx=0)
    plm.weight.requires_grad = False
    plm.weight.data[1:] = torch.from_numpy(feats)
    plm.weight.data[0] = 0
    model = UniSRec(cfg, type("Stub", (), {"num": lambda self, f: n + 1, "plm_embedding": plm})())
    info = {"stage": a.stage, "scratch": a.scratch, "n_items": n, "lr": cfg["learning_rate"],
            "batch": cfg["train_batch_size"], "max_epochs": cfg["epochs"], "patience": cfg["stopping_step"],
            "seed": cfg["seed"]}
    if not a.scratch:
        with torch.serialization.safe_globals([uni_config.Config, EvaluatorType, InputType, ModelType, UniSRec,
                                               yaml.loader.FullLoader, torch.device]):
            ck = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        miss, unexp = model.load_state_dict(ck["state_dict"], strict=False)
        assert not unexp, unexp
        for p in list(model.position_embedding.parameters()) + list(model.trm_encoder.parameters()):
            p.requires_grad = False          # fix_enc, the README default
        info.update({"weights": os.path.basename(WEIGHTS), "weights_sha256": hashlib.sha256(open(WEIGHTS, "rb").read()).hexdigest(),
                     "missing_keys": miss, "frozen": ["position_embedding", "trm_encoder"]})
    model = model.to(dev)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"],
                           weight_decay=cfg["weight_decay"] or 0.0)
    lens = (to[1:] - to[:-1]).astype(np.int64)
    ex_u = np.repeat(np.arange(len(lens)), np.maximum(lens - 1, 0))
    ex_t = np.concatenate([np.arange(1, m) for m in lens if m >= 2]).astype(np.int64)
    rng = np.random.default_rng(cfg["seed"])
    bs = cfg["train_batch_size"]
    cap = CAP if len(ex_u) > CAP_ABOVE else None
    info["max_prefixes_per_user_per_epoch"] = cap

    def evaluate(fl, off, tgt):
        model.eval()
        out = []
        ln_all = (off[1:] - off[:-1]).astype(np.int64)
        assert ln_all.min() >= 1
        with torch.no_grad():
            for s in range(0, len(tgt), 1024):
                rows = np.arange(s, min(len(tgt), s + 1024))
                seq, ln = windows(fl, off, rows, ln_all[rows])
                inter = Interaction({model.ITEM_SEQ: torch.from_numpy(seq), model.ITEM_SEQ_LEN: torch.from_numpy(ln)}).to(dev)
                sc = model.full_sort_predict(inter)[:, 1:].float()
                out.append(rank_rows(sc, [fl[off[r]: off[r + 1]] for r in rows], tgt[rows]))
        return np.concatenate(out).astype(float)

    os.makedirs(a.out, exist_ok=True)
    best, best_ep, bad, t0 = -1.0, 0, 0, time.time()
    for ep in range(1, (2 if a.smoke else cfg["epochs"]) + 1):
        model.train()
        perm = epoch_examples(ex_u, rng, cap)[: 20 * 256 if a.smoke else None]
        tot, steps = 0.0, 0
        for s in range(0, len(perm), 256 if a.smoke else bs):
            b = perm[s: s + (256 if a.smoke else bs)]
            seq, ln = windows(tf, to, ex_u[b], ex_t[b])
            pos = tf[to[ex_u[b]] + ex_t[b]].astype(np.int64) + 1
            inter = Interaction({model.ITEM_SEQ: torch.from_numpy(seq), model.ITEM_SEQ_LEN: torch.from_numpy(ln),
                                 model.POS_ITEM_ID: torch.from_numpy(pos)}).to(dev)
            loss = model.calculate_loss(inter)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot, steps = tot + loss.item(), steps + 1
        score = ndcg10(evaluate(vf, vo, vt))
        print(f"[unisrec {a.stage}{' scratch' if a.scratch else ''}] epoch {ep:3d} loss {tot / max(steps, 1):.4f} "
              f"valid NDCG@10 {score:.4f} ({time.time() - t0:.0f} s)", flush=True)
        if score > best + 1e-5:
            best, best_ep, bad = score, ep, 0
            torch.save(model.state_dict(), os.path.join(a.out, "model.pt"))
        else:
            bad += 1
            if bad >= cfg["stopping_step"]:
                break
    model.load_state_dict(torch.load(os.path.join(a.out, "model.pt"), map_location=dev, weights_only=True))
    r = evaluate(sf, so, st)
    json.dump({str(int(c)): float(x) for c, x in zip(cid, r)}, open(os.path.join(a.out, "test_ranks.json"), "w"))
    info.update({"best_valid_ndcg10": best, "best_epoch": best_ep, "epochs_run": ep, "train_examples": len(ex_u),
                 "test_cases": len(r), "test_ndcg10": ndcg10(r), "seconds": round(time.time() - t0), "smoke": a.smoke})
    json.dump(info, open(os.path.join(a.out, "info.json"), "w"), indent=1)
    print(f"test NDCG@10 {ndcg10(r):.4f} over {len(r)} cases; best epoch {best_ep}", flush=True)


def selfcheck():
    import torch
    rng = np.random.default_rng(0)
    seqs = [rng.integers(0, 30, rng.integers(1, 80)).tolist() for _ in range(200)]
    fl, off = flat(seqs)
    rows = rng.integers(0, len(seqs), 500)
    ends = np.array([rng.integers(1, len(seqs[r]) + 1) for r in rows])
    seq, ln = windows(fl, off, rows, ends)
    for k, (r, e) in enumerate(zip(rows, ends)):
        h = [i + 1 for i in seqs[r][max(0, e - L): e]]
        assert ln[k] == len(h) and seq[k, : len(h)].tolist() == h and not seq[k, len(h):].any()
    sc = torch.randn(len(seqs), 30)
    sc[:, 5] = sc[:, 6]                              # a tie
    tgt = rng.integers(0, 30, len(seqs))
    got = rank_rows(sc.clone(), [fl[off[i]: off[i + 1]] for i in range(len(seqs))], tgt)
    for i, s in enumerate(seqs):
        v = sc[i].numpy().copy()
        v[s] = -np.inf
        assert got[i] == (v > v[tgt[i]]).sum() + (v == v[tgt[i]]).sum()
    ex_u = np.repeat(np.arange(6), [1, 5, 2, 9, 3, 0])
    a1 = np.random.default_rng(1).permutation(len(ex_u))
    assert (epoch_examples(ex_u, np.random.default_rng(1), None) == a1).all()      # no cap: the old draw exactly
    k = epoch_examples(ex_u, np.random.default_rng(2), 3)
    assert len(set(k)) == len(k) and np.bincount(ex_u[k], minlength=6).tolist() == [1, 3, 2, 3, 3, 0]
    print("selfcheck ok", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["export", "fit", "selfcheck"])
    p.add_argument("args", nargs="*")
    p.add_argument("--stage", default="transductive_ft", choices=["transductive_ft", "inductive_ft"])
    p.add_argument("--scratch", action="store_true")
    p.add_argument("--smoke", action="store_true")
    a = p.parse_args()
    if a.cmd == "export":
        for c in a.args:
            export(c)
    elif a.cmd == "fit":
        a.dir, a.out = a.args
        fit(a)
    else:
        selfcheck()
