"""Export fine-tuning records for open decision models (Kev `kev.train` JSONL format).

Supervision comes from ground-truth interactions only. Each record uses the
training prefix of a user who is not part of any evaluation case: the label is
the last training item, the history is everything before it. No output of a
hosted model is used, in line with the vendor's distillation clause.
"""

from __future__ import annotations

import json
import os
import zlib

import numpy as np

from ..data.reserved import reserved_users
from ..data.schema import Case, Dataset, load_cases
from ..decision.types import OUTSIDE_KEY
from ..eval import metrics as M
from ..prompts import INSTR_NOUL, PromptBuilder
from .runner import build_retriever, get_cases, load_config, make_prompt_builder


def _records(ds: Dataset, pb: PromptBuilder, retriever, users: list[int], block: int, outside_rate: float,
             rng: np.random.Generator, batch: int = 512):
    for s in range(0, len(users), batch):
        chunk = users[s: s + batch]
        hist = [[int(x[0]) for x in ds.seqs[u][:-3]] for u in chunk]
        top, _ = retriever.topk(hist, 50)
        for u, h, hard in zip(chunk, hist, top):
            target = int(ds.seqs[u][-3][0])
            hard = [int(i) for i in hard if int(i) != target]
            n_neg = block - 1
            negs = hard[: n_neg // 2 + n_neg % 2]
            while len(negs) < n_neg:
                j = int(rng.integers(0, ds.n_items))
                if j != target and j not in negs and j not in h:
                    negs.append(j)
            absent = rng.random() < outside_rate
            if absent:
                extra = next((i for i in hard if i not in negs), None)
                while extra is None:
                    j = int(rng.integers(0, ds.n_items))
                    extra = j if j != target and j not in negs and j not in h else None
                cands = negs + [extra]
            else:
                cands = negs + [target]
            order = rng.permutation(len(cands)).tolist()
            case = Case(cid=-1, user=u, history=h, target=target, cands=cands, rscores=[0.0] * len(cands),
                        target_pos=-1 if absent else len(cands) - 1)
            with_outside = absent or rng.random() < 0.5
            q = pb.choice(case, order, outside=with_outside)
            label = OUTSIDE_KEY if absent else f"o{order.index(len(cands) - 1) + 1:02d}"
            wire = q.to_wire()
            wire["label"] = label
            probe_pos = rng.random() < 0.5
            probe = target if probe_pos else negs[0]
            yield {"state": pb.state(case),
                   "questions": {"next": wire,
                                 "like": {"type": "noul", "label": bool(probe_pos),
                                          "instructions": INSTR_NOUL.format(item=pb.option(case, probe))}}}


def _case_users(cfg: dict) -> set:
    """Users of every evaluation case file of a config's output folder (any track, K, retriever or split), and the
    users reserved for the confirmation on fresh users (decision4rec.data.reserved), which no development step draws."""
    folder = os.path.join(cfg["out"], "cases")
    names = sorted(f for f in os.listdir(folder) if f.endswith(".jsonl")) if os.path.isdir(folder) else []
    return {c.user for name in names for c in load_cases(os.path.join(folder, name))} | reserved_users(cfg)


def export_choices(cfg_paths: list[str], out: str, n_per_dataset: int = 15000, n_heldout: int = 500,
                   seed: int = 13, min_hist: int = 2, batch: int = 512, in_pool_only: bool = True,
                   crossfit: bool = False, offsets: float = 0.0) -> dict:
    """Training records for a decision model that learns real next-item choices (Kev `kev.train` format).

    For every config, users are taken in random order. The target of a user is the second to last item and the
    history is everything before it. The trained first stage (SASRecF) learned every next-item step of all but
    the last two items of every user, so with an earlier target its top K would be the list of a model that
    already knows the answer: kept records would over-represent targets it memorised, a cue that is absent at
    test time. The second to last item was never a training target of the first stage (it only chose the
    checkpoint on some users). The users of every evaluation case file of the dataset are left out, so no
    validation target that fits a fusion weight is a training label; no test item of any user is read. The
    options are the top K of the config's first stage for that history (the same retriever that builds the
    evaluation cases, so the model learns to separate exactly the items it will see), in random order. The
    state and the options are rendered by the config's prompt settings, as at inference.

    Only records whose target is among the first stage's top K are kept (`in_pool_only`), so users are read
    until enough such records exist. Adding a missing target instead would make it the odd one out among items
    the first stage likes, and the model would learn to pick outliers; at test time the reranker only matters
    when the target is in the pool anyway. With `in_pool_only=False` a missing target replaces the last candidate.

    `crossfit` (multi-agent pool only) is for a scenario too small to leave users apart: MovieLens-1M has 6,040
    users, 3,000 are evaluated and the other 3,000 trained the pool's ranker. The records then come from those
    ranker users, split in two halves; each half is ranked by a copy of the pool whose ranker is trained on the
    other half only, so no record's candidates come from a ranker that learned its target (rule 3). The pool of
    the evaluation is not changed.
    """
    os.makedirs(out, exist_ok=True)
    parts: dict[str, list] = {"train": [], "heldout": []}
    info: dict = {"version": 3, "in_pool_only": in_pool_only, "target": "second to last item", "evidence": {},
                  "datasets": {}}
    for path in cfg_paths:
        cfg = load_config(path)
        ds = Dataset.load(cfg["dataset"])
        get_cases(cfg, ds)                       # the config's own evaluation users exist before they are left out
        excluded = _case_users(cfg)
        pb = make_prompt_builder(cfg, ds, need_kg=False)      # with an `evidence:` section, the store's statements
        retriever = build_retriever(cfg.get("retriever", {}), ds, cfg)
        # the options carry what the config's decision arm reads (`evidence: stats` adds the store's statements; the
        # store leaves out the last two items of every user, so a record never reads its own target)
        evidence = next((a.get("evidence", "none") for a in cfg["arms"] if a.get("method") != "retriever"), "none")
        info["evidence"][ds.name] = evidence
        K = int(cfg["eval"]["K"])
        rng = np.random.default_rng([seed, zlib.crc32(ds.name.encode())])
        rows, counts = [], {"in_top": 0, "read": 0}

        def collect(users, retr):
            for s in range(0, len(users), batch):
                if len(rows) >= n_per_dataset + n_heldout:
                    break
                chunk = users[s: s + batch]
                hist = [[int(x[0]) for x in ds.seqs[u][:-2]] for u in chunk]
                top, vals = retr.topk_users(chunk, hist, K + 1) if hasattr(retr, "topk_users") \
                    else retr.topk(hist, K + 1)
                for u, h, row, val in zip(chunk, hist, top, vals):
                    if len(rows) >= n_per_dataset + n_heldout:
                        break
                    target = int(ds.seqs[u][-2][0])
                    ranked = [int(i) for i in row]
                    counts["read"] += 1
                    counts["in_top"] += int(target in ranked[:K])
                    if in_pool_only and target not in ranked[:K]:
                        continue
                    cands = [i for i in ranked[:K] if i != target][: K - 1] + [target]
                    order = rng.permutation(len(cands)).tolist()
                    case = Case(cid=-1, user=u, history=h, target=target, cands=cands, rscores=[0.0] * len(cands),
                                target_pos=len(cands) - 1)
                    wire = pb.choice(case, order, evidence=evidence).to_wire()
                    wire["label"] = f"o{order.index(len(cands) - 1) + 1:02d}"
                    if offsets:
                        # fusion-aware training: the first stage's score, standardized within the options as in the
                        # fused score, enters the loss as a fixed offset (scripts/kev_offset_train.py)
                        score = dict(zip(ranked, (float(x) for x in val)))
                        z = M.zscore([score[i] for i in cands])
                        wire["offset"] = {k: round(offsets * float(z[j]), 4) for k, j in zip(wire["criteria"], order)}
                    rows.append(({"state": pb.state(case), "questions": {"next": wire}},
                                 {"dataset": ds.name, "user": u, "target": target, "options": len(cands)}))

        if crossfit:
            if not hasattr(retriever, "pool"):
                raise SystemExit("crossfit needs the multi-agent pool as first stage")
            from ..agents.pool import PoolRetriever, refit
            main = retriever.pool
            ranker_users = [u for u in rng.permutation(main.train_users).tolist()
                            if len(ds.seqs[u]) >= min_hist + 2 and u not in excluded]
            halves = (ranker_users[0::2], ranker_users[1::2])
            for k_half in (0, 1):
                other = refit(main, halves[1 - k_half], cfg.get("pool") or {})
                pb.pool = other                          # the "found by" lines of the ranking that is read
                collect(halves[k_half], PoolRetriever(other))
            pb.pool = main
            info.setdefault("crossfit", {})[ds.name] = [len(halves[0]), len(halves[1])]
        else:
            if hasattr(retriever, "pool"):               # users that trained the pool's ranker know their answers
                excluded = excluded | set(retriever.pool.train_users)
            users = [u for u in rng.permutation(ds.n_users).tolist()
                     if len(ds.seqs[u]) >= min_hist + 2 and u not in excluded]
            collect(users, retriever)
        read, in_top = counts["read"], counts["in_top"]
        held, train = rows[:n_heldout], rows[n_heldout:]
        parts["heldout"] += held
        parts["train"] += train
        with open(os.path.join(out, f"heldout_{ds.name}.jsonl"), "w", encoding="utf-8") as f:
            for rec, _ in held:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        info["datasets"][ds.name] = {"config": path, "train": len(train), "heldout": len(held), "options": K,
                                     "excluded_eval_users": len(excluded), "users_read": read,
                                     "target_in_first_stage_top_k": in_top / max(read, 1)}
        if offsets:
            info["offsets"] = offsets
    order = np.random.default_rng(seed).permutation(len(parts["train"])).tolist()
    parts["train"] = [parts["train"][i] for i in order]       # domains mixed within every batch
    for name, part in parts.items():
        with open(os.path.join(out, f"{name}.jsonl"), "w", encoding="utf-8") as f, \
                open(os.path.join(out, f"{name}.meta.jsonl"), "w", encoding="utf-8") as g:
            for rec, meta in part:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                g.write(json.dumps(meta) + "\n")
        info[name] = len(part)
    with open(os.path.join(out, "export_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    return info


def export_rectune(cfg_path: str, out: str, n_train: int = 20000, n_heldout: int = 1000, block: int = 10,
                   outside_rate: float = 0.15, item_chars: int = 0) -> dict:
    cfg = load_config(cfg_path)
    if item_chars:
        cfg["prompt"]["item_chars"] = item_chars
    ds = Dataset.load(cfg["dataset"])
    cases = get_cases(cfg, ds)
    eval_users = {c.user for split in cases.values() for c in split}
    # Exclude the users of every case file of this dataset, whatever the track or K they were built for.
    case_dir = os.path.join(cfg["out"], "cases")
    case_files = sorted(f for f in os.listdir(case_dir) if f.endswith(".jsonl"))
    for name in case_files:
        eval_users |= {c.user for c in load_cases(os.path.join(case_dir, name))}
    pb = make_prompt_builder(cfg, ds, need_kg=False)
    retriever = build_retriever(cfg.get("retriever", {}), ds)
    rng = np.random.default_rng(int(cfg["eval"].get("seed", 13)) + 7)
    users = [u for u in rng.permutation(ds.n_users).tolist()
             if u not in eval_users and len(ds.seqs[u]) >= 6][: n_train + n_heldout]
    os.makedirs(out, exist_ok=True)
    counts = {}
    for name, part in (("heldout", users[:n_heldout]), ("train", users[n_heldout:])):
        path = os.path.join(out, f"{name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            n = 0
            for rec in _records(ds, pb, retriever, part, block, outside_rate, rng):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
        counts[name] = {"path": path, "records": n}
    counts["excluded_eval_users"] = len(eval_users)
    counts["excluded_case_files"] = case_files
    with open(os.path.join(out, "export_info.json"), "w") as f:
        json.dump(counts, f, indent=1)
    return counts
