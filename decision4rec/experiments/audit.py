"""Choice-axiom audit of a decision backend on recommendation candidates.

For a set S and each alternative x in S two blocks are asked in one request:
A_x lists S minus x in a fixed order followed by x, and B_x lists S minus x in
the same order. Every remaining item keeps its position, so the only change is
the presence of x.

* Luce's axiom (IIA) predicts p_i(A_x) = p_i(B_x) * (1 - p_x(A_x)). The
  substitution deviation d_i(x) = log p_i(A_x) - log p_i(B_x) - log(1 - p_x(A_x))
  is zero under IIA and negative when x takes more than its proportional share from i.
* Regularity predicts p_i(B_x) >= p_i(A_x): removing an alternative never lowers the others.
* The similarity hypothesis of choice theory predicts d_i(x) decreases with the
  similarity between i and x, measured here on the knowledge graph.

Question isolation is measured on the same requests: for the first cases every block
A_x is asked again alone, and its probabilities are compared with those it received
inside the full request. Block designs send all blocks in one request, so this is the
property their low cost depends on.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..data.schema import Dataset
from ..decision.base import build_backend, decide_chunked
from ..eval.metrics import bootstrap_ci
from .runner import get_cases, load_config, make_prompt_builder

_EPS = 1e-9


def run_audit(cfg_path: str, n_cases: int = 200, block: int = 8, regularity_tol: float = 0.01,
              backend_path=None, n_isolation: int = 50) -> dict:
    cfg = load_config(cfg_path, backend_path)
    ds = Dataset.load(cfg["dataset"])
    cases = get_cases(cfg, ds)["test"][:n_cases]
    pb = make_prompt_builder(cfg, ds, need_kg=True)
    backend = build_backend(cfg["backend"])
    seed = int(cfg["eval"].get("seed", 13))
    out_dir = os.path.join(cfg["out"], cfg["name"])
    os.makedirs(out_dir, exist_ok=True)
    rows, per_case_slope, tokens, iso = [], [], 0, []
    with open(os.path.join(out_dir, "audit.jsonl"), "w") as f:
        for case in cases:
            rng = np.random.default_rng([seed, case.cid, 991])
            K = len(case.cands)
            size = min(block, K)
            pool = [j for j in range(K) if j != case.target_pos]
            S = rng.permutation(pool)[: size - (1 if case.target_pos >= 0 else 0)].tolist()
            if case.target_pos >= 0:
                S.append(case.target_pos)
            S = rng.permutation(S).tolist()
            state = pb.state(case)
            questions = {}
            for x in S:
                rest = [j for j in S if j != x]
                questions[f"A{x}"] = pb.choice(case, rest + [x])
                questions[f"B{x}"] = pb.choice(case, rest)
            d = decide_chunked(backend, state, questions)
            tokens += d.input_tokens
            if case.cid in {c.cid for c in cases[:n_isolation]}:
                for x in S:
                    alone = backend.decide(state, {f"A{x}": questions[f"A{x}"]}).answers[f"A{x}"].probs
                    full = d.answers[f"A{x}"].probs
                    iso.append((max(abs(alone[k] - full[k]) for k in full), float(np.mean([abs(alone[k] - full[k]) for k in full])),
                                max(full, key=full.get) != max(alone, key=alone.get)))
            xs, ys = [], []
            for x in S:
                rest = [j for j in S if j != x]
                pa = d.answers[f"A{x}"].probs
                pbb = d.answers[f"B{x}"].probs
                p_x = pa[f"o{len(rest) + 1:02d}"]
                for k, i in enumerate(rest):
                    key = f"o{k + 1:02d}"
                    dev = np.log(pa[key] + _EPS) - np.log(pbb[key] + _EPS) - np.log(1 - p_x + _EPS)
                    sim = pb.kg.similarity(case.cands[i], case.cands[x])
                    row = {"cid": case.cid, "i": i, "x": x, "dev": float(dev), "sim": float(sim),
                           "p_a": float(pa[key]), "p_b": float(pbb[key]), "p_x": float(p_x),
                           "reg_violation": bool(pbb[key] < pa[key] - regularity_tol)}
                    rows.append(row)
                    xs.append(sim)
                    ys.append(dev)
                    f.write(json.dumps(row) + "\n")
            xs, ys = np.array(xs), np.array(ys)
            if xs.std() > 1e-9:
                per_case_slope.append(float(np.polyfit(xs, ys, 1)[0]))
    backend.close()
    dev = np.array([r["dev"] for r in rows])
    sim = np.array([r["sim"] for r in rows])
    slopes = np.array(per_case_slope)
    lo, hi = bootstrap_ci(slopes) if len(slopes) else (float("nan"), float("nan"))
    thirds = np.quantile(sim, [1 / 3, 2 / 3]) if len(sim) else [0, 0]
    buckets = {"low": sim <= thirds[0], "mid": (sim > thirds[0]) & (sim <= thirds[1]), "high": sim > thirds[1]}
    summary = {
        "backend": backend.fingerprint(), "n_cases": len(cases), "block": block, "n_pairs": len(rows),
        "mean_abs_deviation": float(np.abs(dev).mean()) if len(dev) else float("nan"),
        "mean_deviation": float(dev.mean()) if len(dev) else float("nan"),
        "regularity_violation_rate": float(np.mean([r["reg_violation"] for r in rows])) if rows else float("nan"),
        "similarity_slope_mean": float(slopes.mean()) if len(slopes) else float("nan"),
        "similarity_slope_ci95": [lo, hi],
        "pearson_dev_sim": float(np.corrcoef(sim, dev)[0, 1]) if len(dev) > 2 and sim.std() > 0 else float("nan"),
        "deviation_by_similarity": {k: float(dev[m].mean()) if m.any() else float("nan") for k, m in buckets.items()},
        "tokens_per_case": tokens / max(len(cases), 1),
        "isolation": {"n_questions": len(iso),
                      "mean_abs_difference": float(np.mean([m for _, m, _ in iso])) if iso else float("nan"),
                      "p95_max_abs_difference": float(np.quantile([x for x, _, _ in iso], 0.95)) if iso else float("nan"),
                      "max_abs_difference": float(max(x for x, _, _ in iso)) if iso else float("nan"),
                      "top1_changed_rate": float(np.mean([t for _, _, t in iso])) if iso else float("nan"),
                      "reading": "probability differences of the same block asked alone and inside the full request"},
        "reading": "IIA holds if deviations are near zero. A negative slope means similar alternatives "
                   "substitute for each other more than Luce's axiom allows.",
    }
    with open(os.path.join(out_dir, "audit.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary
