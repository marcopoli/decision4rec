"""A LlamaRec-style reranker trained on the records of the decision model (Roadmap R17).

LlamaRec (Yue et al., 2023) trains a causal language model with low-rank adapters to rerank the candidates of a
first stage: the prompt lists the history and the candidates with letter labels, and the next-token probabilities of
the labels, read in one forward pass, give a distribution over the candidates. This is the closest published design
to a decision model. Here it reads exactly what the decision model reads (the records of export-choices: state,
30 options with text, statements and "found by" lines, label = the real next item), with the same base model family
as Kev, the same adapter rank and the same training budget (one epoch, 8 records per step, OneCycle schedule,
gradient norm clipped at 1). The prompt is built by decision4rec.decision.logit.LogitBackend itself, so the trained model
is evaluated through the same pipeline (backend kind `logit`) as any decision model, with two orders per request and
the same fusion. What differs is the interface: one sequence with labels, read by a generative model, instead of
typed questions with isolated options and a choice head.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Optional

import torch
import torch.nn.functional as F

from ..decision.logit import LogitBackend
from ..decision.types import Choice


class LoRALinear(torch.nn.Module):
    """y = W x + (alpha / r) B A x with W frozen; A and B are kept in float32."""

    def __init__(self, base: torch.nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base, self.scale = base, alpha / r
        dev = base.weight.device
        self.A = torch.nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype)) * self.scale

    def merged(self) -> torch.nn.Linear:
        w = self.base.weight.data.float() + self.scale * (self.B.data.float() @ self.A.data.float())
        self.base.weight.data = w.to(self.base.weight.dtype)
        return self.base


def add_lora(model, r: int = 16, alpha: float = 32.0) -> int:
    """Freeze the model and put adapters on every linear layer of its decoder layers (attention, linear attention
    and MLP projections; not the embeddings or the output head), as Kev's adapters on all targets."""
    for p in model.parameters():
        p.requires_grad_(False)
    n = 0
    for layer in model.get_decoder().layers:
        for parent in list(layer.modules()):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(parent, name, LoRALinear(child, r, alpha))
                    n += 1
    return n


def merge_lora(model) -> int:
    n = 0
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, name, child.merged())
                n += 1
    return n


def shrink_options(rec: dict, k: int, rng: random.Random) -> dict:
    """The record with its question cut to the chosen option and k - 1 others drawn from the same list, in random order
    (R17 v3: a 0.8B model learns the label read-out over 5 options but not over 30, logs/probe/llm_pilot_k.py)."""
    q = rec["questions"]["next"]
    keys = list(q["criteria"])
    if k <= 0 or k >= len(keys):
        return rec
    keep = [q["label"]] + rng.sample([key for key in keys if key != q["label"]], k - 1)
    rng.shuffle(keep)
    return {**rec, "questions": {"next": dict(q, criteria={key: q["criteria"][key] for key in keep})}}


def encode(backend: LogitBackend, rec: dict) -> tuple[list[int], list[list[int]], int]:
    """Token ids of the prompt of one record exactly as LogitBackend.decide builds it for one question, the token
    ids of each option label, and the index of the chosen option."""
    q = rec["questions"]["next"]
    choice = Choice(instructions=q["instructions"], options=tuple(q["criteria"].items()))
    pre, post = backend._frame(rec["state"])
    text, labs = backend._question_text(choice)
    ids = backend.tok.encode(pre, add_special_tokens=True) + backend.tok.encode(text + post, add_special_tokens=False)
    return ids, labs, list(q["criteria"]).index(q["label"])


def label_scores(model, ids: list[int], labs: list[list[int]]) -> torch.Tensor:
    """Log-scores of the options: logsumexp of the next-token logits of each label's tokens, as LogitBackend."""
    dev = next(model.parameters()).device
    x = torch.tensor([ids], device=dev)
    try:
        logits = model(input_ids=x, use_cache=False, logits_to_keep=1).logits
    except TypeError:                                   # models without logits_to_keep
        logits = model(input_ids=x, use_cache=False).logits
    row = logits[0, -1].float()
    return torch.stack([torch.logsumexp(row[torch.tensor(t, device=dev)], 0) for t in labs])


def train(data: str, base: str, out: str, revision: Optional[str] = None, lr: float = 1e-4, lora: int = 16,
          alpha: float = 32.0, accum: int = 8, epochs: int = 1, seed: int = 0, max_len: int = 6144,
          max_records: int = 0, dtype: str = "bf16", device: Optional[str] = None, checkpointing: bool = True,
          tokenizer=None, model=None, log_every: int = 10, options: int = 0) -> dict:
    """Train on the records of `data` and save the merged model in `out` (a folder the logit backend loads).
    options > 0 cuts every question to the chosen option and options - 1 others (shrink_options); the model is then
    read over blocks of the same size (an arm with m = options)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(
            base, revision=revision, dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[dtype])
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    n_lora = add_lora(model, lora, alpha)
    if checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    backend = LogitBackend(base, tokenizer=tokenizer, lm=model, use_chat_template=False, prefix_cache=False,
                           batch_size=1)
    model.train()
    with open(data, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    rng = random.Random(seed)
    opt_rng = random.Random(seed + 1)
    params = [p for p in model.parameters() if p.requires_grad]
    n_steps = epochs * math.ceil((min(len(recs), max_records) if max_records else len(recs)) / accum)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(n_steps, 1), pct_start=0.1)
    step, seen, skipped, t0 = 0, 0, 0, time.time()
    run_loss, run_n, curve = 0.0, 0, []
    for ep in range(epochs):
        order = list(range(len(recs)))
        rng.shuffle(order)
        if max_records:
            order = order[:max_records]
        for k, i in enumerate(order):
            ids, labs, y = encode(backend, shrink_options(recs[i], options, opt_rng))
            if len(ids) > max_len:
                skipped += 1
            else:
                scores = label_scores(model, ids, labs)
                loss = F.cross_entropy(scores[None], torch.tensor([y], device=scores.device))
                (loss / accum).backward()
                run_loss, run_n, seen = run_loss + float(loss.detach()), run_n + 1, seen + 1
            if (k + 1) % accum == 0 or k == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0 and run_n:
                    curve.append(round(run_loss / run_n, 4))
                    print(f"ep{ep} step {step}/{n_steps} loss {run_loss / run_n:.3f} "
                          f"{(time.time() - t0) / max(seen, 1):.3f}s/rec", flush=True)
                    run_loss, run_n = 0.0, 0
    model.eval()
    n_merged = merge_lora(model)
    metrics = {"records_seen": seen, "records_skipped_too_long": skipped, "optimizer_steps": step,
               "wall_seconds": round(time.time() - t0, 1), "adapters": n_lora, "merged": n_merged,
               "loss_every_10_steps": curve,
               "args": {"data": data, "base": base, "revision": revision, "lr": lr, "lora": lora, "alpha": alpha,
                        "accum": accum, "epochs": epochs, "seed": seed, "max_len": max_len,
                        "max_records": max_records, "dtype": dtype, "options": options}}
    if out:
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        with open(os.path.join(out, "training_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=1)
    return metrics
