"""Kev's trainer (`python -m kev.train`, same arguments) with a fixed offset per option added to the choice logits in
the loss: fusion-aware training of the pool reader.

A choice question of a record may carry "offset": {option key: value}, written by `export-choices --offsets` from the
fast ranker's score of each option (standardized within the options, times a weight). The loss becomes the
cross-entropy of softmax(logits + offset), so the decision model learns what the ranker does not already give, and
the fused score of Eq. (fuse) is what training optimizes. At inference nothing changes: the model answers without
offsets and the fusion step adds the ranker's score. Options without an offset (none, distractors) get 0.

Run inside Kev's environment, for example: cd ~/kev && uv run python <repo>/scripts/kev_offset_train.py --data ...
"""
from __future__ import annotations

import kev.data as kd
import kev.train as kt
import torch


def offset_vector(src_q: dict, keys: list) -> list | None:
    """The offsets of one question in the order of the internal option keys, or None if it has none."""
    off = src_q.get("offset")
    if off is None:
        return None
    return [float(off.get(k, 0.0)) for k in keys]


_materialize = kt.materialize
_question_loss = kt.question_loss


def materialize(req):
    rec = _materialize(req)
    for q, src in zip(rec["questions"], req["questions"].values()):
        v = offset_vector(src, q["keys"])
        if v is not None:
            q["offset"] = v
    return rec


def question_loss(z, q, dev, *args, **kw):
    if q.get("offset") is not None:
        z = z + torch.tensor(q["offset"], device=z.device, dtype=z.dtype)
    return _question_loss(z, q, dev, *args, **kw)


def install() -> None:
    """kev.train imported materialize by name and looks question_loss up at call time: patch both modules."""
    kt.materialize = kd.materialize = materialize
    kt.question_loss = question_loss


if __name__ == "__main__":
    install()
    print("kev_offset_train: the loss adds each option's stored offset to its logit", flush=True)
    kt.main()
