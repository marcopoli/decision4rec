"""Backend conformance test. Run it before any experiment: it fails fast on protocol mismatches."""

from __future__ import annotations

import time

from ..decision.base import CachedBackend, build_backend
from ..decision.types import OUTSIDE_KEY, Choice, Noul, Score
from .runner import load_config

_STATE = ("User interaction history (oldest to newest):\n"
          "1. The Legend of Zelda: Breath of the Wild | Video Games > Nintendo Switch > Games\n"
          "2. Super Mario Odyssey | Video Games > Nintendo Switch > Games\n"
          "3. Mario Kart 8 Deluxe | Video Games > Nintendo Switch > Games")
_INSTR = "Based on the user's interaction history, which candidate item is the user most likely to interact with next?"
_OPTS = (("o01", "Excel 2019 for Dummies | Books > Computers"),
         ("o02", "Super Smash Bros. Ultimate | Video Games > Nintendo Switch > Games"),
         ("o03", "Garden Hose 50 ft | Patio, Lawn and Garden"),
         ("o04", "USB-C Charging Cable | Electronics > Accessories"))


def check_backend(cfg_path: str, backend_path=None) -> bool:
    cfg = load_config(cfg_path, backend_path)
    bcfg = {k: v for k, v in cfg["backend"].items() if k != "cache"}
    backend = build_backend(bcfg)
    assert not isinstance(backend, CachedBackend)
    ok = True

    def line(tag: str, msg: str) -> None:
        print(f"[{tag}] {msg}", flush=True)

    choice = Choice(_INSTR, _OPTS + ((OUTSIDE_KEY, "None of the listed candidates"),))
    questions = {"pick": choice,
                 "like": Noul("Will the user interact with this item next? Item: Super Smash Bros. Ultimate"),
                 "fit": Score("How well does 'Garden Hose 50 ft' match this user's interests?",
                              ("not at all", "somewhat", "very well"))}
    try:
        t0 = time.perf_counter()
        d = backend.decide(_STATE, questions)
        wall = (time.perf_counter() - t0) * 1000
    except Exception as e:
        line("FAIL", f"request failed: {e}")
        return False
    p = d.answers["pick"].probs
    total = sum(p.values())
    if abs(total - 1.0) > 1e-2:
        ok = False
        line("FAIL", f"choice probabilities sum to {total:.4f}")
    else:
        line("PASS", f"schema: choice sums to {total:.4f}, noul={d.answers['like'].p:.3f}, "
                     f"score={d.answers['fit'].value:.2f}, tokens={d.input_tokens}, wall={wall:.0f} ms")
    best = max(p, key=p.get)
    line("PASS" if best == "o02" else "WARN", f"sanity: top option is {best} (expected o02), p={p[best]:.3f}, "
                                               f"outside={p[OUTSIDE_KEY]:.3f}")

    d2 = backend.decide(_STATE, questions)
    diff = max(abs(p[k] - d2.answers["pick"].probs[k]) for k in p)
    line("PASS" if diff < 1e-6 else ("WARN" if diff < 1e-3 else "FAIL"),
         f"determinism: max probability difference between two identical requests = {diff:.2e}")
    ok = ok and diff < 1e-3

    alone = backend.decide(_STATE, {"pick": choice}).answers["pick"].probs
    iso = max(abs(p[k] - alone[k]) for k in p)
    line("PASS" if iso < 1e-3 else "WARN",
         f"question isolation: asked alone vs with other questions, max difference = {iso:.2e} "
         "(above 1e-3 means answers depend on the other questions, keep one block design per request)")

    rev = Choice(_INSTR, tuple(reversed(_OPTS)))
    fwd = backend.decide(_STATE, {"pick": Choice(_INSTR, _OPTS)}).answers["pick"].probs
    bwd = backend.decide(_STATE, {"pick": rev}).answers["pick"].probs
    shift = max(abs(fwd[k] - bwd[k]) for k in fwd)  # keys travel with their items
    line("INFO", f"order sensitivity: reversing 4 options moves a probability by at most {shift:.3f}")

    big = Choice(_INSTR, tuple((f"o{i + 1:02d}", f"Candidate item number {i + 1} | Video Games") for i in range(100)))
    try:
        t0 = time.perf_counter()
        r = backend.decide(_STATE, {"pick": big})
        line("PASS", f"capacity: 100 options answered in {(time.perf_counter() - t0) * 1000:.0f} ms, "
                     f"sum={sum(r.answers['pick'].probs.values()):.4f}")
    except Exception as e:
        line("WARN", f"capacity: a 100-option choice failed ({e}); use block designs with m <= 20")
    backend.close()
    line("PASS" if ok else "FAIL", f"backend {backend.fingerprint()}")
    return ok
