"""Decision4Rec-Full as behaviors on the ActiveGraph runtime (C1, R23; docs/18).

The batch runner (experiments/runner.py) computes the reported numbers; this module runs the same system as an event-driven
agent, so that a deployed form exists and the stream replay (C2) can update evidence between requests.

    request.added       -> pool     : the multi-agent pool ranks the catalogue for the user's history; the 30 best and
                                      their first-stage scores go on the session object
    candidates.proposed -> decide   : the decision model reads the 30 with item text, statements and the agents that found
                                      each item (the rtm arm: block choices, Luce observations)
    choices.observed    -> estimate : Luce fit of the observations (utilities theta)
    ranking.estimated   -> fuse     : z(theta) and z(first-stage score) fused with the weights fitted on validation users;
                                      the slate is the top 10

Every step calls the same functions as the batch runner (PoolRetriever, run_arm, fit_luce, the fusion weights), so the
equivalence check (scripts/c1_equivalence.py) compares top-10 lists of the two forms user by user.
"""
from __future__ import annotations

import numpy as np

from ..data.schema import Case
from ..experiments.asc import _z          # the batch fusion's standardisation (scripts/fusion_table.py)
from ..operators.luce import Obs, fit_luce
from ..operators.rank import run_arm


class FullAgent:
    def __init__(self, retriever, arm, backend, pb, weights, K: int = 30, slate: int = 10, seed: int = 13, block: int = 64):
        self.retriever, self.arm, self.backend, self.pb, self.block = retriever, arm, backend, pb, block
        self.w = np.asarray(weights, dtype=float)
        self.K, self.slate, self.seed = K, slate, seed

    # ---- behaviors ---------------------------------------------------------

    def _pool(self, event, graph, ctx):
        sid = event.payload["sid"]
        d = graph.get_object(sid).data
        # fixed block of 64 rows (the request padded with copies): float kernels differ with the batch shape, and a near
        # tie in the first-stage score (about 1e-6) would otherwise reorder the options of 0.2% of the requests (C1)
        idx, val = self.retriever.topk_users([d["user"]] * self.block, [d["history"]] * self.block, self.K)
        graph.patch_object(sid, {"cands": [int(i) for i in idx[0][: self.K]], "rscores": [float(v) for v in val[0][: self.K]]})
        graph.emit("candidates.proposed", {"sid": sid})

    def _decide(self, event, graph, ctx):
        sid = event.payload["sid"]
        d = graph.get_object(sid).data
        case = Case(cid=d["cid"], user=d["user"], history=d["history"], target=d["target"], cands=d["cands"],
                    rscores=d["rscores"], target_pos=d["cands"].index(d["target"]) if d["target"] in d["cands"] else -1,
                    split=d["split"], target_rating=0.0)
        rec = run_arm(case, self.arm, self.backend, self.pb, seed=self.seed)
        graph.patch_object(sid, {"obs": rec["obs"], "cached": rec.get("cached")})
        graph.emit("choices.observed", {"sid": sid})

    def _estimate(self, event, graph, ctx):
        sid = event.payload["sid"]
        d = graph.get_object(sid).data
        theta = fit_luce([Obs.from_json(o) for o in d["obs"]], len(d["cands"])).theta
        graph.patch_object(sid, {"theta": [float(t) for t in theta]})
        graph.emit("ranking.estimated", {"sid": sid})

    def _fuse(self, event, graph, ctx):
        sid = event.payload["sid"]
        d = graph.get_object(sid).data
        score = np.stack([_z(d["theta"]), _z(d["rscores"])], 1) @ self.w
        order = np.argsort(-score, kind="stable")[: self.slate]
        graph.patch_object(sid, {"score": [float(s) for s in score], "slate": [d["cands"][j] for j in order]})
        graph.emit("slate.ready", {"sid": sid})

    def behaviors(self) -> list:
        from activegraph import Behavior
        return [Behavior(name="pool", fn=self._pool, on=["request.added"]),
                Behavior(name="decide", fn=self._decide, on=["candidates.proposed"]),
                Behavior(name="estimate", fn=self._estimate, on=["choices.observed"]),
                Behavior(name="fuse", fn=self._fuse, on=["ranking.estimated"])]

    # ---- harness -------------------------------------------------------------

    def serve(self, case: Case) -> dict:
        """One request on a fresh graph: returns the session data (candidates, scores, slate) and the event types."""
        from activegraph import Event, Frame, Graph, Runtime
        graph = Graph()
        runtime = Runtime(graph, behaviors=self.behaviors(), frame=Frame(goal="recommend the next item"),
                          budget={"max_events": 100})
        sess = graph.add_object("session", {"cid": case.cid, "user": int(case.user), "history": [int(i) for i in case.history],
                                            "target": int(case.target), "split": case.split})
        graph.emit(Event(id=graph.ids.event(), type="request.added", payload={"sid": sess.id}, actor="harness",
                         timestamp=graph.clock.now()))
        runtime.run_until_idle()
        if runtime.errors:
            raise RuntimeError(f"behavior failed: {runtime.errors[0]}")
        data = dict(graph.get_object(sess.id).data)
        if data.get("slate") is None:
            raise RuntimeError("no slate emitted")
        data["events"] = [e.type for e in graph.events]
        return data
