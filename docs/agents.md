# Extend the pool of agents

The multi-agent pool ([`decision4rec/agents/pool.py`](../decision4rec/agents/pool.py)) is where candidates come from.
Each agent proposes a ranked list of items from its own point of view; the pool is the union of the lists; a residual
ranker orders it and keeps the best 30 for the decision model. For each candidate the pool also remembers which agents
found it, and the decision model reads this as one more statement ("found by sequence model #3, co-purchase").

## The agents of the paper

| Name | Proposes | Depth |
|---|---|---|
| `sasrecf` | the trained recommender of the scenario (SASRec with item features), when there is one | `seq_depth` (50) |
| `evidence` | the fused score of the knowledge-graph generators | `depth` (20) |
| `after` | items that often follow the recent items (transitions) | 20 |
| `with` | items often bought with the user's items (co-purchase) | 20 |
| `text` | items whose text is closest to the recent items | 20 |
| `lately` | items popular in the last days | 20 |
| `kg` | items that share rare catalogue attributes with the recent items | 20 |
| `pop` | best sellers | 20 |

The trained recommender enters with its whole top 50, so the pool never loses a target that it finds, and the ranker
starts from its order: score = z(SASRecF) + λ·f(features), with f learned and λ chosen on users that are never
evaluated. With λ = 0 the pool gives back the order of the trained recommender.

## Adding an agent in three steps

**1. Name it.** Add a key to `AGENTS` and a readable name to `NAMES`; the name is what the decision model reads in
the "found by" line.

```python
AGENTS = ("sasrecf", "evidence", "after", "with", "text", "lately", "kg", "pop", "bundle")
NAMES = {..., "bundle": "bundles"}
```

**2. Produce its list.** In `AgentPool.lists`, add the ranked item indices the agent proposes for a history. Leave
out items already in the history and stop at `self.depth`:

```python
def lists(self, history, now, row):
    ...
    out["bundle"] = [j for j in self.bundles.complete(history) if j not in seen][: self.depth]
    return out
```

An agent can read anything the pool holds: the knowledge graph of the scenario (`self.store`, with statistics that
leave out the evaluation targets), the item graph (`self.kg`), the dataset (`self.ds`) or a model you load in
`__init__`. To keep the evaluation fair, an agent must be built only from training data: use `self.store` for
counts, or train your model on the same split as SASRecF.

**3. Retrain the ranker.** The features of the ranker are built automatically for every agent in `AGENTS` (whether the
agent found the item, and at which rank), so nothing else changes in the code. The trained ranker is cached under
`runs/<setting>/<domain>/pool/`; delete that file, or change the `pool:` section of the configuration (its values are
part of the cache name), and the next run trains the ranker again with the new agent:

```bash
rm runs/small/vg/pool/pool_sasrecf_u15000_s11.pt
decision4rec run --config configs/small/vg_rtm.yaml --backend configs/backends/kev-4b.yaml
```

## Measuring what an agent adds

Two numbers tell whether an agent is worth keeping, both on validation users before any test user is read:

- **new hits**: targets that the agent finds and no other agent finds, as a share of the targets the pool already
  finds (the perspective agents of the paper add 5 to 10%);
- **the change of NDCG@10 of the pool**, with the ranker retrained with and without the agent, compared with a paired
  randomization test.

`scripts/pool_probe.py` and `scripts/recall_probe.py` are the probes we used for the agents of the paper; adapt them to your agent.

## Agents on the ActiveGraph runtime

For interactive use, agents can also run as behaviours of the [ActiveGraph](https://github.com/yoheinakajima/activegraph)
runtime ([`decision4rec/agents/active.py`](../decision4rec/agents/active.py)): every proposal, statement and decision
is an event in the log, so a session can be replayed exactly and a belief that a new event touches is marked stale. The
batch pool above and the runtime agents produce the same candidates; the batch form is the one used for the
experiments of the paper.

## A decision model as an agent

An agent may itself ask a decision model, for example "is this item a complement of the last purchase?" (a `noul`
question per candidate). Use the same backend interface ([decision_models.md](decision_models.md)) and cache the
answers; the perspective agents of the paper (`decision4rec/agents/kgperspectives.py`) are an example of agents built
from rules of the knowledge graph instead.
