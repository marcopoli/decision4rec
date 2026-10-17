# AM1. The event-sourced runtime

The modules of Decision4Rec are written as behaviors of the event-sourced runtime ActiveGraph (Nakajima, 2026). The source of truth is an event log that only grows; the working graph is computed from it, and small functions called behaviors react to events and produce new ones.

| Event | What the behaviors do before the next event |
|---|---|
| `interaction.added` | update the counts, the transitions (sequential rules) and the co-occurrences of the items it touches |
| `logs.updated` | the agents propose their lists; the pool keeps 30 candidates |
| `candidates.proposed` | the decision model answers; the Luce fit and the fusion give a ranking |
| `ranking.estimated` | calibration and risk control |
| `slate.emitted` | the recommendation is shown |

A counterfactual forks the log without one interaction and runs the same behaviors again; the difference between the two lists is the effect of that interaction. Because a decision model does not sample, replaying the log gives the same recommendation.

**Status.** In the released code the runtime runs an earlier version of the agent (one request per user, an outside option and an expansion round). The experiments of the paper run the modules of Decision4Rec-Full as a batch pipeline with the same contracts, and every result is computed from the stored answers of the decision models. Porting Decision4Rec-Full to the runtime, with a check that replayed users receive the same rankings as in the batch pipeline, is ongoing work (AM9).
