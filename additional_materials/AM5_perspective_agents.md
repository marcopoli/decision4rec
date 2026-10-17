# AM5. The perspective agents

Agents that propose items from one point of view of the user: complement, continuation, replenishment, loyalty and exploration. They are not part of the main pool. These results are exploratory, because several versions were tried on the same users.

The perspective agents: the versions tried so far. New hits: targets found by the perspective agents and by no other agent, as a share of the targets found by the rest of the pool (↑); NDCG@10 and its changes: ↑. v1, v2 and v3 on 1,000 validation users per domain; the pool results on the 2,000 development test users.

| Version | What the agents do | Result |
|---|---|---|
| v1 | A small language model (Qwen2.5-1.5B, few examples) imagines the next item from each point of view; the closest item is retrieved. | New hits: 1 to 3%. Three ideas per point of view instead of one change nothing. |
| v2 | The imagined items become features of the residual ranker. | NDCG@10 changes by -0.0022 to +0.0023, never significant; the same as a control without a language model. |
| v4 | Rules over the knowledge graph, no generation. | New hits: 5 to 10% in every domain. |
| v4 in the pool | The rule agents join the pool; the ranker has one hidden layer. | Steam: from 0.0786 to 0.0810 (+3%, p = 0.09). Other domains: no significant change. |
| v4 variants | Fixed places for each perspective among the 30; coarser kinds of items. | No change in how often the target is among the 30, or in the new hits. |
| v5a | v4, and the ranker knows which perspective explained the user's own past steps. | Steam: 0.0820 (+4%, p = 0.016). The same ranker without perspectives: 0.0781. |
| v4 + v5a with the decision model | The trained 0.8B decision model (trained without perspective agents) reads the pool of v5a, with the perspective agents in its "found by" lines. | Steam: fused 0.0834, 4% above the same model without perspectives and 6% above SASRec (p = 0.17). Other domains: -2 to +3%. Read alone, the decision model does not change (≤ 1%). |
| v4 + v5a, decision model trained on it | The 0.8B decision model trained on records of the v5a pool, so that it has seen the perspective agents in its "found by" lines. | Against Decision4Rec-Full: fused +5% on Musical Instruments and +4% on Steam (p = 0.021 for both), -2 to 0% elsewhere; most of it comes from the pool. Read alone, +5% on Musical Instruments (p = 0.007) against the model trained without perspectives. |
| v3 | Five perspective questions asked to the decision model in the same request as the next choice (trained 0.8B; released 4B); answers as features of the ranker. | NDCG@10 changes by -0.0025 to +0.0016 against the fused score, never significant, in all four domains and with both models. |
| v5 | The decision model is also trained to answer which kind of choice the user makes next, labelled by the rules of v4 on the real next item; its answer weights the perspective agents of each candidate. | As a reader of the pool it equals Decision4Rec-Full (fused -0.1% to +2%, never significant; read alone +6% on Musical Instruments, p = 0.002). Its answers add nothing to the order: NDCG@10 changes by -0.0006 to +0.0012 against the fused score, never significant. |
| v6 | A profile statement in the state. | ongoing (AM9) |

**Summary.** The perspectives that help are the rules over the knowledge graph (v4, v5a), not the generated ones (v1, v2). They help most on Steam, where the other agents gain least. On the same perspective pool, a decision model trained on records that contain the perspective agents gains 5% read alone on Musical Instruments (p = 0.007) and changes by -2 to +3% elsewhere (p >= 0.16); its fused score changes by -1 to +2% (p >= 0.21). Compared with Decision4Rec-Full, the whole system is 5% better on Musical Instruments and 4% better on Steam (p = 0.021 for both), so most of this gain comes from the pool.
