# AM11. Tests behind the explanations in the paper

Every explanation that the paper gives for a result is backed by one of these tests. The scripts are in `scripts/`; they read the stored runs and answers and need no new call to a decision model.

| Claim in the paper | Test | Result | Script |
|---|---|---|---|
| A released decision model lacks the behavior of the scenario (G1) | released Kev-4B with and without statements, on the 30 best items of SASRecF | read alone +19% (Video Games), +45% (Musical Instruments), +1% (Steam, n.s.) | `arm_pairs.py rt kev-4b rte kev-4b` |
| The other agents do not find more targets than SASRecF with a longer list | recall of the whole pool against the best items of SASRecF in the same number, validation users | SASRecF of the same size finds more in every domain (p <= 0.005) | `claim_tests.py size` |
| With a trained recommender, the pool gains from information, not from candidates | the same ranker and features over the best items of SASRecF, as many as the pool | same NDCG@10 (-2.8 to +2.4%, p >= 0.078) | configs `*_rtms.yaml`, `ranker_control.py` with `ALT=rtms` |
| The 0.8B decision model reads mostly the inputs of the ranker | R^2 of its within-list scores from the features of the ranker (gradient-boosted regression, held-out users) | 0.8B: 0.80 to 0.89; 4B: 0.67 to 0.82 (p < 0.001) | `claim_tests.py signals` |
| The "found by" lines are the main source of the agreement with the ranker | agreement without them, against without one family of statements | 0.67-0.71 to 0.22-0.42 at 0.8B, 0.57-0.66 to 0.35-0.42 at 4B (p < 0.001); single families at most 0.06 | `explain_probe.py` |
| The gain does not trade the long tail for the head | NDCG@10 by target popularity, paired tests within groups | above SASRecF in all 9 sparse groups, significant in 8 | `claim_tests.py pop` |
| On MovieLens-1M the new attributes change what the agents find | target found by each agent, plain against enriched | knowledge-graph agent 5.6 to 7.3% (p = 0.016), text agent 7.4 to 5.8% (p = 0.008), pool 65.3 to 64.4% (n.s.) | `claim_tests.py agents` |
| "Not significantly different" claims | 95% paired bootstrap intervals | e.g. Decision4Rec-KG against SASRecF on Video Games: -9.5 to +15.6% | `equivalence_ci.py` |
| Gradient-boosted ranker control | LambdaMART on the features of the residual ranker | not better than the linear ranker (-1.4 to +3.8%, p >= 0.25) | `gbdt_ranker.py` |
| Rankings do not depend on the Python process | the knowledge-graph agent under different hash seeds | before the fix, 0.15 to 0.25% of the users changed; after the fix, none | `tests/test_determinism.py` |
| Seed spread of the decision model | 0.8B model trained with three seeds | fused NDCG@10 within 3%, read alone within 7%; a committee of seeds does not help | `arm_pairs.py`, `ensemble_probe.py` |
| Memory of the items explains at most a few percent | the same requests with an empty history and without item titles (0.8B, six settings; 4B, MovieLens-1M) | 0.8B: alone -7 to 0%, fused -5 to +1%, all p >= 0.05; 4B ML-1M: titles -3% fused (p = 0.009, plain), history -2 to -3% (p ~ 0.05) | `arm_pairs.py rtmctl:rtm kev-0.8b-rtm rtmctl:rtm_prior kev-0.8b-rtm` |
| Scales are a choice of form, not the cause of transfer (G2) | a 0.8B decision model trained and read with raw counts instead of levels, six settings | never worse: alone 0 to +9%, fused -1 to +4%; only significant: +9% alone, Industrial and Scientific | `arm_pairs.py rtm kev-0.8b-rtm rtmraw kev-0.8b-rtmraw` |
| More candidates do not lift the two-stage ceiling | the trained 0.8B and 4B models read the 50 best items of the pool instead of 30, no new training | fused: 0.8B -2 to +4%, 4B -1 to +4%; only Steam 4B +4% (p = 0.045, not Holm) | `arm_pairs.py rtm kev-4b-rtm rtm50:rtm kev-4b-rtm` |
