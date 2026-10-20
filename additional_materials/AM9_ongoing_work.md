# AM9. Ongoing work (status of 17 October 2026)

| Item | What it tests | Status |
|---|---|---|
| Confirmation on 10,000 fresh users per domain, with an analysis plan written before scoring, a split on global time, and SASRecF trained also on the targets that train the ranker and the decision model | that the development results are not the effect of our design choices, of the leave-one-out split or of extra training targets | users reserved; needs more computing power |
| Kev-27B trained on a fixed set of 8,000 records, with Kev-0.8B and Kev-4B trained again on the same records | whether the gain from 0.8B to 4B continues at 27B on the same data | 27B done (in the paper: +4 to +8% over the pool in all four domains, significant on Video Games and Steam); 0.8B and 4B trained on the same records: done (in the paper: on the same data the 27B model is 5 to 6% above the 0.8B model in three domains) |
| Kev-27B as released | how much of a strong released model remains once the state carries the statements | done (in the paper: on all users it beats Jev in all six settings and the released Kev-4B in four, but a trained Kev-0.8B beats it in four of six) |
| Kev-9B on the same 8,000 records | the shape of the size curve | done (in the paper: adds 1 to 5% to the pool, not significant; within 2% of the 4B model) |
| The 0.8B model on 2,000, 4,000, 8,000 and 16,000 records | whether more records help at a fixed size | queued |
| Raw counts instead of scales | gap G2: whether positions on scales are what lets one model serve a new scenario | done (in the paper: raw counts are never worse in six settings, so the scales are a choice of form, not the cause of transfer) |
| History-free prior and title-masked requests | how much of a result memorized item names or popular items explain | queued |
| Decision4Rec-Full with a decision model trained on all four domains | what training on the held-out domain adds, for the full system | queued |
| Decision4Rec-KG at 4B | whether a larger reader adds a clear gain to the candidates of the knowledge graph | done in all six settings (in the paper) |
| a stronger text encoder in the pool; a profile statement | whether better content reading and a profile help long histories | queued |
| Fifty options instead of thirty | whether the decision model can use the items that the ranker places between 31 and 50 | done without retraining (in the paper: fused -2 to +4% at 0.8B, -1 to +4% at 4B; one p < 0.05 of eight, not Holm); training on 50 options not done |
| Decision4Rec-Full inside the runtime | that the deployment form is the system that was evaluated | planned |
