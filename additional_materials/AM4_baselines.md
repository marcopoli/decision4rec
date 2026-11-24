# AM4. Training of the baselines

All trained baselines are our implementations, trained and evaluated with one protocol and scored over the whole catalogue, without the items of the history.

- **Protocol** (as in RecBole): at most 300 epochs, Adam with learning rate 1e-3, batches of 256 sequences, validation on the second-to-last item after every epoch, early stopping after 10 validations without improvement, model selection on validation NDCG@10. All models use the full softmax with cross-entropy, which makes SASRec stronger than training with sampled negatives (Klenitskiy and Vasilev, 2023).
- **Input:** the last 50 items of the history, or 200 on MovieLens-1M.
- **Grids:** 9 configurations for SASRec and SASRecF (embedding size 64, 128 or 256; dropout 0.2, 0.3 or 0.5; two or three blocks), 9 for GRU4Rec (size and dropout), 4 for BERT4Rec (size 64 or 128; masking probability 0.3 or 0.5). On all users the full grid changes the NDCG@10 of the best trained baseline by less than 3% on the Amazon domains and on Steam, so the development tables keep the models of the first two configurations there; on MovieLens-1M the tuned models are clearly stronger (SASRecF 0.1870 against 0.1759) and are used everywhere.
- **SASRecF:** SASRec whose item representation adds embeddings of up to 12 catalogue attributes (brand, category, platform; on MovieLens-1M genres, director, cast and the other Wikidata properties).
- **Training-free methods:** popularity; item kNN on co-occurrence within five items; sequential rules (Ludewig and Jannach, 2018) on the last one to three items; content kNN on the text vectors; and a mix of transitions and text (the items within two transition steps of the last two items, ranked by text similarity to the last item), after Han et al. (2026).
- **PrepRec** (Wang et al., 2024): the authors' code and settings, trained on Video Games and on Musical Instruments (the timestamps of our Steam data are only an order, so its popularity periods cannot be built there), scored on the test users of the held-out domains over the whole catalogue. With the item popularity of the test period, as in its paper, it reaches 0.0123 NDCG@10 on Industrial and Scientific; with only the popularity known at the time of the last item of the history, a stricter setting, 0.0095.

## NDCG@10 of the baselines on all users of each domain

Whole catalogue, best configuration of the full grid, all users of each domain (the set from which the development users and the reserved users of the paper are drawn). Bold: best trained baseline of the column. This table was used only to choose the strongest trained baseline of each domain and the trained recommender of the pool. It is not used to compare with Decision4Rec: every comparison with Decision4Rec in the paper is made on the same sample of users for both systems (development or reserved users, see `splits/`). MovieLens-1M has two columns, with genres only (plain) and enriched. The methods that do not read item attributes (popularity, item kNN, sequential rules, GRU4Rec, SASRec, BERT4Rec) give the same value in both.

| NDCG@10 ↑ | VG | MI | Steam | IND | ML-1M plain | ML-1M enriched |
|---|---|---|---|---|---|---|
| *Training-free methods* | | | | | | |
| popularity | 0.0126 | 0.0133 | 0.0259 | 0.0088 | 0.0180 | 0.0180 |
| item kNN | 0.0516 | 0.0314 | 0.0475 | 0.0231 | 0.1337 | 0.1337 |
| sequential rules | 0.0494 | 0.0287 | 0.0461 | 0.0226 | 0.1425 | 0.1425 |
| content kNN | 0.0155 | 0.0111 | 0.0159 | 0.0150 | 0.0274 | 0.0277 |
| transitions and text | 0.0250 | 0.0142 | 0.0274 | 0.0170 | 0.0441 | 0.0370 |
| *Trained in each domain* | | | | | | |
| GRU4Rec | 0.0538 | 0.0318 | 0.0637 | 0.0217 | 0.1853 | **0.1853** |
| SASRec | 0.0551 | 0.0338 | 0.0672 | 0.0262 | 0.1850 | 0.1850 |
| BERT4Rec | 0.0500 | 0.0298 | 0.0604 | 0.0200 | 0.1623 | 0.1623 |
| SASRecF | **0.0585** | **0.0365** | **0.0680** | **0.0296** | **0.1870** | 0.1828 |

## UniSRec on the development users (pre-registered comparison)

The comparison of Decision4Rec-Full with UniSRec was pre-registered (`docs/r1/b5_prereg.md` of the working repository) on the 2,000 development test users for the six development settings. The paper reports the four domains on the 10,000 reserved users instead (rows added after the plan); the pre-registered rows are here. NDCG@10, UniSRec fine-tuned on the scenario (transductive), Holm-corrected `**` p < 0.01.

| Scenario | Users | SASRecF | UniSRec | Decision4Rec-Full | Full vs UniSRec |
|---|---|---|---|---|---|
| Video Games | 2,000 | 0.0622 | 0.0629 | 0.0791 | +25.8% ** |
| Musical Instruments | 2,000 | 0.0370 | 0.0386 | 0.0504 | +30.6% ** |
| Steam | 2,000 | 0.0759 | 0.0676 | 0.0851 | +25.9% ** |
| Industrial and Scientific | 2,000 | 0.0310 | 0.0326 | 0.0416 | +27.7% ** |
| ML-1M | 2,000 | 0.1806 | 0.1440 | 0.1923 | +33.6% ** |
| ML-1M enriched | 2,000 | 0.1812 | 0.1368 | 0.1984 | +45.0% ** |

The same four domains on the 10,000 reserved users (`scripts/b5_score_reserved.py`, UniSRec models unchanged, ranking only): Full is above UniSRec by +26.5%, +20.7%, +30.6% and +14.7% (Holm p = 0.0004 over the four domains).
