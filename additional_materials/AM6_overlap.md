# AM6. How far the decision model repeats the ranker

How far the trained decision model repeats the order of the list it reads, 2,000 development test users per domain. Spearman: mean correlation, within the 30 candidates, between the order of the decision model read alone and the order of the list. Fused: change of the fused NDCG@10 against the order of the list (reference rows of the RQ3 table of the paper). Surprise: negative log-likelihood of the next item among the 30 options (uniform guess 3.40); for the ranker, with a temperature fitted on validation users. Found-by dropout: "found by" left out of a quarter of the training records. Corrects the ranker: the score of the ranker added to the logits in the training loss. *: p < 0.05. n/a: not run. 250 characters per item: the decision model reads the first 250 characters of each item text instead of 90 (AM2). ‡Enriched MovieLens-1M; the other ML-1M values are on the plain version.

| List read | Decision model | Video G. | Musical I. | Steam | Industrial | ML-1M |
|---|---|---|---|---|---|---|
| **Spearman with the order of the list (a diagnostic: no better direction)** |  |  |  |  |  |  |
| pool (30 best of the ranker) | 0.8B | 0.72 | 0.77 | 0.77 | 0.75 | 0.83 |
|  | 4B | 0.66 | 0.70 | 0.74 | 0.67 | 0.77 |
|  | 0.8B, found-by dropout | 0.74 | 0.78 | 0.79 | 0.76 | n/a |
|  | 0.8B, corrects the ranker | -0.18 | -0.24 | -0.31 | -0.08 | n/a |
|  | 0.8B, 250 characters per item | 0.74 | 0.77 | 0.79 | 0.76 | 0.83‡ |
| 30 best of SASRecF | 0.8B | 0.36 | 0.36 | 0.32 | 0.33 | n/a |
|  | 4B | 0.37 | 0.35 | 0.37 | 0.34 | n/a |
| **Fused NDCG@10, change against the order of the list** ↑ |  |  |  |  |  |  |
| pool (30 best of the ranker) | 0.8B | +2% | 0% | +2% | +4% | -2% |
|  | 4B | +7%* | +8%* | +2% | +5% | 0% |
|  | 0.8B, found-by dropout | +3% | +1% | +3%* | +3%* | n/a |
|  | 0.8B, corrects the ranker | +1% | +2% | +2% | +1% | n/a |
|  | 0.8B, 250 characters per item | +1% | +2% | +5%* | +3% | +1%‡ |
| 30 best of SASRecF | 0.8B | +16%* | +22%* | +8%* | +19%* | n/a |
|  | 4B | +17%* | +33%* | +7% | +16%* | n/a |
| **Surprise of the next item, users with the target among the 30 of the pool** ↓ |  |  |  |  |  |  |
| pool (30 best of the ranker) | 0.8B | 3.08 | 3.17 | 3.20 | 3.07 | 3.09 |
|  | 4B | 3.02 | 3.09 | 3.17 | 3.02 | 3.08 |
|  | ranker alone | 3.15 | 3.17 | 3.24 | 3.13 | 3.07 |

**250 characters per item (0.8B).** Read alone, the decision model changes by -2% on Video Games, +7% on Musical Instruments (p = 0.009), +2% on Steam, +4% on Industrial and Scientific (p = 0.052) and +1% on MovieLens-1M; after fusion it gains 3% on Steam (p = 0.031) and changes by -1 to +2% elsewhere. At 0.8B, more text does not stop the decision model from repeating the ranker.
