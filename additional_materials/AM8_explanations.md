# AM8. Explanations by removing one family of statements

For 500 test users per domain we asked the decision model of Decision4Rec-Full again with one family of statements removed (overall popularity, recent popularity, follow-up, co-purchase, text similarity), or without the "found by" lines that name the agents of the pool. The family whose removal changes the probability of the first item most is the *explanation* of that recommendation.

## Means over the four domains

Δp: mean change of the probability of the item ranked first. Changes: share of users whose first choice changes. Share: how often the family is the explanation. Last two columns: change of NDCG@10 without the family, range over the four domains. Agreement: Spearman correlation of the answer with the order of the residual ranker within the list (one question per user); with all inputs it is 0.69 (0.8B) and 0.61 (4B). The columns are diagnostics, with no better direction.

| Removed | Δp, 0.8B | changes, 0.8B | share, 0.8B | Δp, 4B | changes, 4B | share, 4B | NDCG@10 without it, 0.8B | NDCG@10 without it, 4B | agreement, 0.8B | agreement, 4B |
|---|---|---|---|---|---|---|---|---|---|---|
| overall popularity | 0.006 | 5% | 0% | 0.006 | 6% | 1% | -4 to +1% | -2 to +1% | 0.69 | 0.61 |
| recent popularity | 0.009 | 9% | 2% | 0.010 | 12% | 5% | -2 to +4% | -5 to +1% | 0.65 | 0.59 |
| follow-up | 0.024 | 14% | 5% | 0.016 | 14% | 5% | -4 to +1% | -6 to +1% | 0.66 | 0.59 |
| co-purchase | 0.004 | 4% | 0% | 0.005 | 5% | 0% | -3 to +3% | -3 to +1% | 0.69 | 0.59 |
| text similarity | 0.007 | 5% | 0% | 0.005 | 4% | 0% | -2 to +4% | -2 to +1% | 0.68 | 0.60 |
| "found by" | 0.083 | 66% | 93% | 0.063 | 55% | 89% | -15 to -33% | -3 to -22% | 0.35 | 0.40 |

Without the "found by" lines the agreement falls in every domain and at both sizes (paired randomization test, p < 0.001).

## Three 0.8B decision models of Decision4Rec-Full

|  | trained | found-by dropout | corrects the ranker |
|---|---|---|---|
| share: "found by" | 93% (90–98) | 93% (89–98) | 57% (40–81) |
| share: follow-up | 5% (2–7) | 5% (2–7) | 33% (14–52) |
| share: text similarity | 0% (0–1) | 0% (0–0) | 8% (3–14) |
| Δp of "found by" | 0.083 | 0.073 | 0.005 |
| deletion test, chosen against random | 0.080–0.088 / 0.023–0.027 | 0.070–0.078 / 0.021–0.024 | 0.006–0.007 / 0.002 |
| NDCG@10 without "found by" | -33 to -15% | -37 to +3% | n/a |

**Findings.** The explanations are faithful in the sense of a deletion test (removing the chosen family changes the probability of the first item 3 to 4 times more than a family drawn at random), but the explanation is almost always the "found by" lines, that is, which agents proposed the item. Leaving these lines out of a quarter of the training records is not enough to make the model explain by behavior. Trained to correct the ranker (the score of the ranker added to its logits in the loss), the model explains by behavior far more often, but its probabilities move little and the fused ranking does not improve.
