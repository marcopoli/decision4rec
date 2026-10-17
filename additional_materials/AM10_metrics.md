# AM10. HR@10 and MRR@30 for the main systems

Development test users (2,000 per domain). The rerankers order 30 candidates, so the mean reciprocal rank is cut at 30 (a target outside the 30 counts 0). Fused: the final score of each variant. Computed from the stored ranks with `scripts/am_metrics.py`. In every sparse domain, Decision4Rec-Full (4B) is above the best trained baseline on all three measures.

| Domain | System | NDCG@10 | HR@10 | MRR@30 |
|---|---|---|---|---|
| vg | sasrecf (best trained baseline) | 0.0622 | 0.1080 | 0.0527 |
| vg | pool (residual ranker) | 0.0730 | 0.1215 | 0.0630 |
| vg | Decision4Rec-Full, 0.8B, fused | 0.0741 | 0.1245 | 0.0634 |
| vg | Decision4Rec-Full, 4B, fused | 0.0780 | 0.1370 | 0.0644 |
| vg | Decision4Rec-List, 0.8B, fused | 0.0722 | 0.1260 | 0.0595 |
| vg | Decision4Rec-List, 4B, fused | 0.0730 | 0.1270 | 0.0600 |
| vg | Decision4Rec-KG, 0.8B, fused | 0.0637 | 0.1045 | 0.0544 |
| mi | sasrecf (best trained baseline) | 0.0370 | 0.0670 | 0.0310 |
| mi | pool (residual ranker) | 0.0468 | 0.0800 | 0.0400 |
| mi | Decision4Rec-Full, 0.8B, fused | 0.0467 | 0.0825 | 0.0391 |
| mi | Decision4Rec-Full, 4B, fused | 0.0504 | 0.0870 | 0.0423 |
| mi | Decision4Rec-List, 0.8B, fused | 0.0450 | 0.0785 | 0.0376 |
| mi | Decision4Rec-List, 4B, fused | 0.0491 | 0.0810 | 0.0419 |
| mi | Decision4Rec-KG, 0.8B, fused | 0.0401 | 0.0700 | 0.0324 |
| steam | sasrec (best trained baseline) | 0.0790 | 0.1475 | 0.0644 |
| steam | pool (residual ranker) | 0.0786 | 0.1500 | 0.0634 |
| steam | Decision4Rec-Full, 0.8B, fused | 0.0802 | 0.1530 | 0.0644 |
| steam | Decision4Rec-Full, 4B, fused | 0.0801 | 0.1545 | 0.0638 |
| steam | Decision4Rec-List, 0.8B, fused | 0.0821 | 0.1545 | 0.0658 |
| steam | Decision4Rec-List, 4B, fused | 0.0815 | 0.1515 | 0.0660 |
| steam | Decision4Rec-KG, 0.8B, fused | 0.0558 | 0.1030 | 0.0458 |
| ind | sasrecf (best trained baseline) | 0.0310 | 0.0535 | 0.0262 |
| ind | pool (residual ranker) | 0.0386 | 0.0680 | 0.0321 |
| ind | Decision4Rec-Full, 0.8B, fused | 0.0401 | 0.0685 | 0.0339 |
| ind | Decision4Rec-Full, 4B, fused | 0.0407 | 0.0680 | 0.0351 |
| ind | Decision4Rec-List, 0.8B, fused | 0.0370 | 0.0595 | 0.0319 |
| ind | Decision4Rec-List, 4B, fused | 0.0361 | 0.0620 | 0.0299 |
| ind | Decision4Rec-KG, 0.8B, fused | 0.0348 | 0.0560 | 0.0297 |
| ml1m | gru4rec (best trained baseline) | 0.1844 | 0.3200 | 0.1529 |
| ml1m | pool (residual ranker) | 0.1910 | 0.3300 | 0.1593 |
| ml1m | Decision4Rec-Full, 0.8B, fused | 0.1863 | 0.3290 | 0.1532 |
| ml1m | Decision4Rec-Full, 4B, fused | 0.1907 | 0.3365 | 0.1566 |
| ml1m | Decision4Rec-KG, 0.8B, fused | 0.1532 | 0.2615 | 0.1288 |
| ml1m_plus | gru4rec (best trained baseline) | 0.1844 | 0.3200 | 0.1529 |
| ml1m_plus | pool (residual ranker) | 0.1941 | 0.3275 | 0.1635 |
| ml1m_plus | Decision4Rec-Full, 0.8B, fused | 0.1939 | 0.3340 | 0.1609 |
| ml1m_plus | Decision4Rec-Full, 4B, fused | 0.1968 | 0.3375 | 0.1635 |
| ml1m_plus | Decision4Rec-KG, 0.8B, fused | 0.1512 | 0.2550 | 0.1280 |
