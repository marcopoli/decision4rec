# AM7. Jev and Kev

## After fusion

The same models after fusion with the order that they receive, which is the final score of Decision4Rec. Bold: best model of the block. The released models and the trained 0.8B model are within 0.0017 of each other on the four main domains (0.0036 on MovieLens-1M); read alone (paper, Section 8) they are 0.0094 to 0.0626 apart. † as in the paper.

| NDCG@10 ↑ | Video Games | Musical Instr. | Steam | Industrial† | MovieLens-1M† |
|---|---|---|---|---|---|
| **Decision4Rec-Full** |  |  |  |  |  |
| Jev, released | 0.0733 | 0.0479 | 0.0789 | 0.0391 | 0.1890 |
| Kev-0.8B, released | 0.0728 | 0.0475 | 0.0785 | 0.0404 | 0.1899 |
| Kev-4B, released | 0.0733 | 0.0474 | 0.0799 | 0.0389 | 0.1869 |
| Kev-0.8B, trained | 0.0741 | 0.0467 | **0.0802** | 0.0401 | 0.1863 |
| Kev-4B, trained | **0.0780** | **0.0504** | 0.0801 | **0.0407** | **0.1907** |
| **Decision4Rec-KG with logs** |  |  |  |  |  |
| Jev, released | 0.0606 | 0.0380 | 0.0553 | 0.0336 | 0.1530 |
| Kev-0.8B, released | 0.0600 | 0.0361 | **0.0565** | 0.0297 | 0.1514 |
| Kev-4B, released | 0.0612 | 0.0363 | 0.0553 | 0.0317 | 0.1476 |
| Kev-0.8B, trained | **0.0637** | **0.0401** | 0.0558 | **0.0348** | **0.1532** |
| **Decision4Rec-KG, catalogue only** |  |  |  |  |  |
| Jev, released | – | – | – | 0.0185 | 0.0382 |
| Kev-0.8B, released | – | – | – | 0.0171 | 0.0314 |
| Kev-4B, released | – | – | – | 0.0171 | 0.0341 |
| Kev-0.8B, trained | – | – | – | **0.0211** | **0.0383** |

## The paid API against open weights

Hosted Jev and open Kev as the decision model of Decision4Rec.

|  | Jev (paid API) | Kev (open weights) |
|---|---|---|
| Training on the scenario | not possible | LoRA and choice head on real choices |
| Distillation | not allowed by the terms | allowed by its open licence |
| Hardware | none | one GPU (24 GB for 0.8B and 4B) |
| Cost | $0.15–0.22 per 1,000 users | electricity and the GPU |
| Time per user | 310–340 ms | 157–167 ms (0.8B, RTX 3090) |
| User data | sent to the provider | stays in the scenario |
| Replay | from saved answers only | from the weights and saved answers |
| Quality as released | not significantly different from Kev-4B in 8 of 12 settings; better with a catalogue only and on MovieLens-1M | not significantly different from Jev in 8 of 12 settings |
| Quality after training | training not possible | read alone, above Jev in 9 of 12 settings (0.8B, up to +43%) and in 4 of 4 (4B, +15 to +50%) |
| Repeatable answers | no (0.3–0.7% identical) | yes, up to rounding |

## Repeatability of the hosted answers

We sent the same 1,200 requests (300 users of Video Games) a second time, a few hours later. Only 0.3 to 0.7% of the answers were identical; the largest change of a probability had a median of 0.03 and a 99th percentile of 0.12 to 0.13; the first choice changed in 7% of the requests. The effect on accuracy is small, but a hosted result is not a function of the request alone. The answers of Jev are not redistributed, as its terms require.
