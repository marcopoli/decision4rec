# Splits: the users that every table of the paper scores

Each system with a decision model is scored on a sample of users, not on all users of a dataset. These files list the samples, so that a table can be checked or repeated on exactly the same users.

The split of each user is leave-one-out on the 5-core data: the last item is the test target, the one before it is the validation target, and the rest is the training part. A user is a row of `seqs.jsonl` of the processed dataset (the user number is the row index, starting at 0). The datasets are rebuilt from their public sources by the scripts of the repository. `MANIFEST.json` gives the SHA-256 of `seqs.jsonl` of each processed dataset, so a rebuilt dataset can be compared with the one used here.

| File | Users | Where it is used |
|---|---|---|
| `development_<d>.json` | 2,000 test and 1,000 validation users, `d` in vg, mi, steam, ind, ml1m, ml1m_plus (`_rtm` configs) | all design choices, fusion weights, development tables |
| `reserved_<d>.json` | 10,000 test and 1,000 validation users, `d` in vg, mi, steam, ind | confirmation on fresh users; the 11,000 users together are the reservation whose SHA-256 was frozen before scoring (`RESERVED_DIGESTS.json`) |
| `reserved_cold_<d>.json` | all eligible reserved users on the data cut at the 0.9 quantile of time (the user numbers are those of the cut dataset, e.g. `video_games_cold`) | confirmation, cold items after a time cut |
| `unseen_<d>.json` | 5,000 test and 1,000 validation users, `d` in cds, toys, sw, baby, pet, groc, arts, ml32m, mind | scenarios never used to train a decision model |

Digests: the SHA-256 of a list is the SHA-256 of its sorted user numbers written as decimal numbers joined by commas (`decision4rec.data.reserved.digest`). `python scripts/check_splits.py` recomputes the digests of all files and compares them with `MANIFEST.json` and `RESERVED_DIGESTS.json`. Development and reserved users do not overlap, and the 1,000 reserved validation users served diagnostics only.
