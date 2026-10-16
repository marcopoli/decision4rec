# Reproduce the paper

Every number of the paper comes from one stage of [`scripts/setting.sh`](../scripts/setting.sh) and is read from one
report under `runs/small/`. Stages are resumable: run the same command again and finished work is skipped; answers of
decision models are cached, so a rerun is a replay.

```bash
bash scripts/setting.sh small <stage>      # one stage
bash scripts/setting.sh small all          # the core stages in order
DRY_RUN=1 bash scripts/setting.sh small <stage>   # print the commands only
```

## Hardware used

| Machine | Used for |
|---|---|
| 1 x NVIDIA RTX 3090 (24 GB) | SASRecF and baselines; Kev-0.8B and Kev-4B training and evaluation; LlamaRec-style reranker |
| NVIDIA DGX Spark (GB10, 128 GB unified memory) | Kev-9B and Kev-27B training and evaluation |
| CPU | knowledge graph, pool, Luce fits, reports, paired tests |

Kev-0.8B trains in about 2 hours and Kev-4B in about 7 hours on 34,667 records; one evaluation of 2,000 test and
1,000 validation users per domain takes about 15 minutes per domain at 0.8B.

## Pipeline

| Step | Stage(s) | Output |
|---|---|---|
| Environment and tests | `setup` | `.venv`, test report |
| Datasets | `data` | `data/processed/<domain>/` |
| Trained recommender (SASRecF) and baselines | `retriever`, `baselines` | `runs/small/<d>/sasrec`, baseline checkpoints |
| Training records of the decision model | `rt_export` | `runs/small/rectune_choices_<variant>/` |
| Decision model, trained | `rt_train_08`, `rt_train_4`, `rt_train_9`, `rt_train_27` | Kev run folders |
| Evaluation and report | `rt_eval_08`, `rt_eval_4`, `rt_eval_9`, `rt_eval_27` | `runs/small/<variant>_eval_<size>.md` |
| Released decision models | `released08`, `released27` | reports of the released Kev models |
| Hosted Jev (needs `TYPESAFE_API_KEY`) | `jev_eval`, `jev_ml1m`, `jev_ml1m_plus` | `runs/small/jev_*.md` |
| MovieLens-1M (plain and enriched) | `ml1m_eval_08`, `ml1m_eval_4`, `ml1m_plus_eval_08` | `runs/small/ml1m*_eval_*.md` |
| LlamaRec-style reranker | `llm_train_08`, `llm_eval_08` | `runs/small/llm*_rtm_eval_0.8b.md` |
| Explanations | `explain`, `explain_rt` | `runs/small/*/explain_*.json` |

The variant of a stage is chosen with environment variables:

| Variable | Meaning | Paper |
|---|---|---|
| `RTV=rtm` | Decision4Rec-Full (pool reader) | main system |
| `RTV=rte` | Decision4Rec-List (reader of the SASRecF top 30) | RQ3 |
| `EVV=ev2` with the `ev_*` stages | Decision4Rec-KG (evidence reader) | RQ2 |
| `RTV=rtmc` | 250 characters per item | RQ3, content |
| `RT_MODEL=rtm8k` | the 8,000-record subset for the size curve | size curve |

## Tables and figures

| Paper element | Report or script |
|---|---|
| Main results (RQ1) | `runs/small/rtm_eval_0.8b.md`, `rtm_eval_4b.md` |
| What the decision model adds (RQ3) | the same reports; `scripts/arm_pairs.py` for paired tests between two models |
| Size curve | `runs/small/rtm8k_eval_*.md` and `scripts/arm_pairs.py` on the common users |
| MovieLens-1M | `runs/small/ml1m_rtm_eval_*.md`, `ml1m_plus_rtm_eval_*.md` |
| Jev and released models | `runs/small/jev_*.md`, `released08_*.md` |
| Perspective agents | `scripts/why_probe.py`, `scripts/perspective_*.py` |

## Statistics

All comparisons are paired over the same users: a two-sided sign-flip randomization test with 10,000 permutations,
and bootstrap intervals with 2,000 resamples (`decision4rec/eval/metrics.py`). Development tables mark p < 0.05
without correction; the fusion weight is chosen on validation users only.

## Fixed seeds and users

| What | Seed |
|---|---|
| Evaluation users (2,000 test, 1,000 validation per domain) | `eval.seed: 13` |
| Users that train the ranker of the pool | `pool.seed: 11` |
| Training records of the decision model | 15,000 users per source domain, excluding every evaluated user and every ranker user |
| Data-curve subsets | `scripts/subset_records.py ... 2026 --anchor 8000` |

Users reserved for a confirmatory test on fresh users are left out of every sampler
(`decision4rec/data/reserved.py`); their identifiers are not part of this repository.
