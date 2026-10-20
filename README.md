<div align="center">

# Decision4Rec

**A Jev-inspired framework of decision models on knowledge and active graphs for multi-agent recommender systems**

[Paper](http://arxiv.org/abs/2610.26035) · [Quick start](#quick-start) · [Documentation](docs/) · [Models](#trained-models) · [Reproduce the paper](docs/reproduce_the_paper.md)

[![arXiv](https://img.shields.io/badge/arXiv-2610.26035-b31b1b)](http://arxiv.org/abs/2610.26035)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Status](https://img.shields.io/badge/status-research%20code-orange)

</div>

---

Decision models, such as the commercial Jev or the open Kev models, are language models that return **one probability for each option of a typed question**, in one forward pass and without generating text. Decision4Rec makes them usable for recommendation, with the decision model as an exchangeable part:

- a **knowledge graph** of the scenario, kept live on the [ActiveGraph](https://github.com/yoheinakajima/activegraph) runtime, writes the logs as short statements ("sold 1,240 times", "often bought after your last item");
- a **multi-agent pool**, with the trained recommender of the scenario as one of the agents, proposes candidates, and a residual ranker keeps the best 30;
- a **decision model** reads them through one typed-question interface (the System One protocol), and a Luce model turns its answers into a ranking;
- a **fusion** step combines its answers with the score of the ranker.

A commercial model through its API, a released open model and a trained open model run on the same requests, and **changing the decision model is one line of configuration**.

> **Note.** Decision4Rec is not affiliated with TypeSafe AI. "Jev" names the interface that inspired the framework; the framework works with any decision model that speaks the same protocol, such as the open [Kev](https://github.com/jaredpalmer/kev) models.

## Contents

- [Architecture](#architecture)
- [Results at a glance](#results-at-a-glance)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Extending Decision4Rec](#extending-decision4rec)
- [Trained models](#trained-models)
- [Reproducing the paper](#reproducing-the-paper)
- [Data and licenses](#data-and-licenses)
- [Citation](#citation)
- [License](#license)

## Architecture

```
          events (new interactions)                catalogue (text, attributes)
                     │                                        │
                     ▼                                        ▼
   ┌────────────────────────────────────────────────────────────────────────────┐
   │ ActiveGraph runtime: every step is an event in the log (replay, audit)     │
   │                                                                            │
   │  (1) Knowledge graph ──statements──►(3) Decision model ──►(4) Fusion ──► top-N
   │       of the scenario                    exchangeable slot      + Luce model
   │            │                              ▲   Jev (API), Kev (released or trained),
   │            ▼                              │   any System One server, a logit LM
   │  (2) Multi-agent pool ──30 best──────────┘
   │       SASRecF, transitions, co-purchase, text, popular now,
   │       shared attributes, best sellers → residual ranker
   └────────────────────────────────────────────────────────────────────────────┘
```

| Module | Contract | Code |
|---|---|---|
| Knowledge graph of the scenario | in: logs and catalogue; out: statements on scales of the scenario | [`decision4rec/evidence`](decision4rec/evidence), [`decision4rec/graph`](decision4rec/graph) |
| Multi-agent pool | in: user history; out: ranked candidates and "found by" lines | [`decision4rec/agents/pool.py`](decision4rec/agents/pool.py) |
| Decision model | in: state and typed questions; out: one distribution per question | [`decision4rec/decision`](decision4rec/decision) |
| Ranking and fusion | in: answers over random orders; out: calibrated ranking | [`decision4rec/operators`](decision4rec/operators) |

The three variants studied in the paper share these modules:

| Variant | What the decision model reads |
|---|---|
| **Decision4Rec-Full** | the 30 best items of the multi-agent pool, with statements and "found by" lines |
| **Decision4Rec-List** | the 30 best items of the trained recommender (no pool) |
| **Decision4Rec-KG** | the 30 candidates of the knowledge graph (no trained recommender) |

## Results at a glance

NDCG@10 on 2,000 development test users per domain (full-catalogue ranking, leave-one-out). Industrial and Scientific is **never seen** when training the decision model.

| System | Video Games | Musical Instr. | Steam | Industrial & Sci. |
|---|---|---|---|---|
| Best trained baseline | 0.0622 | 0.0370 | 0.0790 | 0.0310 |
| Decision4Rec-Full, no decision model (pool + residual ranker) | 0.0730 | 0.0468 | 0.0786 | 0.0386 |
| Decision4Rec-Full, trained Kev-0.8B | 0.0741 | 0.0467 | 0.0802 | 0.0401 |
| Decision4Rec-Full, trained Kev-4B | **0.0780** | **0.0504** | 0.0801 | **0.0407** |

The contribution of the decision model grows with its size; the paper reports the full analysis, the commercial Jev, MovieLens-1M and the controls.

## Installation

Decision4Rec needs Python 3.11 or newer. A GPU is needed to train the recommenders and to serve or train decision models (a 24 GB GPU is enough for Kev-0.8B and Kev-4B).

```bash
git clone https://github.com/marcopoli/decision4rec.git
cd decision4rec
bash scripts/env.sh            # creates .venv with every library of the experiments
source .venv/bin/activate
```

For a minimal install (the framework only):

```bash
pip install -e .
```

Optional extras: `train` (PyTorch), `logit` (Transformers, for language models read through label logits), `active` (ActiveGraph), `dev` (pytest).

Decision models are served by their own servers. For the open Kev models:

```bash
git clone https://github.com/jaredpalmer/kev.git ~/kev
cd ~/kev && uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

## Quick start

A complete run on synthetic data, with a mock decision model, in a few minutes on a laptop:

```bash
decision4rec prepare --dataset synthetic --out data/processed/synthetic
decision4rec run --config configs/synthetic.yaml          # writes runs/synthetic/synthetic_oracle/report.md
decision4rec report runs/synthetic/synthetic_oracle      # recompute the report of a run folder
```

On a real domain, with the full architecture and a released Kev model:

```bash
bash scripts/setting.sh small data            # download and prepare the datasets
bash scripts/setting.sh small retriever       # train SASRecF on each domain
bash scripts/setting.sh small baselines       # train the baselines
decision4rec run --config configs/small/vg_rtm.yaml --backend configs/backends/kev-4b.yaml
```

`scripts/setting.sh` runs every stage of the paper, starts and stops the decision-model server, and resumes where it stopped. Run `bash scripts/setting.sh small <stage>` for one stage, or see [docs/reproduce_the_paper.md](docs/reproduce_the_paper.md).

## Repository layout

```
decision4rec/
├── decision4rec/
│   ├── agents/        multi-agent pool, residual ranker, ActiveGraph agents
│   ├── data/          dataset format, loaders, evaluation cases, reserved users
│   ├── decision/      decision-model backends (System One, logit, mock) and cache
│   ├── eval/          metrics and paired tests
│   ├── evidence/      statements of the knowledge graph and the prompt builder
│   ├── experiments/   runner, benchmark, record export, reports
│   ├── graph/         item knowledge graph
│   ├── models/        SASRec, SASRecF and the other baselines
│   └── operators/     Luce model, block designs, ranking, risk control
├── configs/           one YAML file per experiment (small/, large/, backends/)
├── scripts/           setting.sh (all stages), env.sh, analysis and probe scripts
├── tests/             pytest suite
└── docs/              guides
```

## Extending Decision4Rec

| I want to... | Guide |
|---|---|
| run on my own dataset | [docs/add_a_dataset.md](docs/add_a_dataset.md) |
| use another decision model (Jev, Kev, a Jev-like model, any LLM) | [docs/decision_models.md](docs/decision_models.md) |
| add an agent to the candidate pool | [docs/agents.md](docs/agents.md) |
| train my own decision model on real choices | [docs/decision_models.md#training-a-decision-model](docs/decision_models.md#training-a-decision-model) |

## Trained models

The trained decision models are adapters on the Qwen3.5 base models, trained from the released Kev models on real choices of three domains (Video Games, Musical Instruments, Steam). They are published on Hugging Face:

| Model | Base | Variant |
|---|---|---|
| `m-polignano/decision4rec-kev-0.8b` | Qwen/Qwen3.5-0.8B-Base | Decision4Rec-Full |
| `m-polignano/decision4rec-kev-4b` | Qwen/Qwen3.5-4B-Base | Decision4Rec-Full |
| `m-polignano/decision4rec-kev-9b` | Qwen/Qwen3.5-9B-Base | Decision4Rec-Full (8,000 records) |
| `m-polignano/decision4rec-kev-27b` | Qwen/Qwen3.8-27B | Decision4Rec-Full (8,000 records) |

Serve one as any Kev checkpoint:

```bash
uv run --extra serve python -m kev.serve --run m-polignano/decision4rec-kev-4b --port 8009
```

## Reproducing the paper

Every number of the paper comes from one stage of `scripts/setting.sh` and one report under `runs/small/`. The mapping from tables to stages, the hardware, the run times and the random seeds are in [docs/reproduce_the_paper.md](docs/reproduce_the_paper.md). The development and test users of every domain are fixed by seeds and can be rebuilt exactly; their identifiers will also be released.

## Data and licenses

The datasets are not redistributed. `scripts/setting.sh small data` downloads them from their sources and prepares them:

| Dataset | Source | Terms |
|---|---|---|
| Amazon Reviews 2023 (Video Games, Musical Instruments, Industrial and Scientific) | [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/) | research use, see the source |
| Steam | [cseweb.ucsd.edu/~wckang](https://cseweb.ucsd.edu/~wckang/) | research use, see the source |
| MovieLens-1M | [grouplens.org](https://grouplens.org/datasets/movielens/1m/) | no redistribution |

The hosted Jev runs need your own key (`export TYPESAFE_API_KEY=...`); their answers are not distributed, as required by the TypeSafe terms of service. Decision4Rec never trains on Jev outputs.

## Citation

The paper is on arXiv: [arXiv:2610.26035](http://arxiv.org/abs/2610.26035).

```bibtex
@misc{polignano2026decision4rec,
  title         = {Decision4Rec: A Jev-Inspired Framework of Decision Models on Knowledge and Active Graphs
                   for Multi-Agent Recommender Systems},
  author        = {Polignano, Marco and Mehta, Het Darshan and Lops, Pasquale and De Luca, Ernesto William
                   and Semeraro, Giovanni},
  year          = {2026},
  eprint        = {2610.26035},
  archivePrefix = {arXiv},
  primaryClass  = {cs.IR},
  url           = {http://arxiv.org/abs/2610.26035}
}
```

## License

The code is released under the [Apache License 2.0](LICENSE). The trained models keep the licenses of their base models (Qwen3.5) and of Kev (Apache-2.0).

## Acknowledgements

Decision4Rec builds on [Kev](https://github.com/jaredpalmer/kev), [ActiveGraph](https://github.com/yoheinakajima/activegraph), the [Qwen](https://huggingface.co/Qwen) models and the System One protocol of TypeSafe AI.
