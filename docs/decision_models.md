# Change the decision model

The decision model is an exchangeable slot. Every module talks to it through one interface: a **state** (text) and a
map of named **typed questions**; the model returns **one probability distribution per question**. Changing the
model is one line of configuration: the `--backend` file of `decision4rec run`.

```bash
decision4rec run --config configs/small/vg_rtm.yaml --backend configs/backends/kev-4b.yaml
decision4rec run --config configs/small/vg_rtm.yaml --backend configs/backends/jev.yaml
```

## The question types

| Type | Meaning | Answer |
|---|---|---|
| `choice` | pick one of several named options (up to 255 for Kev) | one probability per option |
| `noul` | is a statement true? | the probability that it is |
| `score` | a position on an ordered scale | one probability per level |

They are defined in [`decision4rec/decision/types.py`](../decision4rec/decision/types.py) and follow the System One
wire format (`POST /v1/systemone`). Decision4Rec asks one `choice` question per user with the candidates as options,
in two random orders, and combines the answers with a Luce model (worths and position effects), so the ranking does
not depend on the order in which the candidates were shown.

## Backends that are already available

| `kind` | For | Example file |
|---|---|---|
| `systemone` | any server that speaks the System One protocol: hosted Jev, a local Kev server, Jev-like servers | `configs/backends/jev.yaml`, `configs/backends/kev-4b.yaml` |
| `logit` | any causal language model of Hugging Face, read through the next-token probabilities of option labels (LlamaRec-style), in-process | `configs/backends/qwen2.5-7b-logit.yaml` |
| `oracle` | a simulated respondent for tests and synthetic data | `configs/synthetic.yaml` |

A backend file has a tag (used in the names of run folders and caches), the number of parallel requests and the
backend arguments:

```yaml
tag: kev-4b
workers: 1
backend: {kind: systemone, base_url: "http://127.0.0.1:8009", model: kev-latest, name: kev-4b, api_key_env: KEV_API_KEY}
```

### A hosted Jev-like model

Any model behind a System One endpoint needs only a new file. Keys are always read from environment variables, never
written in the file:

```yaml
tag: my-decision-model-1.0
workers: 8
backend: {kind: systemone, base_url: "https://api.example.com", model: my-model-1.0, name: my-decision-model-1.0,
          api_key_env: MY_API_KEY, max_request_chars: 90000}
```

`max_request_chars` splits large requests into several calls; this is exact because each answer depends only on the
state and its own question. `JEV_MAX_MTOK` caps the input tokens one process may send to a paid endpoint.

### An open decision model (Kev and compatible checkpoints)

```bash
cd ~/kev && uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

`--run` accepts a Hugging Face id (`jaredpalmer/kev-0.8b`, `jaredpalmer/kev-4b`, `jaredpalmer/kev-9b`,
`jaredpalmer/kev-27b`, our trained models) or a local run folder. Pin versions with `repo@revision`.

### Any language model, without a server

```yaml
tag: qwen3.5-4b-logit
workers: 1
backend: {kind: logit, model: Qwen/Qwen3.5-4B, dtype: bf16, batch_size: 4, use_chat_template: true}
```

The options are labelled and the model is read through the probability of each label as the next token. With many
options a small model may not learn this read-out; asking over blocks of 5 options (see `configs/small/*_rtmb5.yaml`)
is the fallback used in the paper.

## Writing a new backend

Subclass `DecisionBackend` and register it in `build_backend`:

```python
# decision4rec/decision/mybackend.py
from typing import Mapping
from .base import DecisionBackend
from .types import Answer, Choice, Decision, Question

class MyBackend(DecisionBackend):
    def __init__(self, model: str, name: str | None = None):
        self.model, self.name = model, name or model

    def fingerprint(self) -> str:          # everything that can change an answer: used as cache key
        return f"mybackend:{self.model}"

    def decide(self, state: str, questions: Mapping[str, Question]) -> Decision:
        answers = {}
        for qname, q in questions.items():
            if isinstance(q, Choice):
                probs = my_model_probabilities(state, q.instructions, q.options)   # {option key: probability}
                answers[qname] = Answer(kind="choice", probs=probs)
        return Decision(answers=answers)
```

```python
# decision4rec/decision/base.py, in build_backend
elif kind == "mybackend":
    from .mybackend import MyBackend
    backend = MyBackend(**cfg)
```

Check it with the conformance test, which also verifies that identical requests give identical answers:

```bash
decision4rec check-backend --config configs/synthetic.yaml --backend configs/backends/my-backend.yaml
```

Every answer is cached in SQLite under its fingerprint, the state and the questions, so runs can be resumed and
replayed exactly.

## Training a decision model

The paper trains open Kev models on **real next choices** (never on outputs of a commercial model): each record is the
state of a user and a `choice` question over the candidates of the pool, with the item the user really chose next as
the label.

```bash
bash scripts/setting.sh small rt_export     # export the training records (15,000 users per source domain)
RTV=rtm bash scripts/setting.sh small rt_train_08    # Kev-0.8B; rt_train_4, rt_train_9 and rt_train_27 for larger sizes
RTV=rtm bash scripts/setting.sh small rt_eval_08     # evaluate on all development users and write the report
```

The stages call Kev's trainer (`python -m kev.train`, adapters on a frozen base, warm-started from the released Kev
model) with the settings of the paper. Records are exported only for users that are never evaluated and never used to
train the ranker of the pool, and the target is the second-to-last item, which the trained recommender never saw as a
training target. To train on your own domain, add its configuration to `RT_DOMAINS` (see `scripts/setting.sh`).

| Setting | 0.8B | 4B | 9B | 27B |
|---|---|---|---|---|
| base | Qwen/Qwen3.5-0.8B-Base | Qwen/Qwen3.5-4B-Base | Qwen/Qwen3.5-9B-Base | Qwen/Qwen3.8-27B |
| warm start | jaredpalmer/kev-0.8b | jaredpalmer/kev-4b | jaredpalmer/kev-9b | jaredpalmer/kev-27b@v1-lora |
| learning rate | 1e-4 | 5e-5 | 5e-5 | 2e-5 |
| records per step | 8 | 8 | 8 | 8 |
| GPU memory used for training in the paper | 24 GB (RTX 3090) | 24 GB (RTX 3090) | 128 GB unified (DGX Spark) | 128 GB unified (DGX Spark) |
