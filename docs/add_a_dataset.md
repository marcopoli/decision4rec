# Add your own dataset

Decision4Rec reads every domain from one plain folder format. Once your data are in this format, the knowledge graph,
the agents, the baselines and the decision model work on them without code changes.

## 1. The folder format

A prepared dataset is a folder with three files:

```
data/processed/my_shop/
├── meta.json       {"name": "my_shop", "n_users": ..., "n_items": ..., "n_interactions": ..., "extra": {}}
├── items.jsonl     one line per item, in item-index order
└── seqs.jsonl      one line per user: the chronological interactions of that user
```

Each line of `items.jsonl` describes one item. Only `id` and `title` are required:

```json
{"id": "B07XJ8C8F5", "title": "Zelda: Breath of the Wild", "meta": "Video Games > Nintendo Switch > Games",
 "desc": "Step into a world of discovery ...", "attrs": [["brand", "Nintendo"], ["category", "Action-Adventure"]]}
```

| Field | Used by |
|---|---|
| `title` | the decision model (item text), the text agent, the content baselines |
| `meta` | item text (for example the category path); shown after the title |
| `desc` | item text; cut to `prompt.item_chars` characters in the questions |
| `attrs` | the knowledge graph: shared-attribute agent, rare-attribute statements, perspectives |

Each line of `seqs.jsonl` is the list of `[item_index, timestamp, rating]` of one user, oldest first. The item index
is the line number of the item in `items.jsonl` (from 0). Timestamps can be Unix seconds or any increasing order;
ratings can be `1` for implicit feedback.

```json
[[12, 1600000000, 5], [873, 1600090000, 4], [12, 1600500000, 1]]
```

## 2. Build it from Python

The `Dataset` class writes the folder for you:

```python
from decision4rec.data.schema import Dataset

items = [{"id": "sku-1", "title": "Red running shoes", "meta": "Shoes > Running", "desc": "...",
          "attrs": [["brand", "Acme"], ["color", "red"]]},
         {"id": "sku-2", "title": "Trail socks", "meta": "Socks", "desc": "", "attrs": [["brand", "Acme"]]}]
seqs = [[[0, 1700000000, 1], [1, 1700003600, 1]],      # user 0
        [[1, 1700000500, 1], [0, 1700100000, 1]]]      # user 1
Dataset(name="my_shop", items=items, seqs=seqs).save("data/processed/my_shop")
```

Two rules matter for a fair evaluation:

1. **k-core filtering.** Keep users and items with at least 5 interactions, as the loaders do
   (`decision4rec.data.loaders.k_core`).
2. **Leave-one-out.** Nothing else is needed: the evaluation cases (test and validation users, each with its held-out
   item as the target) are drawn from `seqs.jsonl` with the seed of the configuration (`eval.seed`), so they can be
   rebuilt exactly.

If your data come in a format that is already supported, use the loader instead:

| Source | Command |
|---|---|
| Amazon Reviews 2023 category | `decision4rec prepare --dataset amazon23 --reviews <reviews.jsonl.gz> --meta <meta.jsonl.gz> --name my_cat --out data/processed/my_cat` |
| MovieLens (ml-1m) | `decision4rec prepare --dataset ml1m --folder <ml-1m> --out data/processed/ml1m` |
| MovieLens CSV (ml-20m, ml-25m, ...) | `decision4rec prepare --dataset mlcsv --folder <folder> --name ml20m --out data/processed/ml20m` |
| Steam | `decision4rec prepare --dataset steam --reviews <steam_reviews.json.gz> --meta <steam_games.json.gz> --out data/processed/steam` |

## 3. Train the trained recommender of the scenario (SASRecF) and the baselines

Copy the configuration of a similar domain and change the names and paths. For a sparse product catalogue start from
the Video Games files:

```bash
cp configs/small/vg_semantic.yaml configs/small/my_shop_semantic.yaml   # reference config: baselines, SASRecF
cp configs/small/vg_rtm.yaml      configs/small/my_shop_rtm.yaml        # Decision4Rec-Full
```

In both files set `name`, `dataset: data/processed/my_shop` and `out: runs/small/my_shop`; in `my_shop_rtm.yaml` set
`pool.ref: configs/small/my_shop_semantic.yaml`. Remove the `facets:` and `relations:` sections of the reference
config if you do not use them. Then:

```bash
decision4rec train-retriever --dataset data/processed/my_shop --out runs/small/my_shop/sasrec
decision4rec baselines --config configs/small/my_shop_semantic.yaml            # SASRec, SASRecF, GRU4Rec, ...
```

## 4. Run Decision4Rec on it

Start a decision-model server (see [decision_models.md](decision_models.md)) and run the full architecture:

```bash
decision4rec run --config configs/small/my_shop_rtm.yaml --backend configs/backends/kev-4b.yaml
```

The first run trains the residual ranker of the pool on users that are never evaluated and caches it under
`runs/small/my_shop/pool/`. The results, the report and every answer of the decision model are written to
`runs/small/my_shop/my_shop_rtm_kev-4b/`; answers are cached, so a second run is a replay.

A decision model trained on other domains can be used **unchanged** on your domain: the statements mean the same
thing everywhere ("sold in the top 5% of this scenario"). This is how Industrial and Scientific is evaluated in the
paper. To train a decision model on your domain as well, see
[decision_models.md](decision_models.md#training-a-decision-model).

## Checklist

- [ ] items have a `title`; `attrs` filled where the catalogue has brands, categories or series
- [ ] interactions sorted by time per user, at least 5 per user after filtering
- [ ] SASRecF trained and the baselines computed with the reference config
- [ ] `pool.ref` of the Decision4Rec config points to the reference config
