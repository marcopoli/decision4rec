"""Enrich MovieLens-1M with plots and knowledge-graph properties; writes data/processed/ml1m_plus.

    python scripts/enrich_ml1m.py data/processed/ml1m data/raw/enrich/REPRO_plot_text.tsv data/processed/ml1m_plus

Plots: IMDb plot texts released by Spillo et al., "Binge Watch" (M3L, Zenodo record 18499145, CC BY 4.0), keyed by the
MovieLens movie id, which MovieLens keeps across its releases. Properties: Wikidata, matched by the property "MovieLens
movie ID" (P14449; P1970 is the MovieMeter id), read through the QLever mirror of Wikidata (the public query service
was rate limited to one query per minute). A match is kept only if the year of the film is within one year of the
year in the MovieLens title: director, cast, screenwriter, composer, production company, country, original language, main
subject, the work it is based on, the series it belongs to, and the movies it follows or is followed by. The users,
the sequences and the item indices are those of the source folder, so every evaluation case stays the same.
"""

import json
import os
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request

PROPS = {"P57": "director", "P161": "cast", "P58": "writer", "P86": "composer", "P272": "studio",
         "P495": "country", "P364": "language", "P921": "subject", "P144": "based on", "P179": "series",
         "P155": "follows", "P156": "followed by"}
MAX_PER = {"cast": 6}
UA = "Decision4Rec-research/0.1 (academic recommender systems study; contact via the repository)"
ENDPOINT = os.environ.get("WD_ENDPOINT", "https://qlever.dev/api/wikidata")
PREFIXES = ("PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n")


def sparql(query: str) -> list[dict]:
    url = ENDPOINT + "?" + urllib.parse.urlencode({"query": PREFIXES + query})
    for attempt in range(12):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/sparql-results+json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)["results"]["bindings"]
        except Exception as e:                       # rate limits and timeouts: wait and retry
            print(f"[wikidata] retry {attempt + 1}: {e}", flush=True)
            time.sleep(min(70 * (attempt + 1), 600))   # the public endpoint may allow one query per minute
    raise SystemExit("Wikidata query failed twelve times; run again, finished batches are kept")


def wikidata(ids: list[int], cache_dir: str, batch: int = 150) -> dict[int, dict[str, list[tuple[str, str]]]]:
    """{movielens id: {relation: [(wikidata id, English label), ...]}}. Each batch is saved, so a restart resumes."""
    out: dict = {}
    os.makedirs(cache_dir, exist_ok=True)
    for s in range(0, len(ids), batch):
        chunk = ids[s: s + batch]
        part = os.path.join(cache_dir, f"batch_{chunk[0]}_{chunk[-1]}_{len(chunk)}.json")
        if os.path.exists(part):
            for ml, rels in json.load(open(part)).items():
                if ml == "_years":
                    out.setdefault("_years", {}).update({int(k): v for k, v in rels.items()})
                else:
                    out[int(ml)] = {r: [tuple(v) for v in vals] for r, vals in rels.items()}
            continue
        got: dict = {}
        values = " ".join(f'"{i}"' for i in chunk)
        props = " ".join(f"wdt:{p}" for p in PROPS)
        q = f"""SELECT ?ml ?p ?v ?vLabel WHERE {{
  VALUES ?ml {{ {values} }}
  ?film wdt:P14449 ?ml .
  VALUES ?p {{ {props} }}
  ?film ?p ?v .
  ?v rdfs:label ?vLabel .
  FILTER(LANG(?vLabel) = "en")
}}"""
        years: dict = {}
        for b in sparql(f"""SELECT ?ml (MIN(YEAR(?d)) AS ?y) WHERE {{
  VALUES ?ml {{ {values} }}
  ?film wdt:P14449 ?ml .
  ?film wdt:P577 ?d .
}} GROUP BY ?ml"""):
            years[int(b["ml"]["value"])] = int(float(b["y"]["value"]))
        got["_years"] = {str(k): v for k, v in years.items()}
        for b in sparql(q):
            if "vLabel" not in b:
                continue
            ml = int(b["ml"]["value"])
            rel = PROPS[b["p"]["value"].rsplit("/", 1)[1]]
            qid = b["v"]["value"].rsplit("/", 1)[1]
            got.setdefault(ml, {}).setdefault(rel, [])
            if (qid, b["vLabel"]["value"]) not in got[ml][rel]:
                got[ml][rel].append((qid, b["vLabel"]["value"]))
        with open(part + ".tmp", "w") as f:
            json.dump({str(k): v for k, v in got.items()}, f)
        os.replace(part + ".tmp", part)
        out.setdefault("_years", {}).update(years)
        out.update({k: v for k, v in got.items() if k != "_years"})
        print(f"[wikidata] {min(s + batch, len(ids))}/{len(ids)} movies queried, {len(out) - 1} found", flush=True)
        time.sleep(float(os.environ.get("WD_SLEEP", 2)))
    return out


def main(src: str, plots_path: str, dst: str) -> None:
    items = [json.loads(line) for line in open(os.path.join(src, "items.jsonl"))]
    plots = {}
    with open(plots_path, encoding="utf-8") as f:
        next(f)
        for line in f:
            mid, _, text = line.rstrip("\n").partition("\t")
            if mid.isdigit() and text.strip():
                plots[int(mid)] = text.strip()
    ids = [int(it["id"]) for it in items]
    wd = wikidata(ids, os.path.join(os.path.dirname(plots_path), "wikidata_batches"))
    years = wd.pop("_years", {})
    stats = {"items": len(items), "with_plot": 0, "with_wikidata": 0, "rejected_year": 0}
    for it in items:                                   # keep a Wikidata match only if the release year agrees
        ml, m = int(it["id"]), re.search(r"\((\d{4})\)\s*$", it.get("title", ""))
        if ml in wd and m and ml in years and abs(years[ml] - int(m.group(1))) > 1:
            del wd[ml]
            stats["rejected_year"] += 1
    for rel in PROPS.values():
        stats[f"with_{rel.replace(' ', '_')}"] = 0
    out = []
    for it in items:
        ml = int(it["id"])
        it = dict(it)
        if ml in plots:
            it["desc"] = plots[ml]
            stats["with_plot"] += 1
        rels = wd.get(ml, {})
        if rels:
            stats["with_wikidata"] += 1
        attrs = [list(a) for a in it.get("attrs", [])]
        meta = [it.get("meta", "")]
        for rel in PROPS.values():
            vals = rels.get(rel, [])[: MAX_PER.get(rel, 3)]
            if not vals:
                continue
            stats[f"with_{rel.replace(' ', '_')}"] += 1
            for _, label in vals:
                attrs.append([rel, label])
            if rel in ("director", "cast", "series", "based on"):
                meta.append(f"{rel.capitalize()}: {', '.join(l for _, l in vals)}")
        it["attrs"] = attrs
        it["meta"] = "; ".join(m for m in meta if m)
        out.append(it)
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "items.jsonl"), "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    shutil.copy(os.path.join(src, "seqs.jsonl"), os.path.join(dst, "seqs.jsonl"))
    meta = json.load(open(os.path.join(src, "meta.json")))
    meta["name"] = "ml-1m-plus"
    meta["enrichment"] = {"plots": "Spillo et al., Binge Watch (M3L), Zenodo 18499145, CC BY 4.0",
                          "properties": "Wikidata (CC0) via QLever, matched by P14449 MovieLens movie ID, release year within one year", **stats}
    with open(os.path.join(dst, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    with open(os.path.join(dst, "wikidata.json"), "w") as f:
        json.dump({str(k): v for k, v in wd.items()}, f)
    print("example:", json.dumps(out[0], ensure_ascii=False)[:600])
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:4])
