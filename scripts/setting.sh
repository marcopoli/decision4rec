#!/usr/bin/env bash
# One command per experimental setting.
#
#   bash scripts/setting.sh small all     preliminary experiment on small datasets (RTX 3090)
#   bash scripts/setting.sh large all     confirmatory experiment on RecSys-scale datasets (Mac Studio)
#
# `all` runs every stage in order, installs what is missing, starts and stops the decision-model
# server by itself, and writes one results archive at the end. It is resumable: run the same command
# again and finished stages are skipped. A single stage can be run by name instead of `all`.
#
# Every stage runs by default. Core stages stop the run on failure. The agent stage (the agent on ActiveGraph,
# primary row of the benchmark) and the model stages (rec-tuning, other model sizes, a generative LLM, hosted Jev)
# are non-blocking: a failure is logged, the run continues and the archive is still written.
#
# Environment (all optional):
#   DATASETS   space-separated subset of the setting's datasets
#   MODELS     comma-separated subset of baselines
#   MAX_USERS  cap the users of the large datasets
#   SKIP       space-separated stages to skip, for example SKIP="rectune logit"
#   FORCE=1    ignore the markers of finished stages
#   DRY_RUN=1  print the commands without running them
#   KEV_DIR    where Kev is installed (default ~/kev)     PORT  server port (default 8009)
#   TYPESAFE_API_KEY  if set, the small setting also runs hosted Jev on one dataset
set -euo pipefail

setting="${1:?usage: setting.sh <small|large> <stage|all>}"
stage="${2:-all}"
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
RAW=data/raw
AMZ=https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw
KEV_DIR="${KEV_DIR:-$HOME/kev}"
PORT="${PORT:-8009}"
LOGS="logs/$setting"
PY="$ROOT/.venv/bin/python"
# The Kev server takes the whole GPU (it showed 24 GB of 24 on the 3090). Our own steps therefore run with the GPU
# hidden, on the CPU, whenever the server may be up. Only the steps that train models or read an LLM through its
# logits use the GPU, and they run while the server is stopped.
J="env CUDA_VISIBLE_DEVICES= $PY -m decision4rec.cli"
JG="$PY -m decision4rec.cli"
B=configs/backends
# RUN_MAX / RUN_MAX_VALID (the 27B runs on the Spark): only the first test / validation users of each domain, no report
RUN_LIMIT="${RUN_MAX:+--max-cases $RUN_MAX --no-report}${RUN_MAX_VALID:+ --max-valid $RUN_MAX_VALID}"
export PYTORCH_ENABLE_MPS_FALLBACK=1     # Apple Silicon: operators missing on MPS fall back to the CPU
# PATH_FIRST (the Spark, shared account): our own tools come before those of the shared home.
export PATH="${PATH_FIRST:+$PATH_FIRST:}$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
unset PYTHONPATH PYTHONHOME              # packages of another Python must not leak into .venv

case "$setting" in
  small) all_datasets="ml1m vg mi" ;;
  large) all_datasets="ml20m movies books steam" ;;
  *) echo "unknown setting $setting"; exit 1 ;;
esac
DATASETS="${DATASETS:-$all_datasets}"
first="${DATASETS%% *}"
if command -v nvidia-smi >/dev/null 2>&1; then DEVICE=cuda; elif [ "$(uname)" = Darwin ]; then DEVICE=mps; else DEVICE=cpu; fi

say() { echo "[$setting $(date '+%m-%d %H:%M')] $*"; }
run() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "+ $*"; else "$@"; fi; }
# Kev has its own uv environment: an activated .venv of this project must not leak into it.
in_kev() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "+ (cd $KEV_DIR && $*)"; else (cd "$KEV_DIR" && env -u VIRTUAL_ENV "$@"); fi; }
cfg() { echo "configs/$setting/$1_$2.yaml"; }
skipped() { case " ${SKIP:-} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

folder() {
  case "$1" in
    ml1m) echo ml1m ;; vg) echo video_games ;; mi) echo musical_instruments ;; ind) echo industrial_scientific ;;
    ml20m) echo ml20m ;; movies) echo movies_tv ;; books) echo books ;; steam) echo steam ;;
  esac
}

fetch() {    # resumable download into $RAW with wget or curl (macOS ships curl only)
  mkdir -p "$RAW"
  for url in "$@"; do
    if command -v wget >/dev/null 2>&1; then run wget -nc -P "$RAW" "$url"
    else (cd "$RAW" && run curl -L --fail -C - -O "$url"); fi
  done
}

amazon() {   # tag, category name
  fetch "$AMZ/review_categories/$2.jsonl.gz" "$AMZ/meta_categories/meta_$2.jsonl.gz"
  run $J prepare --dataset amazon23 --reviews "$RAW/$2.jsonl.gz" --meta "$RAW/meta_$2.jsonl.gz" --name "$(folder "$1")" \
      --out "data/processed/$(folder "$1")" ${MAX_USERS:+--max-users $MAX_USERS}
}

prepare() {
  [ -f "data/processed/$(folder "$1")/meta.json" ] && { say "dataset $1 already prepared"; return 0; }
  case "$1" in
    ml1m)  fetch https://files.grouplens.org/datasets/movielens/ml-1m.zip; run unzip -o -q "$RAW/ml-1m.zip" -d "$RAW"
           run $J prepare --dataset ml1m --folder "$RAW/ml-1m" --out data/processed/ml1m ;;
    vg)    amazon vg Video_Games ;;
    mi)    amazon mi Musical_Instruments ;;
    ind)   amazon ind Industrial_and_Scientific ;;
    ml20m) fetch https://files.grouplens.org/datasets/movielens/ml-20m.zip; run unzip -o -q "$RAW/ml-20m.zip" -d "$RAW"
           run $J prepare --dataset mlcsv --folder "$RAW/ml-20m" --name ml20m --out data/processed/ml20m ${MAX_USERS:+--max-users $MAX_USERS} ;;
    movies) amazon movies Movies_and_TV ;;
    books)  amazon books Books ;;
    steam) fetch https://cseweb.ucsd.edu/~wckang/steam_reviews.json.gz https://cseweb.ucsd.edu/~wckang/steam_games.json.gz
           run $J prepare --dataset steam --reviews "$RAW/steam_reviews.json.gz" --meta "$RAW/steam_games.json.gz" \
               --out data/processed/steam ${MAX_USERS:+--max-users $MAX_USERS} ;;
  esac
}

# ---- decision-model server ------------------------------------------------

port_busy() { "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(('127.0.0.1', $PORT)) != 0)" 2>/dev/null; }

server_stop() {   # the recorded process is `uv run`; the model process is its child, so both are stopped
  if [ -f "$LOGS/server.pid" ]; then
    local pid; pid="$(cat "$LOGS/server.pid")"
    run pkill -TERM -P "$pid" 2>/dev/null || true
    run kill "$pid" 2>/dev/null || true
    run rm -f "$LOGS/server.pid" "$LOGS/server.ckpt"   # through run: a dry run must not lose the record (2026-10-12)
    [ "${DRY_RUN:-0}" = 1 ] || sleep 15          # let the GPU memory and the port be released
  fi
}

# Kev's precompiled CUDA graphs took 5.3 GB of the 3090 and the server then ran out of memory on a normal request
# (2026-10-04). They are off by default: slower, but the model keeps the memory it needs. KEV_CUDA_GRAPHS=1 turns
# them on again. Neither setting has any effect on Apple silicon.
KEV_GRAPHS="${KEV_CUDA_GRAPHS:-0}"
ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

server_start() {   # checkpoint (Hub id or local run folder). A server already serving it is reused.
  if [ -f "$LOGS/server.pid" ] && [ "$(cat "$LOGS/server.ckpt" 2>/dev/null)" = "$1" ] \
     && kill -0 "$(cat "$LOGS/server.pid")" 2>/dev/null; then
    return 0
  fi
  server_stop
  if [ "${DRY_RUN:-0}" != 1 ] && port_busy; then   # never let an unknown server answer for another model
    say "port $PORT is already used by another process. Stop it (pkill -f kev.serve) and run the command again."
    return 1
  fi
  say "starting the decision-model server with $1 (Kev picks the GPU by itself)"
  # kev.serve accepts only --run, --fallback, --host and --port (checked on 2026-10-03).
  if [ "${DRY_RUN:-0}" = 1 ]; then echo "+ (cd $KEV_DIR && uv run --extra serve python -m kev.serve --run $1 --port $PORT &)"; return 0; fi
  # The background process redirects its own output first and then replaces itself with the server (exec),
  # so no shell stays alive holding the stage's log pipe: otherwise the stage could never finish.
  (exec > "$ROOT/$LOGS/server_$(basename "$1").log" 2>&1 < /dev/null
   cd "$KEV_DIR" && exec nohup env -u VIRTUAL_ENV KEV_CUDA_GRAPHS="$KEV_GRAPHS" PYTORCH_CUDA_ALLOC_CONF="$ALLOC_CONF" \
     uv run --extra serve python -m kev.serve --run "$1" --port "$PORT") &
  echo $! > "$LOGS/server.pid"
  echo "$1" > "$LOGS/server.ckpt"
  for _ in $(seq 1 720); do                      # up to two hours: the first start downloads the weights
    if "$PY" -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(s.connect_ex(('127.0.0.1', $PORT)))" 2>/dev/null; then
      say "server is up"; return 0
    fi
    kill -0 "$(cat "$LOGS/server.pid")" 2>/dev/null || { say "the server process died, see $LOGS/server_$(basename "$1").log"; return 1; }
    sleep 10
  done
  say "the server did not come up in time"; return 1
}

benchmark_all() {  # backend file, extra benchmark arguments
  local configs=""
  for d in $DATASETS; do configs="$configs $(cfg "$d" main)"; done
  run $J benchmark --config $configs --backend "$1" "${@:2}"
}

system_with() {    # backend file: the fixed system on every dataset, then the joint table
  for d in $DATASETS; do run $J run --config "$(cfg "$d" main)" --backend "$1" --no-report; done
  benchmark_all "$1"
}

# ---- stages -----------------------------------------------------------------

stage_setup() {
  run bash scripts/env.sh                   # creates .venv, or rebuilds one that this machine cannot load
  run "$PY" -m pytest -q
}

stage_data() { for d in $DATASETS; do prepare "$d"; done; }

stage_retriever() {
  for d in $DATASETS; do
    [ -f "runs/$setting/$d/sasrec/sasrec.json" ] && continue
    local extra=""; case "$d" in ml1m|ml20m) extra="--max-len 200 --batch-size 128" ;; esac
    run $JG train-retriever --dataset "data/processed/$(folder "$d")" --out "runs/$setting/$d/sasrec" $extra
  done
}

stage_baselines() {   # one dataset, one model, one configuration at a time
  for d in $DATASETS; do run $JG baselines --config "$(cfg "$d" main)" ${MODELS:+--models $MODELS}; done
}

stage_kev() {
  command -v uv >/dev/null 2>&1 || { say "installing uv"; run sh -c "curl -LsSf https://astral.sh/uv/install.sh | sh"; }
  [ -d "$KEV_DIR" ] || run git clone https://github.com/jaredpalmer/kev.git "$KEV_DIR"
  in_kev uv sync --extra serve
}

stage_check() {
  server_start jaredpalmer/kev-4b
  run $J check-backend --config "$(cfg "$first" main)" --backend "$B/kev-4b.yaml"
  run $J run --config "$(cfg "$first" main)" --backend "$B/kev-4b.yaml" --max-cases 20
}

stage_system() { server_start jaredpalmer/kev-4b; system_with "$B/kev-4b.yaml"; }

stage_ablation() {
  server_start jaredpalmer/kev-4b
  for d in $DATASETS; do
    run $J run --config "$(cfg "$d" main_sasrec)" --backend "$B/kev-4b.yaml"
    run $J retrieval --config "$(cfg "$d" main)" --n-users 1000
  done
}

stage_agent() {       # the agent on ActiveGraph on every validation and test user: the primary row of the benchmark
  server_start jaredpalmer/kev-4b
  for d in $DATASETS; do run $J agent-bench --config "$(cfg "$d" main)" --backend "$B/kev-4b.yaml"; done
  benchmark_all "$B/kev-4b.yaml"
  if [ "$setting" = small ]; then    # policies, incremental user model, replay and counterfactuals on 100 users
    for d in $DATASETS; do [ "$d" = mi ] || run $J agent --config "$(cfg "$d" agent)" --backend "$B/kev-4b.yaml" --n-cases 100; done
  fi
}

stage_trust() {
  server_start jaredpalmer/kev-4b
  for d in $DATASETS; do
    run $J run --config "$(cfg "$d" trust)" --backend "$B/kev-4b.yaml"
    run $J run --config "$(cfg "$d" outside)" --backend "$B/kev-4b.yaml"
    run $J audit --config "$(cfg "$d" agent)" --backend "$B/kev-4b.yaml" --n-cases 200 --block 8
  done
}

stage_appendix() { server_start jaredpalmer/kev-4b; run $J rating --config "$(cfg ml1m outside)" --backend "$B/kev-4b.yaml" --n-cases 1000; }

stage_rectune() {     # rec-tuned Kev-4B: in-domain on Video Games, cross-domain on the other datasets
  run $J export-rectune --config "$(cfg vg main_sasrec)" --out "runs/$setting/vg/rectune" --n-train 20000 --n-heldout 1000 --block 10 --item-chars 120   # Kev trains on rows of at most 1,024 tokens
  server_stop
  in_kev uv run python -m kev.train --data "$ROOT/runs/$setting/vg/rectune/train.jsonl" \
      --base Qwen/Qwen3.5-4B-Base --init_from jaredpalmer/kev-4b --epochs 2 --lr 2e-5 --batch 1 --accum 8 \
      --dtype bf16 --checkpointing 1 --device "$DEVICE" --out runs/kev-4b-rec-vg
  server_start runs/kev-4b-rec-vg
  system_with "$B/kev-4b-rectuned.yaml"
}

stage_model_small() { server_start jaredpalmer/kev-0.8b; system_with "$B/kev-0.8b.yaml"; }

stage_model_mid() { server_start jaredpalmer/kev-9b; system_with "$B/kev-9b.yaml"; }   # tight on 24 GB

stage_model_large() { server_start jaredpalmer/kev-27b; system_with "$B/kev-27b.yaml"; }

stage_logit() {       # a generative LLM read through next-token probabilities, as an extra benchmark row
  server_stop
  for d in $DATASETS; do run $JG run --config "$(cfg "$d" main)" --backend "$B/qwen2.5-7b-logit.yaml" --arms retriever,luce_m10_r2; done
  benchmark_all "$B/kev-4b.yaml" --extra "$B/qwen2.5-7b-logit.yaml:luce_m10_r2:Qwen2.5-7B (logits, blocks)"
}

stage_jev() {         # hosted Jev, only when a key is provided
  [ -n "${TYPESAFE_API_KEY:-}" ] || { say "TYPESAFE_API_KEY is not set, hosted Jev is skipped"; return 0; }
  run $J run --config "$(cfg vg main)" --backend "$B/jev.yaml"
}

stage_confirm() {     # pre-registered test of the long-tail gate on fresh users (docs/04_confirmatory_plan.md)
  [ "$setting" = small ] || { say "the confirmatory test exists only in the small setting"; return 1; }
  server_start jaredpalmer/kev-4b
  for d in vg mi; do run $J run --config "$(cfg "$d" confirm)" --backend "$B/kev-4b.yaml"; done
  server_stop           # the baselines score on the GPU (trained checkpoints are reused)
  for d in vg mi; do run $JG baselines --config "$(cfg "$d" confirm)" --models gru4rec,sasrec,sasrecf; done
  # training-free baselines asked for by recent critiques; reported next to the test, not part of its claim
  for d in vg mi; do run $J baselines --config "$(cfg "$d" confirm)" --models seqrules,contentknn,transtext; done
  run $J asc --runs "runs/$setting/vg/vg_confirm_kev-4b" "runs/$setting/mi/mi_confirm_kev-4b" \
    --arms wholepool@p0 --only gate,shrink --head 100 --out "runs/$setting/confirm.md"
}

# The semantic layer (docs/03_architecture_v2.md, sections 9 and 10): the decision model answers questions decided
# by text, the trained model keeps learning behaviour. Run on request, in this order: e1_facets, e2_relations,
# e3_cold; e4_generalization repeats all three on two more domains. SEM lists the datasets (default: vg mi).
sem_only_small() { [ "$setting" = small ] || { say "the semantic-layer stages exist only in the small setting"; return 1; }; }

stage_e1_facets() {   # E1: item facets from Kev-4B, then SASRecF with them, against SASRecF on every user
  sem_only_small || return 1
  local d sem="${SEM:-vg mi}" cfgs=""
  server_start jaredpalmer/kev-4b
  for d in $sem; do   # 50 items first: their summary (agreement with metadata) shows early whether the questions work
    run $J facets --config "$(cfg "$d" semantic)" --backend "$B/kev-4b.yaml" --limit 50
    run $J facets --config "$(cfg "$d" semantic)" --backend "$B/kev-4b.yaml"
  done
  server_stop           # the embedding control and the baselines need no decision model
  for d in $sem; do run $J facets --config "$(cfg "$d" semantic)" --embedding; done
  for d in $sem; do run $JG baselines --config "$(cfg "$d" semantic)"; cfgs="$cfgs $(cfg "$d" semantic)"; done
  run $J facet-report --config $cfgs --out "runs/$setting/e1_facets${SEM_TAG:-}.md"
}

stage_e2_relations() {   # E2: typed relations of item pairs from Kev-4B, added to the trained backbone
  sem_only_small || return 1
  local d sem="${SEM:-vg mi}" cfgs="" pairs=""
  server_start jaredpalmer/kev-4b
  for d in $sem; do
    run $J run --config "$(cfg "$d" semantic)" --backend "$B/kev-4b.yaml"     # whole-pool answers, cached for vg and mi
    run $J relations --config "$(cfg "$d" semantic)" --backend "$B/kev-4b.yaml"
    cfgs="$cfgs $(cfg "$d" semantic)"
    pairs="$pairs --pair $(cfg "$d" semantic) runs/$setting/$d/${d}_semantic_kev-4b"
  done
  server_stop
  run $J relation-eval --config $cfgs --out "runs/$setting/e2_relations${SEM_TAG:-}.md"
  # the system row: best trained backbone (facets of E1 included) plus relations and the decision model's gate
  run $J system-eval $pairs --out "runs/$setting/e2_system${SEM_TAG:-}.md"
}

stage_e3_cold() {     # E3: global temporal split with new items; reuses the facets of E1
  sem_only_small || return 1
  local d src sem="${SEM:-vg mi}" runs="" pairs=""
  local -a strat=()
  for d in $sem; do
    [ "${DRY_RUN:-0}" = 1 ] || [ -s "runs/$setting/$d/facets/kev-4b.jsonl" ] \
      || { say "E3 reuses the facets of E1: run the stage e1_facets first"; return 1; }
  done
  for d in $sem; do
    src="data/processed/$(folder "$d")"
    run $J prepare-cold --src "$src" --out "${src}_cold" --quantile 0.9 --min-pre 3
    [ -f "runs/$setting/${d}_cold/sasrec/sasrec.json" ] || run $JG train-retriever --dataset "${src}_cold" --out "runs/$setting/${d}_cold/sasrec"
    run mkdir -p "runs/$setting/${d}_cold/retrieval"   # item-text embeddings do not depend on interactions: reuse them
    run cp -n runs/$setting/$d/retrieval/emb_*_content_*.npy "runs/$setting/${d}_cold/retrieval/" 2>/dev/null || true
  done
  server_start jaredpalmer/kev-4b
  for d in $sem; do
    run $J run --config "$(cfg "$d" cold)" --backend "$B/kev-4b.yaml"
    run $J relations --config "$(cfg "$d" cold)" --backend "$B/kev-4b.yaml"
  done
  server_stop
  for d in $sem; do
    run $JG baselines --config "$(cfg "$d" cold)"
    runs="$runs runs/$setting/${d}_cold/${d}_cold_kev-4b"
    pairs="$pairs --pair $(cfg "$d" cold) runs/$setting/${d}_cold/${d}_cold_kev-4b"
    strat+=(--pair sasrecf "${d}_cold_kev-4b: wholepool@p0")
  done
  run $J stratify --runs $runs --pair sasrecf sasrecf_dm "${strat[@]}" --out "runs/$setting/e3_cold_stratify${SEM_TAG:-}.md"
  run $J asc --runs $runs --arms wholepool@p0 --only dm,fuse,gate,shrink --head 1 --out "runs/$setting/e3_cold_asc${SEM_TAG:-}.md"
  run $J system-eval $pairs --out "runs/$setting/e3_system${SEM_TAG:-}.md"
}

stage_e4_generalization() {   # E1 to E3 on two more domains: Amazon Industrial and Scientific, and Steam (not Amazon)
  sem_only_small || return 1
  local d
  for d in ind steam; do
    if [ "$d" = steam ]; then MAX_USERS="${MAX_USERS:-100000}" prepare "$d"; else prepare "$d"; fi
    [ -f "runs/$setting/$d/sasrec/sasrec.json" ] || run $JG train-retriever --dataset "data/processed/$(folder "$d")" --out "runs/$setting/$d/sasrec"
  done
  SEM="ind steam" SEM_TAG="_generalization" stage_e1_facets
  SEM="ind steam" SEM_TAG="_generalization" stage_e2_relations
  SEM="ind steam" SEM_TAG="_generalization" stage_e3_cold
}

# Decision4Rec-T (docs/07_decision4rec_t.md): Kev trained further on real next-item choices (Kev-Rec) on Video Games, Musical
# Instruments and Steam; Industrial and Scientific is held out as a domain the decision model never saw. Run on
# request, in this order: rt_export, rt_train_08, rt_eval_08, rt_train_4, rt_eval_4.
# RTV picks the variant: rt (the decision model reads item text only), rte (item text and the scenario's behavioural
# statements on every option) or rtm (the full architecture: the multi-agent pool with SASRecF as one agent, its
# residual fast ranker, statements and the agents that found each candidate; paper Section 4.6). Each variant has its own configs, records, model and report.
RTV="${RTV:-rt}"
# RT_DOMAINS are the domains the records come from (default: the three source domains). RT_MODEL names the trained
# model, its records and its runs (default: the variant); for example RTV=rte RT_MODEL=rteall RT_DOMAINS="vg mi steam
# ind" trains on all four domains, so Industrial and Scientific is read both by a model that never saw it (rte) and by
# one trained on it, as SASRecF is (rteall).
RT_DOMAINS="${RT_DOMAINS:-vg mi steam}"
RT_MODEL="${RT_MODEL:-$RTV}"
if [ "$RT_MODEL" = rt ]; then RT_DIR="runs/$setting/rectune_choices"; else RT_DIR="runs/$setting/rectune_choices_$RT_MODEL"; fi
case "$RTV" in rt) RT_STATE=640 ;; rte) RT_STATE=1536 ;; rtmc) RT_STATE=3200 ;; *) RT_STATE=1792 ;; esac   # longer rows: statements, "found by"; rtmc: 250 characters per item, rows up to ~3.7k tokens (row limit = RT_STATE + 640)
rt_only_small() { [ "$setting" = small ] || { say "Decision4Rec-T exists only in the small setting"; return 1; }; }

stage_rt_export() {   # training records: real choices over the SASRecF top 30, 15,000 users per domain
  rt_only_small || return 1
  server_stop           # SASRecF scores the training histories on the GPU
  local d configs=""
  for d in $RT_DOMAINS; do configs="$configs $(cfg "$d" "$RTV")"; done
  run $JG export-choices --config $configs --out "$RT_DIR" \
    --n-per-dataset 15000 --n-heldout 500 --offsets "${RT_OFFSETS:-0}"
}

rt_train() {   # size (0.8b or 4b), base model, base revision (as in the Kev README), learning rate, weights dtype
  local size="$1" base="$2" rev="$3" lr="$4" wdt="$5" batch="$6" accum="$7" f steps
  rt_only_small || return 1
  # RT_STEPS_08 and RT_STEPS_4 cap the optimizer steps of each size (8 records per step); 0 means one full epoch.
  # RT_BATCH_* and RT_ACCUM_* change the micro-batch; batch x accumulation stays 8 records per step by default.
  if [ "$size" = 27b ]; then
    steps="${RT_STEPS_27:-0}"; batch="${RT_BATCH_27:-$batch}"; accum="${RT_ACCUM_27:-$accum}"
  elif [ "$size" = 9b ]; then
    steps="${RT_STEPS_9:-0}"; batch="${RT_BATCH_9:-$batch}"; accum="${RT_ACCUM_9:-$accum}"
  elif [ "$size" = 4b ]; then
    steps="${RT_STEPS_4:-0}"; batch="${RT_BATCH_4:-$batch}"; accum="${RT_ACCUM_4:-$accum}"
  else
    steps="${RT_STEPS_08:-0}"; batch="${RT_BATCH_08:-$batch}"; accum="${RT_ACCUM_08:-$accum}"
  fi
  # Older records had cues absent at test time: version 1 added a missing target to the candidates (the odd one
  # out is the answer), version 2 took targets SASRecF had been trained on. Export again unless they are version 3.
  if [ "${DRY_RUN:-0}" != 1 ] && ! grep -q '"version": 3' "$RT_DIR/export_info.json" 2>/dev/null; then
    say "training records missing or of an old kind: exporting them again"
    stage_rt_export
  fi
  # RT_OFFSETS > 0 (fusion-aware training): the records carry the fast ranker's score of every option and Kev's trainer
  # adds it to the logits in the loss (scripts/kev_offset_train.py, same arguments as kev.train)
  local trainer="-m kev.train"
  if [ "${RT_OFFSETS:-0}" != 0 ]; then
    if [ "${DRY_RUN:-0}" != 1 ] && ! grep -q '"offsets": '"$RT_OFFSETS" "$RT_DIR/export_info.json" 2>/dev/null; then
      say "records without offsets $RT_OFFSETS: exporting them again"
      stage_rt_export
    fi
    trainer="$ROOT/scripts/kev_offset_train.py"
  fi
  server_stop
  # A new checkpoint makes the old one, and every answer cached under its name, stale.
  run rm -rf "$KEV_DIR/runs/kev-$size-$RT_MODEL" "$KEV_DIR/runs/kev-$size-$RT_MODEL-eval" runs/"$setting"/*/*_kev-"$size"-"$RT_MODEL" \
    runs/"$setting"/*/cache/kev-"$size"-"$RT_MODEL".sqlite
  # Augmentation is off: Kev's trainer checks the row length before it adds options, so added options could
  # overflow a row (what made the old rectune stage fail). --max_state raises the row limit to 1024 + max_state - 384
  # tokens (1,280 for rt, 2,176 for rte); longer records are dropped by the trainer, which prints how many.
  in_kev uv run python $trainer --data "$ROOT/$RT_DIR/train.jsonl" --base "$base" --base_revision "$rev" \
    --init_from "${RT_INIT:-jaredpalmer/kev-$size}" --epochs 1 --lr "$lr" --batch "$batch" --accum "$accum" --length_sort 1 --dtype bf16 --weights_dtype "$wdt" \
    --checkpointing 1 --max_state "$RT_STATE" --p_none 0 --p_none_distract 0 --p_distract 0 --device "$DEVICE" \
    --max_steps "$steps" --seed "${RT_SEED:-0}" --out "runs/kev-$size-$RT_MODEL"
  # No kev.benchmark on the held-out records: it encodes within the training default of 1,024 tokens per row and
  # rejects most of them; the evaluation runs (rt_eval) are the measure.
  run mkdir -p "$RT_DIR/kev-$size-$RT_MODEL"
  for f in training_config.json training_metrics.json; do run cp "$KEV_DIR/runs/kev-$size-$RT_MODEL/$f" "$RT_DIR/kev-$size-$RT_MODEL/" || true; done
}

# Micro-batches as in the Kev README recipes (0.8B: 8 records per pass; 4B: 4 records, 2 passes per step): with one
# record per pass the GPU mostly waits on kernel launches.
# R17, a LlamaRec-style reranker: Qwen3.5-0.8B-Base with adapters of rank 16, trained on the records of RT_MODEL with
# the decision model's budget (one epoch, 8 records per step, lr 1e-4) and read through option-label logits by the
# logit backend (configs/backends/llm-0.8b-<RT_MODEL>.yaml), with two orders per request and the same fusion.
# LLM_TAG, LLM_BASE and LLM_REV choose the starting model (default: v1, Qwen3.5-0.8B-Base). R17 v2 (2026-10-15):
# LLM_TAG=llmi-0.8b LLM_BASE=Qwen/Qwen3.5-0.8B LLM_REV=2fc06364715b967f1860aea9cf38778875588b17, the post-trained model,
# because v1 did not learn (uniform label scores); its backend is configs/backends/<LLM_TAG>-<RT_MODEL>.yaml.
# R17 v3 (2026-10-15): v2 did not learn either, but the base model learns over 5 options (logs/probe/llm_pilot_k.py), so
# LLM_TAG=llmb5-0.8b LLM_OPTIONS=5 trains on 5-option questions and RTV=rtmb5 reads it over blocks of 5 (Luce fit).
LLM_TAG="${LLM_TAG:-llm-0.8b}"
LLM_BASE="${LLM_BASE:-Qwen/Qwen3.5-0.8B-Base}"
LLM_REV="${LLM_REV:-dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68}"
stage_llm_train_08() {
  rt_only_small || return 1
  [ "${DRY_RUN:-0}" = 1 ] || [ -s "$RT_DIR/train.jsonl" ] || { say "no records in $RT_DIR"; return 1; }
  server_stop
  run rm -rf "runs/$LLM_TAG-$RT_MODEL" runs/"$setting"/*/*_"$LLM_TAG"-"$RT_MODEL" runs/"$setting"/*/cache/"$LLM_TAG"-"$RT_MODEL".sqlite
  run $JG train-llm-ranker --data "$RT_DIR/train.jsonl" --base "$LLM_BASE" \
    --revision "$LLM_REV" --out "runs/$LLM_TAG-$RT_MODEL" --lr 1e-4 --lora 16 --accum 8 --options "${LLM_OPTIONS:-0}"
}

stage_llm_eval_08() {
  local d runs=""
  rt_only_small || return 1
  server_stop           # the model is read in-process on the GPU
  for d in vg mi steam ind; do
    run $JG run --config "$(cfg "$d" "$RTV")" --backend "$B/$LLM_TAG-$RT_MODEL.yaml"
    runs="$runs runs/$setting/$d/${d}_${RTV}_$LLM_TAG-$RT_MODEL"
  done
  run $J asc --runs $runs --arms "$RTV" --only dm,fuse,gate,shrink --head 100 --out "runs/$setting/${LLM_TAG%%-*}_${RT_MODEL}_eval_0.8b.md"
}

# R16, seeds: RT_MODEL=rtm_s1 RT_SEED=1 RT_SEED_FROM=rtm trains the same records again with another seed of Kev's
# trainer (data order, option order, adapter initialization). The records are hard links to those of RT_SEED_FROM.
stage_seed_prep() {
  rt_only_small || return 1
  local src="runs/$setting/rectune_choices_${RT_SEED_FROM:?RT_SEED_FROM names the variant whose records are reused}"
  [ -s "$src/export_info.json" ] || { say "no records in $src"; return 1; }
  run mkdir -p "$RT_DIR"
  run cp -lf "$src"/*.jsonl "$src/export_info.json" "$RT_DIR/"
}

stage_rt_train_08() { rt_train 0.8b Qwen/Qwen3.5-0.8B-Base dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 1e-4 fp32 8 1; }
stage_rt_train_4() { rt_train 4b Qwen/Qwen3.5-4B-Base 1001bb4d826a52d1f399e183466143f4da7b741b 5e-5 bf16 4 2; }

rt_eval() {   # size: Kev-Rec of that size on the development users of every domain, then the fusion report
  local size="$1" d runs=""
  rt_only_small || return 1
  server_start "runs/kev-$size-$RT_MODEL"
  for d in vg mi steam ind; do
    run $J run --config "$(cfg "$d" "$RTV")" --backend "$B/kev-$size-$RT_MODEL.yaml" $RUN_LIMIT
    runs="$runs runs/$setting/$d/${d}_${RTV}_kev-$size-$RT_MODEL"
  done
  [ -n "${RUN_MAX:-}" ] && { server_stop; say "partial runs (RUN_MAX=$RUN_MAX): compare them with logs/probe/arm_pairs.py on the common users"; return 0; }
  server_start jaredpalmer/kev-4b    # the released model on the same users shows what training adds (cached after once)
  for d in vg mi steam ind; do
    run $J run --config "$(cfg "$d" "$RTV")" --backend "$B/kev-4b.yaml" --arms "retriever,$RTV"
    runs="$runs runs/$setting/$d/${d}_${RTV}_kev-4b"
  done
  server_stop           # the baselines score the same users on the GPU (trained checkpoints are reused)
  for d in vg mi steam ind; do run $JG baselines --config "$(cfg "$d" "$RTV")"; done
  run $J asc --runs $runs --arms "$RTV" --only dm,fuse,gate,shrink --head 100 --out "runs/$setting/${RT_MODEL}_eval_$size.md"
}

stage_rt_eval_08() { rt_eval 0.8b; }
# R3b (2026-10-15, the Spark): Kev-27B trained like the 0.8B and 4B pool readers, an adapter on a frozen base started
# from the released adapter: base Qwen/Qwen3.8-27B (revision of the Kev-27B model card), warm start from Kev-27B v1, the
# rank-16 LoRA release (tag v1-lora); Kev-27B v2 is a full fine-tune and has no adapter to start from.
# RT_STEPS_27 caps the steps (a speed probe uses a few); RT_LR_27 the learning rate.
stage_rt_train_27() { RT_INIT="${RT_INIT:-jaredpalmer/kev-27b@v1-lora}" rt_train 27b Qwen/Qwen3.8-27B 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 "${RT_LR_27:-2e-5}" bf16 1 8; }
stage_rt_eval_27() { rt_eval 27b; }
stage_rt_eval_4() { rt_eval 4b; }
# R21 (2026-10-15, the Spark): Kev-9B in the size curve, an adapter on the frozen Qwen3.5-9B-Base (revision of the Kev-9B
# model card) warm-started from the released Kev-9B, with the recipe of the 4B (learning rate 5e-5, 8 records per step).
stage_rt_train_9() { rt_train 9b Qwen/Qwen3.5-9B-Base 68c46c4b3498877f3ef123c856ecfde50c39f404 "${RT_LR_9:-5e-5}" bf16 2 4; }
stage_rt_eval_9() { rt_eval 9b; }

# Data-matched runs (2026-10-15): RT_MODEL=rtm8k RT_SUBSET_FROM=rtm RT_SUBSET=8000 keeps a fixed random subset of the
# records of RT_SUBSET_FROM (scripts/subset_records.py, seed 2026), read by the 0.8B, 4B, 9B and 27B alike (R3b, R21);
# RT_SUBSET_ANCHOR=8000 nests the subsets of the data curve (R22) around those 8,000 records.
stage_subset_prep() {
  rt_only_small || return 1
  local src="runs/$setting/rectune_choices_${RT_SUBSET_FROM:?RT_SUBSET_FROM names the variant whose records are sampled}"
  [ -s "$src/export_info.json" ] || { say "no records in $src"; return 1; }
  run $PY scripts/subset_records.py "$src" "$RT_DIR" "${RT_SUBSET:?RT_SUBSET is the number of records}" 2026 ${RT_SUBSET_ANCHOR:+--anchor $RT_SUBSET_ANCHOR}
}

# Decision4Rec v3 pilot (docs/08_architecture_v3.md): Kev-Rec-E reads a store's behavioural evidence as statements.
# Trained on Video Games, Musical Instruments and Steam; Industrial and Scientific is never seen, with its logs (R1)
# and as a new store with a catalogue only (R0). Run on request: ev_export, ev_train_08, ev_eval_08.
# EVV picks the pilot version: ev (first pilot: quota mix of generators, regime drawn per user) or ev2 (one fused
# score of the generators, regimes balanced over kept records). Each version has its own configs, records, model
# and report, so EVV=ev2 never touches the first pilot's files.
EVV="${EVV:-ev}"
if [ "$EVV" = ev ]; then EV_DIR="runs/$setting/evidence_choices"; else EV_DIR="runs/$setting/evidence_choices_$EVV"; fi
EV_PAIRS="vg:$EVV mi:$EVV steam:$EVV ind:$EVV ind:ev0"

stage_ev_export() {   # records from the stores of the three source domains, five regimes of purchase data
  rt_only_small || return 1
  server_stop
  run $JG export-evidence --config "$(cfg vg "$EVV")" "$(cfg mi "$EVV")" "$(cfg steam "$EVV")" --out "$EV_DIR" \
    --n-per-dataset 15000 --n-heldout 500
}

ev_train() {   # size, base model, base revision, learning rate, weights dtype, batch, accumulation
  local size="$1" base="$2" rev="$3" lr="$4" wdt="$5" batch="$6" accum="$7" f
  rt_only_small || return 1
  [ "${DRY_RUN:-0}" = 1 ] || [ -s "$EV_DIR/export_info.json" ] || stage_ev_export   # written last by the export
  server_stop
  run rm -rf "$KEV_DIR/runs/kev-$size-$EVV" "$KEV_DIR/runs/kev-$size-$EVV-eval" runs/"$setting"/*/*_kev-"$size"-"$EVV" \
    runs/"$setting"/*/cache/kev-"$size"-"$EVV".sqlite
  # Rows hold 30 options with their statements: --max_state 1152 raises the row limit to 1,792 tokens.
  in_kev uv run python -m kev.train --data "$ROOT/$EV_DIR/train.jsonl" --base "$base" --base_revision "$rev" \
    --init_from "jaredpalmer/kev-$size" --epochs 1 --lr "$lr" --batch "$batch" --accum "$accum" --length_sort 1 \
    --dtype bf16 --weights_dtype "$wdt" --checkpointing 1 --max_state 1152 --p_none 0 --p_none_distract 0 \
    --p_distract 0 --device "$DEVICE" --max_steps "${EV_STEPS:-0}" --out "runs/kev-$size-$EVV"
  # No kev.benchmark on the held-out records: it encodes within the training default of 1,024 tokens per row and
  # rejected about 95% of them; the evaluation runs (ev_eval) are the measure.
  run mkdir -p "$EV_DIR/kev-$size-$EVV"
  for f in training_config.json training_metrics.json; do run cp "$KEV_DIR/runs/kev-$size-$EVV/$f" "$EV_DIR/kev-$size-$EVV/" || true; done
}

stage_ev_train_08() { ev_train 0.8b Qwen/Qwen3.5-0.8B-Base dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68 1e-4 fp32 8 1; }
stage_ev_train_4() { ev_train 4b Qwen/Qwen3.5-4B-Base 1001bb4d826a52d1f399e183466143f4da7b741b 5e-5 bf16 2 4; }

ev_eval() {   # size: Kev-Rec-E and the released Kev-4B (same statements, no training) on every pilot config
  local size="$1" p d k runs=""
  rt_only_small || return 1
  server_start "runs/kev-$size-$EVV"
  for p in $EV_PAIRS; do
    d="${p%%:*}"; k="${p##*:}"
    run $J run --config "$(cfg "$d" "$k")" --backend "$B/kev-$size-$EVV.yaml"
    runs="$runs runs/$setting/$d/${d}_${k}_kev-$size-$EVV"
  done
  server_start jaredpalmer/kev-4b    # counts in the prompt without training, as in STAR (cached after the first time)
  for p in $EV_PAIRS; do
    d="${p%%:*}"; k="${p##*:}"
    run $J run --config "$(cfg "$d" "$k")" --backend "$B/kev-4b.yaml" --arms retriever,ev
    runs="$runs runs/$setting/$d/${d}_${k}_kev-4b"
  done
  server_stop
  for p in $EV_PAIRS; do run $JG baselines --config "$(cfg "${p%%:*}" "${p##*:}")"; done
  run $J asc --runs $runs --arms ev --only dm,fuse --head 100 --out "runs/$setting/${EVV}_eval_$size.md"
}

stage_ev_eval_08() { ev_eval 0.8b; }
stage_ev_eval_4() { ev_eval 4b; }

# The released Kev-0.8B on the same requests as every trained model (2026-10-12): with the released Kev-4B and Jev it
# completes the released-versus-trained grid of the Jev section. Same configs and users; three reports.
stage_released08() {
  rt_only_small || return 1
  local d p k runs_rtm="" runs_rte="" runs_ev=""
  server_start jaredpalmer/kev-0.8b
  for d in vg mi steam ind; do
    run $J run --config "$(cfg "$d" rtm)" --backend "$B/kev-0.8b.yaml" --arms retriever,rtm; runs_rtm="$runs_rtm runs/$setting/$d/${d}_rtm_kev-0.8b"
    run $J run --config "$(cfg "$d" rte)" --backend "$B/kev-0.8b.yaml" --arms retriever,rte; runs_rte="$runs_rte runs/$setting/$d/${d}_rte_kev-0.8b"
  done
  for p in vg:ev2 mi:ev2 steam:ev2 ind:ev2 ind:ev0; do
    d="${p%%:*}"; k="${p##*:}"
    run $J run --config "$(cfg "$d" "$k")" --backend "$B/kev-0.8b.yaml" --arms retriever,ev; runs_ev="$runs_ev runs/$setting/$d/${d}_${k}_kev-0.8b"
  done
  server_stop
  run $J asc --runs $runs_rtm --arms rtm --only dm,fuse --head 100 --out "runs/$setting/released08_rtm.md"
  run $J asc --runs $runs_rte --arms rte --only dm,fuse --head 100 --out "runs/$setting/released08_rte.md"
  run $J asc --runs $runs_ev --arms ev --only dm,fuse --head 100 --out "runs/$setting/released08_ev2.md"
}

# The released Kev-0.8B on MovieLens-1M, plain and enriched (needs the cases of ml1m_eval_08 and ml1m_plus_eval_08).
stage_released08_ml1m() {
  rt_only_small || return 1
  local m c runs_rtm="" runs_ev=""
  server_start jaredpalmer/kev-0.8b
  for m in ml1m ml1m_plus; do
    run $J run --config "$(cfg $m rtm)" --backend "$B/kev-0.8b.yaml" --arms retriever,rtm; runs_rtm="$runs_rtm runs/$setting/$m/${m}_rtm_kev-0.8b"
    for c in ev2 ev0; do run $J run --config "$(cfg $m $c)" --backend "$B/kev-0.8b.yaml" --arms retriever,ev; runs_ev="$runs_ev runs/$setting/$m/${m}_${c}_kev-0.8b"; done
  done
  server_stop
  run $J asc --runs $runs_rtm --arms rtm --only dm,fuse --head 100 --out "runs/$setting/released08_ml1m_rtm.md"
  run $J asc --runs $runs_ev --arms ev --only dm,fuse --head 100 --out "runs/$setting/released08_ml1m_ev2.md"
}

# MovieLens-1M as a second domain never used to train a decision model (2026-10-11): the trained 0.8B evidence reader
# (version 2) with logs and with a catalogue only, the full architecture with ML-1M's own SASRecF as one agent, and the
# released Kev-4B on the same inputs; then the baselines and the two reports.
ml1m_eval() {   # dataset (ml1m or ml1m_plus): the trained Kev-0.8B models and the released Kev-4B, then two reports
  rt_only_small || return 1
  local m="$1" c runs_ev="" runs_rtm=""
  server_start runs/kev-0.8b-ev2
  for c in ev2 ev0; do run $J run --config "$(cfg $m $c)" --backend "$B/kev-0.8b-ev2.yaml"; runs_ev="$runs_ev runs/$setting/$m/${m}_${c}_kev-0.8b-ev2"; done
  server_start runs/kev-0.8b-rtm
  run $J run --config "$(cfg $m rtm)" --backend "$B/kev-0.8b-rtm.yaml"; runs_rtm="runs/$setting/$m/${m}_rtm_kev-0.8b-rtm"
  server_start jaredpalmer/kev-4b
  for c in ev2 ev0; do run $J run --config "$(cfg $m $c)" --backend "$B/kev-4b.yaml" --arms retriever,ev; runs_ev="$runs_ev runs/$setting/$m/${m}_${c}_kev-4b"; done
  run $J run --config "$(cfg $m rtm)" --backend "$B/kev-4b.yaml" --arms retriever,rtm; runs_rtm="$runs_rtm runs/$setting/$m/${m}_rtm_kev-4b"
  server_stop
  for c in ev2 ev0 rtm; do run $JG baselines --config "$(cfg $m $c)"; done
  run $J asc --runs $runs_ev --arms ev --only dm,fuse --head 100 --out "runs/$setting/${m}_ev_eval_0.8b.md"
  run $J asc --runs $runs_rtm --arms rtm --only dm,fuse,gate,shrink --head 100 --out "runs/$setting/${m}_rtm_eval_0.8b.md"
}

stage_ml1m_eval_08() { ml1m_eval ml1m; }
# ML-1M with plots (Spillo et al.) and Wikidata properties; its baselines (SASRecF included) are trained first,
# on this machine or elsewhere, into runs/small/ml1m_plus.
stage_ml1m_plus_eval_08() { ml1m_eval ml1m_plus; }

# Hosted Jev (paid API, $0.042 per million input tokens) on the same users and states as the trained models.
# Jev cannot be trained by us (no fine-tuning, and its terms forbid training a model on its outputs), so it runs as
# released. The key is read from TYPESAFE_API_KEY or from ~/.typesafe_key (written by the user, mode 600); JEV_MAX_MTOK
# caps the input tokens of each run. Run after ml1m_eval_08, which trains the ML-1M pool.
jev_key() {   # TypeSafe key (~/.typesafe_key) or OpenRouter key (~/.openrouter_key); the first one found is used
  if [ -n "${TYPESAFE_API_KEY:-}" ] || [ -s "$HOME/.typesafe_key" ]; then
    [ -n "${TYPESAFE_API_KEY:-}" ] || export TYPESAFE_API_KEY="$(tr -d ' \n' < "$HOME/.typesafe_key")"
    JEV_CFG=jev; JEV_TAG=jev-1.13.0
  elif [ -n "${OPENROUTER_API_KEY:-}" ] || [ -s "$HOME/.openrouter_key" ]; then
    [ -n "${OPENROUTER_API_KEY:-}" ] || export OPENROUTER_API_KEY="$(tr -d ' \n' < "$HOME/.openrouter_key")"
    JEV_CFG=jev-openrouter; JEV_TAG=jev-1.13-openrouter
  else
    say "no Jev key (~/.typesafe_key or ~/.openrouter_key), hosted Jev is skipped"; return 1
  fi
  export JEV_MAX_MTOK="${JEV_MAX_MTOK:-20}"
}
stage_jev_pilot() {   # 50 validation and 50 test users of Video Games: real token counts before the full run
  rt_only_small || return 1; jev_key || return 1
  JEV_MAX_MTOK=1 run $J run --config "$(cfg vg rtm)" --backend "$B/$JEV_CFG.yaml" --arms retriever,rtm --max-cases 50 --no-report
}

jev_runs() {   # domains: hosted Jev on the full architecture (rtm) and on the evidence reader without a trained recommender
  local d k
  for d in "$@"; do
    run $J run --config "$(cfg "$d" rtm)" --backend "$B/$JEV_CFG.yaml" --arms retriever,rtm
    for k in ev2 ev0; do
      { [ "$k" = ev0 ] && [ "$d" != ind ] && [ "${d#ml1m}" = "$d" ]; } && continue   # catalogue-only scenarios: ind, ml1m
      run $J run --config "$(cfg "$d" "$k")" --backend "$B/$JEV_CFG.yaml" --arms retriever,ev
    done
  done
}

jev_reports() {   # every Jev run that exists, next to the trained Kev-0.8B and the released Kev-4B on the same users
  local d k r m runs_rtm="" runs_ev=""
  for d in vg mi steam ind ml1m ml1m_plus; do
    for m in "$JEV_TAG" kev-0.8b-rtm kev-4b; do
      r="runs/$setting/$d/${d}_rtm_$m"; [ -s "$r/ranks.json" ] && runs_rtm="$runs_rtm $r"
    done
    for k in ev2 ev0; do for m in "$JEV_TAG" kev-0.8b-ev2 kev-4b; do
      r="runs/$setting/$d/${d}_${k}_$m"; [ -s "$r/ranks.json" ] && runs_ev="$runs_ev $r"
    done; done
  done
  run $J asc --runs $runs_rtm --arms rtm --only dm,fuse,gate,shrink --head 100 --out "runs/$setting/jev_rtm_eval.md"
  run $J asc --runs $runs_ev --arms ev --only dm,fuse --head 100 --out "runs/$setting/jev_ev_eval.md"
}

stage_jev_eval() {    # the four domains of the main study, then a second pass on 300 users for repeatability
  rt_only_small || return 1; jev_key || return 1
  jev_runs vg mi steam ind
  run $J run --config "$(cfg vg rtm)" --backend "$B/$JEV_CFG-repeat.yaml" --arms retriever,rtm --max-cases 300 --no-report
  run $PY scripts/jev_repeat.py "runs/$setting/vg/vg_rtm_$JEV_TAG" "runs/$setting/vg/vg_rtm_$JEV_TAG-repeat" rtm \
    > "runs/$setting/jev_repeat.txt"
  jev_reports
}

stage_jev_ml1m() {    # after ml1m_eval_08, which trains the ML-1M pool and runs the Kev models there
  rt_only_small || return 1; jev_key || return 1
  jev_runs ml1m
  jev_reports
}

stage_jev_ml1m_plus() {   # after ml1m_plus_eval_08
  rt_only_small || return 1; jev_key || return 1
  jev_runs ml1m_plus
  jev_reports
}

# Perspective agents as questions to the decision model (scripts/perspective_questions_probe.py): five more choice
# questions in the request of the full architecture, development validation users, cross-validated.
stage_persp_q() {
  rt_only_small || return 1
  local m d
  for m in kev-0.8b-rtm kev-4b; do
    if [ "$m" = kev-4b ]; then server_start jaredpalmer/kev-4b; else server_start "runs/$m"; fi
    for d in vg mi steam ind; do
      run env CUDA_VISIBLE_DEVICES= $PY scripts/perspective_questions_probe.py "$(cfg "$d" rtm)" "$B/$m.yaml" 1000 \
        > "logs/probe/perspq_${d}_$m.log" 2>&1 || say "perspective questions failed on $d with $m"
    done
  done
  server_stop
}

# The trained decision model of the full architecture on the pool with the perspective agents of the knowledge graph
# and the personal-mode ranker (configs *_rtmpr.yaml, perspective agents v4 + v5a), against the same model on the pool.
stage_persp_dm() {
  rt_only_small || return 1
  local d runs=""
  server_start runs/kev-0.8b-rtm
  for d in vg mi steam ind; do
    run $J run --config "$(cfg "$d" rtmpr)" --backend "$B/kev-0.8b-rtm.yaml"
    runs="$runs runs/$setting/$d/${d}_rtmpr_kev-0.8b-rtm runs/$setting/$d/${d}_rtm_kev-0.8b-rtm"
  done
  server_stop
  for d in vg mi steam ind; do run $JG baselines --config "$(cfg "$d" rtmpr)"; done
  run $J asc --runs $runs --arms rtm --only dm,fuse,gate,shrink --head 100 --out "runs/$setting/persp_dm_eval_0.8b.md"
}

# Explanations by removing one family of statements (Roadmap R9), trained 0.8B and 4B decision models, 500 test users.
stage_explain() {
  rt_only_small || return 1
  local m d
  for m in kev-0.8b-rtm kev-4b-rtm; do
    server_start "runs/$m"
    for d in vg mi steam ind; do
      run env CUDA_VISIBLE_DEVICES= $PY scripts/explain_probe.py "$(cfg "$d" rtm)" "$B/$m.yaml" 500 \
        > "logs/probe/explain_${d}_$m.log" 2>&1 || say "explanations failed on $d with $m"
    done
  done
  server_stop
}

# The 4B decision model of the full architecture on MovieLens-1M (plain and enriched); runs after ml1m_eval_08 and
# ml1m_plus_eval_08, which build the pools and cases there.
# Ladder versions prepared on 2026-10-13 (records built on .13 and .34, docs/09_open_items.md):
#   perspectives v5   RT_MODEL=rtmv5: the pool reader also learns "which kind of choice will the user make next?"
#                     (labels from the rules of the knowledge graph); stage persp_v5 reads its answers (why_probe).
#   explanations v2   RT_MODEL=rtmfd: "found by" left out of 25% of the records; stage explain_fd repeats R9.
#   dense source      RT_MODEL=rtmml: plus cross-fitted MovieLens-1M records (x3); stage ml1m_rtmml evaluates ML-1M.
#   fusion-aware      RT_MODEL=rtmres RT_OFFSETS=0.6: the same records as rtm plus the ranker's score of every option,
#                     added to the logits in the loss, so the model learns what the ranker misses (pool reader v3).
stage_persp_v5() {
  rt_only_small || return 1
  local d
  server_start runs/kev-0.8b-rtmv5
  for d in vg mi steam ind; do
    run env CUDA_VISIBLE_DEVICES= $PY scripts/why_probe.py "$(cfg "$d" rtm)" "$B/kev-0.8b-rtmv5.yaml" 1000 \
      > "logs/probe/why_${d}.log" 2>&1 || say "why probe failed on $d"
  done
  server_stop
}

stage_explain_fd() {
  rt_only_small || return 1
  local d
  server_start runs/kev-0.8b-rtmfd
  for d in vg mi steam ind; do
    run env CUDA_VISIBLE_DEVICES= $PY scripts/explain_probe.py "$(cfg "$d" rtm)" "$B/kev-0.8b-rtmfd.yaml" 500 \
      > "logs/probe/explain_${d}_kev-0.8b-rtmfd.log" 2>&1 || say "explanations failed on $d with rtmfd"
  done
  server_stop
}

stage_explain_rt() {   # the explanation test of Appendix app:explain for the 0.8B pool reader named by RT_MODEL
  rt_only_small || return 1
  local d
  server_start "runs/kev-0.8b-$RT_MODEL"
  for d in vg mi steam ind; do
    run env CUDA_VISIBLE_DEVICES= $PY scripts/explain_probe.py "$(cfg "$d" rtm)" "$B/kev-0.8b-$RT_MODEL.yaml" 500 \
      > "logs/probe/explain_${d}_kev-0.8b-$RT_MODEL.log" 2>&1 || say "explanations failed on $d with $RT_MODEL"
  done
  server_stop
}

stage_rtmml_prep() {
  rt_only_small || return 1
  [ "${DRY_RUN:-0}" = 1 ] || [ -s "runs/$setting/rectune_choices_ml1m_cf/train.jsonl" ] || { say "no MovieLens-1M records yet"; return 1; }
  run env CUDA_VISIBLE_DEVICES= $PY scripts/merge_records.py "runs/$setting/rectune_choices_rtmml" \
    "runs/$setting/rectune_choices_rtm" "runs/$setting/rectune_choices_ml1m_cf:3"
}

stage_ml1m_rtmml() {
  rt_only_small || return 1
  local m runs=""
  server_start runs/kev-0.8b-rtmml
  for m in ml1m ml1m_plus; do
    run $J run --config "$(cfg $m rtm)" --backend "$B/kev-0.8b-rtmml.yaml"
    runs="$runs runs/$setting/$m/${m}_rtm_kev-0.8b-rtmml runs/$setting/$m/${m}_rtm_kev-0.8b-rtm"
  done
  server_stop
  run $J asc --runs $runs --arms rtm --only dm,fuse --head 100 --out "runs/$setting/ml1m_rtmml_eval_0.8b.md"
}

stage_ml1m_plus_rt() {   # the 0.8B pool reader named by RT_MODEL (configs of RTV) on enriched MovieLens-1M, against rtm
  rt_only_small || return 1
  server_start "runs/kev-0.8b-$RT_MODEL"
  run $J run --config "$(cfg ml1m_plus "$RTV")" --backend "$B/kev-0.8b-$RT_MODEL.yaml"
  server_stop
  run $J asc --runs "runs/$setting/ml1m_plus/ml1m_plus_${RTV}_kev-0.8b-$RT_MODEL" "runs/$setting/ml1m_plus/ml1m_plus_rtm_kev-0.8b-rtm" \
    --arms "$RTV,rtm" --only dm,fuse --head 100 --out "runs/$setting/ml1m_plus_${RT_MODEL}_eval_0.8b.md"
}

stage_ml1m_rt() {   # as ml1m_plus_rt, on plain MovieLens-1M (titles and genres), against rtm
  rt_only_small || return 1
  server_start "runs/kev-0.8b-$RT_MODEL"
  run $J run --config "$(cfg ml1m "$RTV")" --backend "$B/kev-0.8b-$RT_MODEL.yaml"
  server_stop
  run $J asc --runs "runs/$setting/ml1m/ml1m_${RTV}_kev-0.8b-$RT_MODEL" "runs/$setting/ml1m/ml1m_rtm_kev-0.8b-rtm" \
    --arms "$RTV,rtm" --only dm,fuse --head 100 --out "runs/$setting/ml1m_${RT_MODEL}_eval_0.8b.md"
}

# Contamination controls (2026-10-16, review R2): the trained pool reader of a size answers the requests of Decision4Rec-Full
# with an empty history (history-free prior) and with item titles removed (title-masked); configs <domain>_rtmctl.yaml,
# whose normal arm is read from the cache. CTL_DOMAINS limits the domains. Compare with logs/probe/arm_pairs.py
# rtmctl:rtm kev-<size>-rtm rtmctl:rtm_masked kev-<size>-rtm.
ctl_eval() {
  local size="$1" d
  rt_only_small || return 1
  server_start "runs/kev-$size-rtm"
  for d in ${CTL_DOMAINS:-vg mi steam ind ml1m ml1m_plus}; do
    run $J run --config "$(cfg "$d" rtmctl)" --backend "$B/kev-$size-rtm.yaml" --no-report
  done
  server_stop
}
stage_ctl_08() { ctl_eval 0.8b; }
stage_ctl_4() { ctl_eval 4b; }

stage_ml1m_plus_rt_4() {   # as ml1m_plus_rt, for the 4B pool reader named by RT_MODEL (R19 at 4B on the Spark), against rtm 4B
  rt_only_small || return 1
  server_start "runs/kev-4b-$RT_MODEL"
  run $J run --config "$(cfg ml1m_plus "$RTV")" --backend "$B/kev-4b-$RT_MODEL.yaml"
  server_stop
  run $J asc --runs "runs/$setting/ml1m_plus/ml1m_plus_${RTV}_kev-4b-$RT_MODEL" "runs/$setting/ml1m_plus/ml1m_plus_rtm_kev-4b-rtm" \
    --arms "$RTV,rtm" --only dm,fuse --head 100 --out "runs/$setting/ml1m_plus_${RT_MODEL}_eval_4b.md"
}

# R3a (2026-10-15, the Spark): the released Kev-27B reads the requests of the full architecture (rtm) in the four domains
# and on both MovieLens-1M versions, as the released 0.8B and 4B did; compared with the trained 0.8B and 4B pool readers.
stage_released27() {
  rt_only_small || return 1
  local d runs=""
  server_start jaredpalmer/kev-27b
  for d in vg mi steam ind; do
    run $J run --config "$(cfg "$d" rtm)" --backend "$B/kev-27b.yaml" --arms retriever,rtm $RUN_LIMIT
    runs="$runs runs/$setting/$d/${d}_rtm_kev-27b runs/$setting/$d/${d}_rtm_kev-0.8b-rtm runs/$setting/$d/${d}_rtm_kev-4b-rtm"
  done
  for d in ml1m ml1m_plus; do run $J run --config "$(cfg "$d" rtm)" --backend "$B/kev-27b.yaml" --arms retriever,rtm $RUN_LIMIT; done
  server_stop
  [ -n "${RUN_MAX:-}" ] && { say "partial runs (RUN_MAX=$RUN_MAX): compare them with logs/probe/arm_pairs.py on the common users"; return 0; }
  run $J asc --runs $runs --arms rtm --only dm,fuse --head 100 --out "runs/$setting/released27_rtm.md"
  run $J asc --runs runs/$setting/ml1m/ml1m_rtm_kev-27b runs/$setting/ml1m/ml1m_rtm_kev-4b-rtm \
    runs/$setting/ml1m_plus/ml1m_plus_rtm_kev-27b runs/$setting/ml1m_plus/ml1m_plus_rtm_kev-4b-rtm \
    --arms rtm --only dm,fuse --head 100 --out "runs/$setting/released27_ml1m_rtm.md"
}

stage_ml1m_eval_4() {
  rt_only_small || return 1
  local m runs=""
  server_start runs/kev-4b-rtm
  for m in ml1m ml1m_plus; do
    run $J run --config "$(cfg $m rtm)" --backend "$B/kev-4b-rtm.yaml"
    runs="$runs runs/$setting/$m/${m}_rtm_kev-4b-rtm runs/$setting/$m/${m}_rtm_kev-0.8b-rtm"
  done
  server_stop
  run $J asc --runs $runs --arms rtm --only dm,fuse,gate,shrink --head 100 --out "runs/$setting/ml1m_rtm_eval_4b.md"
}

stage_collect() {
  server_stop
  local out="decision4rec_${setting}_$(date +%Y%m%d_%H%M).tgz"
  [ "${DRY_RUN:-0}" = 1 ] && { echo "+ archive of reports and logs -> $out"; return 0; }
  { find runs -type f \( -name "report.md" -o -name "report.json" -o -name "benchmark.md" -o -name "benchmark.json" \
      -o -name "benchmark_summary_*.md" -o -name "ranks.json" -o -name "audit.json" -o -name "agent_study.json" \
      -o -name "rating.md" -o -name "rating.json" -o -name "retrieval_study.md" -o -name "retrieval_study.json" \
      -o -name "agent_bench.json" -o -name "agent_bench.md" -o -name "agent.test.jsonl" -o -name "manifest.json" -o -name "*_info.json" -o -name "info.json" -o -name "sasrec.json" \
      -o -name "confirm.md" -o -name "confirm.json" -o -name "asc.md" -o -name "asc.json" -o -name "stratify.md" -o -name "stratify.json" \
      -o -name "e[123]_*.md" -o -name "e[123]_*.json" -o -name "rt*_eval_*.md" -o -name "rt*_eval_*.json" \
      -o -name "ev*_eval_*.md" -o -name "ev*_eval_*.json" -o -name "jev_*_eval.md" -o -name "jev_repeat.txt" -o -name "ml1m_*_eval_*.md" -o -name "ml1m_*_eval_*.json" -o -path "*/evidence_choices*/kev-*" \
      -o -name "export_info.json" -o -path "*/rectune_choices*/kev-*" \) -print0
    find logs data/processed -maxdepth 3 -type f \( -name "*.log" -o -name "meta.json" -o -name "status.txt" \) -print0
  } | tar czf "$out" --null -T -
  say "results archive: $ROOT/$out"
}

core_small="setup data retriever baselines kev check system ablation trust appendix"
# The stages below are part of the default run. They are non-blocking: if one fails, the failure is logged
# and the run goes on, so a new stage or a model that does not fit cannot lose the core results.
# `agent` comes first: it gives the primary row of the benchmark and reuses the server of the core stages.
optional_small="agent rectune model_small model_mid logit jev"
core_large="setup data retriever baselines kev check system ablation"
optional_large="agent model_large"

do_stage() {   # name, core|optional. Each stage runs in a child shell so that any failing command aborts it.
  local name="$1" kind="$2" marker="$LOGS/$1.done"
  if skipped "$name"; then say "stage $name skipped on request"; return 0; fi
  if [ -f "$marker" ] && [ "${FORCE:-0}" != 1 ] && [ "${DRY_RUN:-0}" != 1 ]; then say "stage $name already done"; return 0; fi
  say "stage $name started"
  local status=0
  if [ "${DRY_RUN:-0}" = 1 ]; then
    bash "$0" "$setting" "__$name" || status=$?
  else
    set +e
    bash "$0" "$setting" "__$name" 2>&1 | tee -a "$LOGS/$name.log"
    status=${PIPESTATUS[0]}
    set -e
  fi
  if [ "$status" = 0 ]; then
    [ "${DRY_RUN:-0}" = 1 ] || { touch "$marker"; echo "$name ok" >> "$LOGS/status.txt"; }
    say "stage $name finished"
  else
    [ "${DRY_RUN:-0}" = 1 ] || echo "$name FAILED" >> "$LOGS/status.txt"
    if [ "$kind" = core ]; then
      say "stage $name FAILED, see $LOGS/$name.log. Fix it or send me that log, then run the same command again."
      server_stop
      exit 1
    fi
    say "optional stage $name failed, see $LOGS/$name.log. Continuing."
  fi
}

mkdir -p "$LOGS"

if [ "${stage#__}" != "$stage" ]; then          # internal: run one stage body in this shell, fail fast
  "stage_${stage#__}"
  exit 0
fi

# One run per setting at a time: a second copy would share the server, the cache and the result files.
LOCK="$LOGS/run.lock"
if [ "${DRY_RUN:-0}" != 1 ]; then
  if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
    echo "another $setting run is active (process $(cat "$LOCK")). Stop it first, see RUNBOOK.md."; exit 1
  fi
  echo $$ > "$LOCK"
fi
# Stopping the run stops the server too, and then really ends the run.
trap 'server_stop; rm -f "$LOCK"; exit 130' INT TERM
trap '[ "$(cat "$LOCK" 2>/dev/null)" = "$$" ] && rm -f "$LOCK"' EXIT

if [ "$stage" = all ]; then
  eval "core=\$core_$setting; optional=\$optional_$setting"
  for s in $core; do do_stage "$s" core; done
  for s in $optional; do do_stage "$s" optional; done
  say "all stages finished: $([ -f "$LOGS/status.txt" ] && tr '\n' ' ' < "$LOGS/status.txt" || true)"
  stage_collect
else
  name="${stage//-/_}"
  declare -F "stage_$name" >/dev/null || { echo "unknown stage $stage"; exit 1; }
  if [ "$name" = collect ]; then stage_collect; else FORCE=1 do_stage "$name" core; server_stop; fi
fi
