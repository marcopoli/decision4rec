#!/usr/bin/env bash
# No-GPU smoke test of the whole pipeline on synthetic data with the oracle backend.
set -euo pipefail
python -m decision4rec.cli prepare --dataset synthetic --out data/processed/synthetic --max-users 400
python -m decision4rec.cli check-backend --config configs/synthetic.yaml
python -m decision4rec.cli run --config configs/synthetic.yaml
python -m decision4rec.cli audit --config configs/synthetic_agent.yaml --n-cases 60
python -m decision4rec.cli agent --config configs/synthetic_agent.yaml --n-cases 40
python -m decision4rec.cli rating --config configs/synthetic_rating.yaml --n-cases 100
python -m decision4rec.cli retrieval --config configs/synthetic_retrieval.yaml --n-users 200 > /dev/null
python -m decision4rec.cli baselines --config configs/synthetic_main.yaml --device cpu
python -m decision4rec.cli run --config configs/synthetic_main.yaml --no-report
python -m decision4rec.cli agent-bench --config configs/synthetic_main.yaml > /dev/null
python -m decision4rec.cli benchmark --config configs/synthetic_main.yaml
echo "smoke test finished: see runs/synthetic"
