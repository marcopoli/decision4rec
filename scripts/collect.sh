#!/usr/bin/env bash
# Bundle every report (no raw data, no model outputs) into one archive to send back for analysis.
set -euo pipefail
out="decision4rec_results_$(date +%Y%m%d_%H%M).tgz"
find runs -type f \( -name "report.md" -o -name "report.json" -o -name "benchmark.md" -o -name "benchmark.json" \
  -o -name "benchmark_summary_*.md" -o -name "ranks.json" -o -name "audit.json" -o -name "agent_study.json" \
  -o -name "rating.md" -o -name "rating.json" -o -name "retrieval_study.md" -o -name "retrieval_study.json" \
  -o -name "agent_bench.json" -o -name "agent_bench.md" -o -name "agent.test.jsonl" -o -name "manifest.json" -o -name "*_info.json" -o -name "info.json" -o -name "sasrec.json" \) -print0 | tar czf "$out" --null -T -
echo "wrote $out"
