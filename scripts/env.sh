#!/usr/bin/env bash
# Creates the Python environment of Decision4Rec in .venv and installs every library the experiments need.
#
#   bash scripts/env.sh                      create the environment, or repair a broken one
#   REBUILD=1 bash scripts/env.sh            rebuild it from scratch
#   PYTHON=python3.12 bash scripts/env.sh    choose the interpreter (3.11 or newer)
#
# For manual work afterwards:  source .venv/bin/activate
# The run scripts call this file themselves and do not need the environment to be activated.
set -euo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME                  # packages of another Python must not leak into this one
py="${PYTHON:-python3}"
venv=.venv/bin/python

"$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' || {
  echo "Python 3.11 or newer is required, $py is $("$py" --version 2>&1). Set PYTHON, or use conda (see RUNBOOK.md)."; exit 1; }

[ -f decision4rec/data/loaders.py ] || {
  echo "the copy of the project is incomplete (decision4rec/data is missing): sync again with the command of RUNBOOK.md"; exit 1; }

# An environment copied from another machine, or built by another Python, holds compiled libraries that
# this interpreter cannot load (numpy fails first). It cannot be repaired in place, so it is rebuilt.
healthy() { [ -x "$venv" ] && "$venv" -c 'import numpy, pip' >/dev/null 2>&1; }
if [ "${REBUILD:-0}" = 1 ] || ! healthy; then
  echo "creating the virtual environment in .venv with $("$py" --version 2>&1)"
  "$py" -m venv --clear .venv || {
    echo "could not create the environment. On Debian or Ubuntu run: sudo apt install python3-venv"; exit 1; }
fi

"$venv" -m pip install -q -U pip
"$venv" -m pip install -q -e ".[train,logit,active,dev]"

"$venv" - <<'EOF'
import platform, shutil, sys
import numpy, yaml, tqdm, torch, transformers, activegraph, pytest, decision4rec
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"python {platform.python_version()}  numpy {numpy.__version__}  torch {torch.__version__}  "
      f"transformers {transformers.__version__}  activegraph {activegraph.__version__}  device {device}")
if shutil.which("nvidia-smi") and device != "cuda":
    sys.exit("an NVIDIA GPU is present but torch cannot use it: check the driver, then REBUILD=1 bash scripts/env.sh")
EOF
echo "environment ready. To use it by hand: source .venv/bin/activate"
