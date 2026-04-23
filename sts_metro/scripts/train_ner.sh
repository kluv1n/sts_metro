#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -x "$(pwd)/.venv/bin/python" ]]; then
  PY="$(pwd)/.venv/bin/python"
elif [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
else
  PY="python3"
fi
"$PY" -m pip install -r requirements-ner.txt
PYTHONPATH=src "$PY" scripts/prepare_ner_splits.py
PYTHONPATH=src "$PY" scripts/train_sts_ner.py \
  --epochs 10 \
  --batch-size 8 \
  --lr 3e-5 \
  --model-name DeepPavlov/rubert-base-cased \
  --output-dir source_model_train/ner_model
PYTHONPATH=src "$PY" scripts/eval_sts_ner.py \
  --model-dir source_model_train/ner_model \
  --test-json source_model_train/splits/test.json
