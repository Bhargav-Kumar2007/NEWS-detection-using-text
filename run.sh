#!/usr/bin/env bash
# Linux / macOS helper.  Usage: ./run.sh <command> [extra args passed to the Python script]
set -e
cd "$(dirname "$0")"

PY=python3
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; fi

cmd="${1:-help}"
shift || true

case "$cmd" in
  setup)
    python3 -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    # Optional GPU wheel index, e.g.: TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 ./run.sh setup
    if [ -n "$TORCH_INDEX_URL" ]; then .venv/bin/python -m pip install torch --index-url "$TORCH_INDEX_URL"; fi
    .venv/bin/python -m pip install -r requirements.txt
    ;;
  download)     "$PY" download_data.py "$@" ;;
  train-small)  "$PY" train.py --mode small "$@" ;;
  train-full)   "$PY" train.py --mode full "$@" ;;
  evaluate)     "$PY" evaluate.py "$@" ;;
  predict)      "$PY" predict.py "$@" ;;
  test)         "$PY" tests/test_smoke.py ;;
  *)
    echo "Usage: ./run.sh {setup|download|train-small|train-full|evaluate|predict|test} [args]"
    echo "Examples:"
    echo "  ./run.sh setup"
    echo "  ./run.sh download"
    echo "  ./run.sh train-small --model textcnn"
    echo "  ./run.sh evaluate --run-dir outputs/bilstm_small"
    echo "  ./run.sh predict --text \"Scientists discover a cat that can speak fluent French\""
    ;;
esac
