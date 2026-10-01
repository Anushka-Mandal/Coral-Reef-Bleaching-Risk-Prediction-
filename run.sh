#!/usr/bin/env bash
# Start CoralWatch: website + live NOAA data + 7-day forecast + AI briefings.
#   ./run.sh            then open http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  echo "Setting up the Python environment (first run only)..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi

if ! curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama isn't running - start it for free AI briefings:  ollama serve   (once: ollama pull qwen2.5:3b)"
fi

[ -f backend/data/corpus.json ] || .venv/bin/python -m backend.ai.build_corpus
[ -f forecast/6_saved_models/v2/meta.json ] || echo "No trained forecast model yet - see forecast/README.md"

exec .venv/bin/python -m backend.app
