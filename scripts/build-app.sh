#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

INDEX="data/flat_index.bin"
if [ ! -f "$INDEX" ]; then
  echo "flat_index.bin nao encontrado. Execute primeiro: scripts/build-index.sh"
  exit 1
fi

echo "=== Building Docker image (app only, sem rebuild do index) ==="
docker build -f Dockerfile.app -t vigilant:latest .

echo "=== Build complete ==="
docker images vigilant --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
