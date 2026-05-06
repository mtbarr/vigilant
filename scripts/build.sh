#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Building Docker image ==="
docker compose build

echo "=== Build complete ==="
docker images vigilant --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
