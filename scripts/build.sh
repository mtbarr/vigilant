#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Building Docker image (clean) ==="
docker compose build --no-cache

echo "=== Build complete ==="
docker images --filter "reference=${API_IMAGE:-vigilant}" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
