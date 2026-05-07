#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "=== Building Docker image ==="
export API_IMAGE=vigilant:latest
docker compose build

echo "=== Build complete ==="
docker images vigilant --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
