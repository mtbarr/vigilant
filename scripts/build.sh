#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# pre-requisitos
MISSING=false
for f in data/references.json.gz data/mcc_risk.json; do
    if [ ! -f "$f" ]; then
        echo "ERRO: $f nao encontrado. Coloque os arquivos da Rinha em ./data/"
        MISSING=true
    fi
done
$MISSING && exit 1

echo "=== Building Docker image ==="
docker compose build

echo "=== Build complete ==="
docker images vigilant --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
