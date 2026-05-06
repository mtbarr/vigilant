#!/usr/bin/env bash
set -euo pipefail

echo "=== Verificando k6 ==="
if ! command -v k6 &> /dev/null; then
    echo "  baixando k6..."
    curl -sL https://github.com/grafana/k6/releases/latest/download/k6-linux-amd64.tar.gz \
        | tar xz -C /tmp
    mv /tmp/k6-v*linux-amd64/k6 /usr/local/bin/k6 2>/dev/null || \
    mv /tmp/k6 /usr/local/bin/k6 2>/dev/null || true
    rm -rf /tmp/k6-*linux-amd64 2>/dev/null || true
fi
echo "  k6 $(k6 version 2>&1 | head -1)"

echo "=== Baixando smoke test oficial da Rinha ==="
curl -sL https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/test/smoke.js \
    -o /tmp/rinha-smoke.js

echo "=== Aguardando API na porta 9999 ==="
for attempt in $(seq 1 30); do
    if curl -sf http://localhost:9999/ready > /dev/null 2>&1; then
        echo "  API pronta"
        break
    fi
    echo "  tentativa $attempt/30..."
    sleep 2
done

echo "=== Executando smoke test ==="
K6_NO_USAGE_REPORT=true k6 run /tmp/rinha-smoke.js

echo "=== Smoke test concluido ==="
