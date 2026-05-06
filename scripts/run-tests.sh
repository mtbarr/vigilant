#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cleanup() {
    echo "=== Cleaning up ==="
    docker compose down --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Building and starting services ==="
docker compose up --build -d

echo "=== Waiting for API readiness ==="
for attempt in $(seq 1 30); do
    if curl -sf http://localhost:9999/ready > /dev/null 2>&1; then
        echo "API ready on port 9999"
        break
    fi
    echo "  attempt $attempt/30..."
    sleep 2
done

echo "=== Running smoke test ==="
if command -v k6 &> /dev/null; then
    k6 run .references/rinha-de-backend-2026-main/test/smoke.js
else
    echo "k6 not found, using curl..."
    curl -s -X POST http://localhost:9999/fraud-score \
        -H 'Content-Type: application/json' \
        -d '{
            "id": "tx-smoke-001",
            "transaction": {"amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z"},
            "customer": {"avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009","MERC-001","MERC-001"]},
            "merchant": {"id": "MERC-001", "mcc": "5912", "avg_amount": 298.95},
            "terminal": {"is_online": false, "card_present": true, "km_from_home": 13.7090520965},
            "last_transaction": {"timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.8626479774}
        }' | python3 -m json.tool 2>/dev/null || python -m json.tool 2>/dev/null || cat
fi

echo "=== Smoke test complete ==="
