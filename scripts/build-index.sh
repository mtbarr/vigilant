#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Building index.bin (offline) ==="
./gradlew :compileJava --no-daemon

mkdir -p data
java -Xmx4g \
  -cp "$(./gradlew -q :printClasspath --no-daemon):build/classes/java/main" \
  io.github.mtbarr.rinha.index.OfflineIndexBuilder \
  src/main/resources/references.json.gz \
  data/index.bin

echo "=== index.bin ready at data/index.bin ==="
ls -lh data/index.bin
