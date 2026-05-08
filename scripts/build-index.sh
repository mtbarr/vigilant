#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Building flat_index.bin (offline) ==="
./gradlew :compileJava --no-daemon

mkdir -p data
java -Xmx4g \
  --add-modules=jdk.incubator.vector \
  -cp "$(./gradlew -q :printClasspath --no-daemon):build/classes/java/main" \
  io.github.mtbarr.rinha.index.OfflineIndexBuilder \
  src/main/resources/references.json.gz \
  data/flat_index.bin

echo "=== flat_index.bin ready at data/ ==="
ls -lh data/flat_index.bin
