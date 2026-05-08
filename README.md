# Rinha de Backend 2026 — Java (Quarkus + GraalVM)

Solução em Java para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026), foco em detecção de fraude em transações com busca k-NN
vetorial.

**Resultado final:** p99 **2.35ms** · 0% failure rate · **5419 pontos**

## Arquitetura

```
cliente
  │
  ▼
HAProxy (porta 9999)          0.10 CPU / 30 MB
  ├─ round-robin via TCP
  │
  ├──▶ instância 1 (Java)     0.45 CPU / 160 MB
  └──▶ instância 2 (Java)     0.45 CPU / 160 MB
```

## Pipeline

```
POST /fraud-score (Vert.x)
      │
      ▼
  parser JSON (String.indexOf intrínseco)
      │
      ▼
  vetorização → float[14]
      │
      ▼
  IVF — distância a 512 centroides (FMA unrolled)
      │
      ▼
  quickselect → top 12 clusters
      │
      ▼
  ADC table → float[3584] (distâncias query × codebooks)
      │
      ▼
  scan: 12 clusters × ~140 vec = ~1680 distâncias ADC
      │
      ▼
  early exit (após 8 clusters: ≤1 ou ≥4 fraudes nos top 5)
      │
      ▼
  reranking exato (L2 via mmap: top 10 → reordena → top 5)
      │
      ▼
  contagem de fraudes → resposta (String estática)
```

## Vetorização

| #  | Dimensão             | Fórmula                             |
|----|----------------------|-------------------------------------|
| 0  | `amount`             | `clamp(amount / 10_000)`            |
| 1  | `installments`       | `clamp(installments / 12)`          |
| 2  | `amount_vs_avg`      | `clamp((amount / avg_amount) / 10)` |
| 3  | `hour`               | `hour / 23`                         |
| 4  | `day_of_week`        | `(dow - 1) / 6` (dow 1..7)          |
| 5  | `minutes_since_last` | `clamp(min / 1440)` ou `-1`         |
| 6  | `km_from_last`       | `clamp(km / 1000)` ou `-1`          |
| 7  | `km_from_home`       | `clamp(km / 1000)`                  |
| 8  | `tx_24h`             | `clamp(count / 20)`                 |
| 9  | `is_online`          | `1` ou `0`                          |
| 10 | `card_present`       | `1` ou `0`                          |
| 11 | `unknown_merchant`   | `1` ou `0`                          |
| 12 | `mcc_risk`           | lookup table                        |
| 13 | `merchant_avg`       | `clamp(avg / 10_000)`               |

## Índice — IVF + PQ

### Build (offline, roda no Dockerfile stage 1)

`OfflineIndexBuilder` processa `references.json.gz` (~72k vetores):

1. **K-Means++** com K=512 sobre todos os vetores, 20 iterações Lloyd
2. **Product Quantization 1D**: 14 subespaços × 1 dimensão, 256 centroids cada, 20 iterações K-Means
3. Cada vetor comprimido de 14 floats (56 bytes) para **14 bytes**
4. Índice invertido: 512 clusters, IDs + PQ codes
5. Saída: `index.bin` (~11 MB)

### Runtime — InvertedFileIndex

1. Distância da query aos **512 centroides** — loop desenrolado com `Math.fma` (FMA)
2. Quickselect → **12 clusters** mais próximos
3. **ADC table** `float[3584]` — distâncias query × codebooks (14×256), pré-computada por query
4. Scan: ~1680 vetores nos 12 clusters, distância via lookup na ADC table
5. **Insertion sort** mantém top 10 candidatos
6. **Early exit** após 8 clusters: se top 5 têm ≤1 fraude ou ≥4 fraudes, retorna
7. **Reranking exato**: lê os 10 vetores originais via mmap (Panama `MemorySegment`), computa L2, reordena, vota nos top 5

## Otimizações

### Parse JSON

`String.indexOf()` — intrínseco JVM, compilado para AVX2 no native-image. Chaves pré-compiladas como `"\"key\""`, sem concatenação. Floats e timestamps
parseados manualmente (sem `substring()`/`parseFloat()`).

### ADC table achatada

`float[14][256]` → `float[3584]`. Uma indireção a menos nas 23.520 lookups/query.

### Arrays planos invertidos

`int[][]` + `byte[][]` → `int[]` + `byte[]` + `int[] clusterOffsets`. Sem indireção dupla, melhor localidade.

### Reranking via mmap

Vetores originais no `index.bin` mapeados com `MemorySegment` (Arena.global). Lê 10 × 56 = 560 bytes/query. Zero alocação heap.

### Centroide desenrolado

14 dimensões escritas explicitamente com FMA. Zero overhead de loop.

### Respostas pré-computadas

6 strings `String[]` indexadas por 0..5 fraudes.

### Warmup

3 buscas no `@PostConstruct` para page faults e cache CPU.

### Threading

Vert.x com 4 event loops, busca em `executeBlocking(false)` com 32 workers. ThreadLocals para buffers — zero alocação no hot path.

### Compilação

GraalVM CE 25 native-image com `-march=haswell` (AVX2+FMA). Sem garbage collector — `--gc=epsilon` no build. O parser aloca zero objetos por request, heap de
160MB contém só dados fixos.

## Stack

| Componente    | Tecnologia                        |
|---------------|-----------------------------------|
| HTTP          | Vert.x (custom, sem Quarkus REST) |
| DI            | ArC (Quarkus CDI)                 |
| Runtime       | GraalVM CE 25 native-image        |
| Memória       | `--gc=epsilon` (sem coleta)       |
| Busca         | IVF + PQ 1D + reranking L2        |
| Build índice  | K-Means++ + Lloyd                 |
| Decompressão  | GZIPInputStream                   |
| Load balancer | HAProxy 3.1                       |
| Container     | Docker / docker-compose           |
| Alvo CPU      | `-march=haswell` (AVX2, FMA)      |

## Executar

```bash
docker compose up --build
```

`http://localhost:9999`.

- `GET /ready` — health check
- `POST /fraud-score` — avalia transação

```bash
curl -s -X POST http://localhost:9999/fraud-score \
  -H "Content-Type: application/json" \
  -d '{
    "id": "abc123",
    "transaction": { "amount": 250.0, "installments": 1, "requested_at": "2025-01-15T14:30:00Z" },
    "customer": { "avg_amount": 200.0, "tx_count_24h": 3, "known_merchants": ["mcid_1"] },
    "merchant": { "id": "mcid_1", "mcc": "5411", "avg_amount": 180.0 },
    "terminal": { "is_online": false, "card_present": true, "km_from_home": 2.5 },
    "last_transaction": { "timestamp": "2025-01-15T10:00:00Z", "km_from_current": 1.2 }
  }'
```

```json
{
  "approved": true,
  "fraud_score": 0.0
}
```

## Build manual

```bash
./scripts/build-index.sh     # gera data/index.bin
./scripts/build-app.sh       # docker build -f Dockerfile.app
docker compose up -d
```
