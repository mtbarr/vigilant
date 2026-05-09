# Vigilant — Rinha de Backend 2026

Implementação em Java para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026)

## O Problema

Duas instâncias com 0.45 CPU e 160 MB de RAM cada, atrás de HAProxy com 0.10 CPU e 30 MB. Cada request precisa encontrar os 5 vizinhos mais próximos em ~72.000
vetores de 14 dimensões e classificar como fraude ou não — em menos de 1ms de CPU por requisição.

## Como Funciona

```
POST /fraud-score
  │
  ▼ Parser JSON → float[14]           (~1μs, indexOf intrínseco, zero regex)
  │
  ▼ Distância a 512 centroides         (~3μs, 14 Math.fma desenrolados)
  ▼ Quickselect → top 12 clusters      (~2μs)
  ▼ ADC table float[3584]              (~2μs, (query[m] - codebook[m][c])²)
  ▼ Scan 12 clusters (~1680 vetores)   (~50μs, 14 lookups/vetor na ADC table)
  ├─ Early exit após 8 clusters se ≤1 ou ≥4 fraudes nos top-5
  │
  ▼ Reranking exato dos 10 candidatos  (~2μs, L2 real via mmap)
  ▼ Votação nos 5 mais próximos
  │
  ▼ String pré-computada → response    (zero alocação)
```

### As três camadas

| Camada        | O que faz                                                  | Por que importa                                                         |
|---------------|------------------------------------------------------------|-------------------------------------------------------------------------|
| **IVF**       | K-Means com 512 centroides, probe só os 12 mais próximos   | Poda ~98% do dataset antes de qualquer distância                        |
| **PQ 1D**     | Cada vetor vira 14 bytes; distância = 14 lookups em tabela | Elimina leitura de floats, cada distância custa 14 lookups + 13 adições |
| **Reranking** | L2 exata nos 10 melhores candidatos via mmap               | Corrige erros de ordenamento da PQ, garante vizinhos corretos           |

PQ 1D (14 subespaços × 1 dimensão) em vez de PQ 2D (7×2): cada dimensão ganha 256 níveis independentes em vez de ~16, reduzindo drasticamente o erro de
quantização.

### Classificação

Voto majoritário entre os 5 vizinhos mais próximos:

| Votos fraude | `approved` | `fraud_score`   |
|--------------|------------|-----------------|
| 0–2          | `true`     | 0.0 / 0.2 / 0.4 |
| 3–5          | `false`    | 0.6 / 0.8 / 1.0 |

O índice `fraudVoteCount` seleciona diretamente uma string pré-computada do array `FRAUD_SCORE_RESPONSES` — zero formatação no hot path.

## Por que é Rápido

- **Zero alocação**: `ThreadLocal<float[]>` para query, centroides, ADC table, vizinhos. Resposta é `String[]` estática.
- **FMA desenrolado**: `Math.fma` com 14 termos → GraalVM AOT gera VFMADD (1 ciclo cada).
- **ADC table flat**: `float[3584]` em vez de `float[14][256]` — elimina uma indireção por lookup.
- **Offset por adição**: `codeOff += PQ_M` em vez de `codeOff = i * PQ_M`.
- **mmap + Panama**: vetores originais acessados via `MemorySegment.get(FLOAT_LE)` — zero cópia heap, só 10×56 bytes tocados no reranking.
- **Epsilon GC**: binário compilado com `--gc=epsilon`, que elimina a coleta de lixo. Como o hot path não aloca objetos (ThreadLocals, strings
  pré-computadas, buffers reutilizados), nunca há lixo para coletar. Sem pausas de GC, a latência fica determinística — em 0.45 CPU, uma única pausa de 10ms
  já afetaria o p99.
- **Warmup**: 500 buscas com vetor zero no `@PostConstruct` — força page faults e aquece cache de instruções.

## Arquitetura

```
cliente → HAProxy (9999, 0.10 CPU / 30 MB)
            ├─ round-robin + health check GET /ready
            ├──▶ api1 (Vert.x, 0.45 CPU / 160 MB)
            └──▶ api2 (Vert.x, 0.45 CPU / 160 MB)
```

- 4 event loops Vert.x para I/O, 32 workers para busca (`executeBlocking(false)`)
- HAProxy só roteia quando `GET /ready` retorna 200
- GraalVM CE 25 native-image com `-march=haswell` (AVX2 + FMA)

## Endpoints

| Método | Path           | Descrição                                           |
|--------|----------------|-----------------------------------------------------|
| `GET`  | `/ready`       | 200 se índice carregado, 503 caso contrário         |
| `POST` | `/fraud-score` | Avalia transação, retorna `{approved, fraud_score}` |

## Executar

```bash
docker compose up --build
```

Disponível em `http://localhost:9999`.

Build manual:

```bash
./scripts/build-index.sh     # gera data/index.bin
./scripts/build-app.sh       # docker build -f Dockerfile.app
docker compose up -d
```

## Estrutura

```
src/main/java/io/github/mtbarr/rinha/
├── api/VertxHttpServer.java           # HTTP server + roteamento
├── service/FraudRequestParser.java    # Parser JSON → float[14]
└── index/
    ├── InvertedFileIndex.java         # Runtime: carrega índice, busca k-NN
    └── OfflineIndexBuilder.java       # Offline: K-Means + PQ + escrita bin
```
