# Fine-tuning Vector Databases for RAG Systems

A high-performance Retrieval-Augmented Generation (RAG) pipeline comparing PostgreSQL (pgvector) vs FAISS vector databases.

## Overview

- **Dataset**: ML-ArXiv-Papers (100K abstracts)
- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2 (384 dimensions)
- **Vector Stores**: pgvector (PostgreSQL) + FAISS
- **Index Types**: Flat, IVFFlat, HNSW
- **Benchmark Queries**: 50 queries × 3 runs

## Prerequisites

1. **Python 3.10+**
2. **PostgreSQL 15+** with pgvector extension

### Install PostgreSQL + pgvector

```bash
docker run -d \
  --name pgvector \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=rag_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

```sql
CREATE EXTENSION vector;
```

## Installation

```bash
cd capstone-phase2
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate
# Activate (Linux/Mac)
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Create `.env` file:

```env
DB_USER=postgres
DB_PASSWORD=secret
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_db
```

## Usage

### 1. Setup Databases

#### A. PostgreSQL + pgvector Setup
To populate PostgreSQL:
```bash
python -m src.setup_db
```

#### B. Milvus Setup
To populate the Milvus database with the same 100k documents and cached embeddings:
```bash
python -m src.setup_milvus 100000
```

### 2. Run Benchmarks

#### Option A: pgvector vs FAISS (Standard)
```bash
# Comprehensive index comparison (pgvector vs FAISS)
python -m src.benchmarks.index_comparison
```

#### Option B: pgvector vs pgvector+FAISS vs Milvus (New Three-Way Benchmark)
To run the benchmark suite comparing all three configurations:
```bash
python -m src.benchmarks.compare_all
```
This runs:
- 50 queries across 3 runs
- 3 engines: pgvector, pgvector+FAISS (hybrid), and Milvus Lite
- All index types: Flat, IVF, HNSW
- Metrics: Latency, Recall@K, Precision@K, F1@K, MRR

### 3. Generate Graphs

Graphs are generated automatically in `data/results/`:

| Graph | Description |
|-------|-------------|
| `compare_latency.png` | Search latency comparison (log scale) |
| `compare_recall.png` | Recall@5 comparison |
| `compare_speedup.png` | Speedup factor comparison vs pgvector Flat |
| `compare_summary.png` | Complete comparison summary table |
| `graph_*.png` | Standard pgvector vs FAISS graphs |

## Project Structure

```
src/
├── config.py              # Configuration
├── exceptions.py          # Custom exceptions
├── setup_db.py           # Database setup
├── setup_milvus.py       # Milvus setup [NEW]
├── test_rag.py           # Interactive RAG query
├── benchmark.py          # Legacy benchmark
├── benchmark_metadata.py # Metadata filtering benchmark
├── benchmarks/           # New benchmark suite
│   ├── __init__.py
│   ├── latency.py        # Component latency
│   ├── recall.py         # Recall@K benchmark
│   ├── precision.py      # Precision@K benchmark
│   ├── comprehensive.py # Full benchmark runner
│   ├── index_comparison.py # Main comparison
│   ├── compare_all.py     # Three-way comparison [NEW]
│   └── graphs.py        # Graph generation
├── data/
│   └── loader.py         # Dataset loading
├── db/
│   ├── pgvector.py     # PostgreSQL/pgvector wrapper
│   ├── faiss_index.py  # FAISS index wrapper
│   └── milvus.py       # Milvus wrapper [NEW]
└── rag/
    └── generator.py    # RAG pipeline
```

## Performance Results

| Engine | Index | Latency (ms) | Recall | Precision | F1 | Speedup |
|--------|-------|--------------|--------|-----------|-----|---------|
| pgvector | Flat | ~320ms | 100% | 100% | 100% | 1x |
| pgvector | IVF | ~70ms | 66% | 66% | 66% | - |
| pgvector | HNSW | ~290ms | 100% | 100% | 100% | - |
| FAISS | Flat | ~50ms | 100% | 100% | 100% | 6x |
| FAISS | IVF | ~1ms | 96% | 96% | 96% | 70x |
| FAISS | HNSW | ~0.5ms | 97% | 97% | 97% | 600x |

**Key Findings:**
- FAISS HNSW: **600x faster** than pgvector with ~97% recall
- FAISS IVF: **70x faster** than pgvector with 96% recall
- Statistical significance: 3 runs × 50 queries

## Benchmark Files

- `data/results/comprehensive_benchmark.json` - Full results
- `data/results/index_comparison_*.json` - Index comparison results
- `data/results/graph_*.png` - Generated graphs

## Requirements

See `requirements.txt` for full dependency list.