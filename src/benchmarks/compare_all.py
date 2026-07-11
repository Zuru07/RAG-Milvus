"""Comprehensive benchmark: pgvector vs pgvector+FAISS vs Milvus across all index types."""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.pgvector import PGVectorDB
from src.db.faiss_index import FAISSIndex
from src.db.milvus import MilvusDB
from src.data.loader import load_sample_data
from src.config import EMBEDDING_CONFIG
from sentence_transformers import SentenceTransformer

RESULTS_DIR = Path("data/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Full query set (from index_comparison.py)
QUERIES = [
    "deep learning neural networks",
    "machine learning optimization",
    "natural language processing",
    "computer vision techniques",
    "reinforcement learning algorithms",
    "transformer attention mechanism",
    "convolutional neural networks",
    "optimization gradient descent",
    "recurrent neural networks LSTM",
    "generative adversarial networks",
    "support vector machine",
    "bert language model",
    "image classification CNN",
    "graph neural networks",
    "federated learning",
    "meta learning few-shot",
    "attention is all you need",
    "word embedding BERT",
    "object detection YOLO",
    "neural machine translation",
    "semi-supervised learning",
    "self-supervised contrastive",
    "diffusion model generation",
    "stable diffusion",
    "large language model",
    "prompt engineering",
    "in-context learning",
    "chain of thought reasoning",
    "retrieval augmented generation",
    "vector database embedding",
    "approximate nearest neighbor",
    "hierarchical navigable small world",
    "inverted file index",
    "product quantization",
    "locality sensitive hashing",
    "semantic search embedding",
    "cross-encoder reranking",
    "dense retrieval sparse",
    "question answering system",
    "information retrieval",
    "document ranking",
    "passage retrieval",
    "semantic similarity",
    "text matching",
    "semantic embedding space",
    "neural search",
    "embedding quantization",
    "index compression",
    "memory efficient search",
]

NUM_RUNS = 3

INDEX_CONFIGS = {
    "flat": {
        "pgvector": {"index_type": "flat", "params": {}},
        "faiss": {"index_type": "flat", "params": {}},
        "milvus": {"index_type": "flat", "params": {}},
    },
    "ivf": {
        "pgvector": {"index_type": "ivfflat", "params": {"nlist": 100}},
        "faiss": {"index_type": "ivf", "params": {"nlist": 100, "nprobe": 10}},
        "milvus": {"index_type": "ivfflat", "params": {"nlist": 100}},
    },
    "hnsw": {
        "pgvector": {"index_type": "hnsw", "params": {"ef_construction": 100}},
        "faiss": {"index_type": "hnsw", "params": {"hnsw_m": 16, "ef_construction": 100, "ef_search": 50}},
        "milvus": {"index_type": "hnsw", "params": {"ef_construction": 100}},
    },
}


def benchmark_pgvector(db, query_embs, limit=5, warmup=3):
    """Benchmark pgvector search."""
    for _ in range(warmup):
        if query_embs.shape[0] > 0:
            db.search(query_embs[0].tolist(), limit=limit)
    
    times = []
    results = []
    for emb in query_embs:
        start = time.perf_counter()
        r = db.search(emb.tolist(), limit=limit)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        results.append([x.id for x in r])
    
    return {
        "times": times, 
        "avg_ms": np.mean(times), 
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "results": results
    }


def benchmark_pgvector_faiss(db, faiss_idx, query_embs, limit=5, warmup=3):
    """Benchmark pgvector + FAISS hybrid search."""
    # Warm-up (query + document fetch)
    for _ in range(warmup):
        if query_embs.shape[0] > 0:
            _, _, r = faiss_idx.search(query_embs[0], k=limit)
            for x in r:
                db.get_document_by_id(x["id"])
    
    times = []
    results = []
    for emb in query_embs:
        start = time.perf_counter()
        # 1. FAISS Search
        _, _, r = faiss_idx.search(emb, k=limit)
        # 2. PG Fetch
        docs = []
        for x in r:
            doc = db.get_document_by_id(x["id"])
            if doc:
                docs.append(doc)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        results.append([x["id"] for x in r])
    
    return {
        "times": times, 
        "avg_ms": np.mean(times), 
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "results": results
    }


def benchmark_milvus(milvus_db, query_embs, idx_type, limit=5, warmup=3):
    """Benchmark Milvus search."""
    search_params = None
    if idx_type == "ivfflat":
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    elif idx_type == "hnsw":
        search_params = {"metric_type": "L2", "params": {"ef": 50}}
    
    # Warmup
    for _ in range(warmup):
        if query_embs.shape[0] > 0:
            milvus_db.search(query_embs[0].tolist(), limit=limit, search_params=search_params)
            
    times = []
    results = []
    for emb in query_embs:
        start = time.perf_counter()
        r = milvus_db.search(emb.tolist(), limit=limit, search_params=search_params)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
        results.append([x.id for x in r])
        
    return {
        "times": times, 
        "avg_ms": np.mean(times), 
        "std_ms": np.std(times),
        "min_ms": np.min(times),
        "max_ms": np.max(times),
        "results": results
    }


def calculate_recall(predicted, ground_truth, k=5):
    """Calculate recall@K: what fraction of ground truth did we find?"""
    recalls = []
    for pred, gt in zip(predicted, ground_truth):
        pred_set = set(pred[:k])
        gt_set = set(gt[:k])
        if len(gt_set) == 0:
            recalls.append(0.0)
        else:
            recalls.append(len(pred_set & gt_set) / len(gt_set))
    return np.mean(recalls)


def calculate_precision(predicted, ground_truth, k=5):
    """Calculate precision@K."""
    precisions = []
    for pred, gt in zip(predicted, ground_truth):
        pred_set = set(pred[:k])
        gt_set = set(gt[:k])
        if k == 0:
            precisions.append(0.0)
        else:
            precisions.append(len(pred_set & gt_set) / k)
    return np.mean(precisions)


def calculate_mrr(predicted, ground_truth, k=5):
    """Mean Reciprocal Rank."""
    reciprocal_ranks = []
    for pred, gt in zip(predicted, ground_truth):
        gt_set = set(gt)
        found_rank = 0
        for i, doc_id in enumerate(pred[:k]):
            if doc_id in gt_set:
                found_rank = i + 1
                break
        if found_rank > 0:
            reciprocal_ranks.append(1.0 / found_rank)
        else:
            reciprocal_ranks.append(0.0)
    return np.mean(reciprocal_ranks)


def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def aggregate_runs(all_run_results):
    aggregated = {}
    for run_res in all_run_results:
        for r in run_res:
            key = (r["engine"], r["index_type"])
            if key not in aggregated:
                aggregated[key] = {
                    "engine": r["engine"],
                    "index_type": r["index_type"],
                    "latency_ms": [],
                    "recall": [],
                    "precision": [],
                    "mrr": [],
                    "f1": []
                }
            aggregated[key]["latency_ms"].append(r["latency_ms"])
            aggregated[key]["recall"].append(r["recall"])
            aggregated[key]["precision"].append(r["precision"])
            aggregated[key]["mrr"].append(r["mrr"])
            aggregated[key]["f1"].append(r["f1"])
            
    final_results = []
    for key, data in aggregated.items():
        final_results.append({
            "engine": data["engine"],
            "index_type": data["index_type"],
            "latency_ms": np.mean(data["latency_ms"]),
            "std_ms": np.std(data["latency_ms"]),
            "recall": np.mean(data["recall"]),
            "precision": np.mean(data["precision"]),
            "mrr": np.mean(data["mrr"]),
            "f1": np.mean(data["f1"]),
        })
    return final_results


def plot_latency(results, output_dir):
    index_types = ["flat", "ivf", "hnsw"]
    engines = ["pgvector", "pgvector+FAISS", "Milvus"]
    colors = {"pgvector": "#e74c3c", "pgvector+FAISS": "#3498db", "Milvus": "#2ecc71"}
    
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(index_types))
    width = 0.25
    
    for i, eng in enumerate(engines):
        lats = []
        for idx in index_types:
            r = next((x for x in results if x["engine"] == eng and x["index_type"] == idx), None)
            lats.append(r["latency_ms"] if r else 0)
        ax.bar(x + (i - 1) * width, lats, width, label=eng, color=colors[eng], edgecolor="black", alpha=0.9)
        
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.set_title("Search Latency Comparison (Log Scale)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Flat", "IVF", "HNSW"], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_dir / "compare_latency.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: compare_latency.png")


def plot_recall(results, output_dir):
    index_types = ["flat", "ivf", "hnsw"]
    engines = ["pgvector", "pgvector+FAISS", "Milvus"]
    colors = {"pgvector": "#e74c3c", "pgvector+FAISS": "#3498db", "Milvus": "#2ecc71"}
    
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(index_types))
    width = 0.25
    
    for i, eng in enumerate(engines):
        recs = []
        for idx in index_types:
            r = next((x for x in results if x["engine"] == eng and x["index_type"] == idx), None)
            recs.append(r["recall"] if r else 0)
        ax.bar(x + (i - 1) * width, recs, width, label=eng, color=colors[eng], edgecolor="black", alpha=0.9)
        
    ax.set_ylabel("Recall@5", fontsize=12)
    ax.set_title("Recall@5 Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Flat", "IVF", "HNSW"], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_dir / "compare_recall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: compare_recall.png")


def plot_speedup(results, output_dir):
    index_types = ["flat", "ivf", "hnsw"]
    engines = ["pgvector+FAISS", "Milvus"]
    colors = {"pgvector+FAISS": "#3498db", "Milvus": "#2ecc71"}
    
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(index_types))
    width = 0.35
    
    # Baseline is pgvector Flat
    baseline = next(r["latency_ms"] for r in results if r["engine"] == "pgvector" and r["index_type"] == "flat")
    
    for i, eng in enumerate(engines):
        speedups = []
        for idx in index_types:
            r = next((x for x in results if x["engine"] == eng and x["index_type"] == idx), None)
            speedups.append(baseline / r["latency_ms"] if r and r["latency_ms"] > 0 else 0)
            
        bars = ax.bar(x + (i - 0.5) * width, speedups, width, label=f"{eng} Speedup", color=colors[eng], edgecolor="black", alpha=0.9)
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height * 1.05 if height > 1 else height + 0.1,
                    f"{height:.1f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")
            
    ax.set_ylabel("Speedup (vs pgvector Flat)", fontsize=12)
    ax.set_title("Search Speedup vs. Baseline pgvector Flat", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["Flat", "IVF", "HNSW"], fontsize=11)
    ax.legend(fontsize=11)
    ax.axhline(y=1, color="red", linestyle="--", alpha=0.5)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(output_dir / "compare_speedup.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: compare_speedup.png")


def plot_summary_table(results, output_dir):
    index_types = ["flat", "ivf", "hnsw"]
    engines = ["pgvector", "pgvector+FAISS", "Milvus"]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    
    table_data = [["Index", "Engine", "Latency (ms)", "Recall@5", "Precision@5", "F1@5", "Speedup"]]
    
    baseline = next(r["latency_ms"] for r in results if r["engine"] == "pgvector" and r["index_type"] == "flat")
    
    for idx in index_types:
        for eng in engines:
            r = next((x for x in results if x["engine"] == eng and x["index_type"] == idx), None)
            if r:
                speedup_val = baseline / r["latency_ms"] if r["latency_ms"] > 0 else 0
                speedup_str = f"{speedup_val:.1f}x"
                std_str = f" ± {r['std_ms']:.2f}" if "std_ms" in r else ""
                table_data.append([
                    idx.upper(), eng, f"{r['latency_ms']:.2f}{std_str}",
                    f"{r['recall']:.1%}", f"{r['precision']:.1%}",
                    f"{r['f1']:.1%}", speedup_str
                ])
                
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(color="white", fontweight="bold")
    ax.set_title("Performance Summary (pgvector vs pgvector+FAISS vs Milvus)", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(output_dir / "compare_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: compare_summary.png")


def main():
    print("=" * 80)
    print("COMPREHENSIVE RAG ENGINE COMPARISON: pgvector vs pgvector+FAISS vs Milvus")
    print("=" * 80)
    
    print("\n[1/5] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_CONFIG.model_name)
    query_embs = model.encode(QUERIES)
    
    print("[2/5] Loading cached embeddings...")
    embeddings, ids = load_sample_data(100000)
    print(f"  Loaded {len(embeddings)} embeddings, {embeddings.shape[1]}D")
    
    print("[3/5] Connecting to Databases...")
    pg_db = PGVectorDB()
    milvus_db = MilvusDB()
    print(f"  pgvector document count: {pg_db.count()}")
    print(f"  Milvus document count:   {milvus_db.count()}")
    
    if milvus_db.count() == 0:
        print("\n[WARNING] Milvus VDB appears empty. Run setup first:")
        print("  python -m src.setup_milvus 100000")
        sys.exit(1)
        
    all_runs_results = [[] for _ in range(NUM_RUNS)]
    ground_truths = {}
    
    print(f"\n[4/5] Running benchmark suite (3 index configurations x {NUM_RUNS} runs)...")
    
    index_types = ["flat", "ivf", "hnsw"]
    for idx_type in index_types:
        print(f"\n{'='*60}")
        print(f"INDEX CONFIGURATION: {idx_type.upper()}")
        print(f"{'='*60}")
        
        config = INDEX_CONFIGS[idx_type]
        
        # 1. Build indexes (exactly once)
        # A. pgvector
        print(f"\n  Building pgvector {idx_type.upper()} index...")
        pg_params = config["pgvector"]["params"]
        pg_db.create_indexes(config["pgvector"]["index_type"], **pg_params)
        
        # B. pgvector+FAISS (only FAISS index is built/loaded)
        print(f"  Building FAISS {idx_type.upper()} index...")
        fa_params = config["faiss"]["params"]
        faiss_idx = FAISSIndex(
            dimension=EMBEDDING_CONFIG.dimension,
            index_type=config["faiss"]["index_type"],
            **{k: v for k, v in fa_params.items() if k in ["nlist", "nprobe", "hnsw_m", "hnsw_ef_construction", "hnsw_ef_search"]}
        )
        faiss_idx.build(embeddings, ids)
        
        # C. Milvus
        print(f"  Building Milvus {idx_type.upper()} index...")
        mil_params = config["milvus"]["params"]
        milvus_db.create_indexes(config["milvus"]["index_type"], **mil_params)
        
        # 2. Run query searches for each run
        for run_num in range(1, NUM_RUNS + 1):
            print(f"\n  RUN {run_num}/{NUM_RUNS}:")
            
            print("    pgvector...")
            pg_res = benchmark_pgvector(pg_db, query_embs, limit=5)
            
            print("    pgvector+FAISS...")
            fa_res = benchmark_pgvector_faiss(pg_db, faiss_idx, query_embs, limit=5)
            
            print("    Milvus...")
            mil_res = benchmark_milvus(milvus_db, query_embs, idx_type, limit=5)
            
            if idx_type == "flat":
                if run_num not in ground_truths:
                    ground_truths[run_num] = {}
                ground_truths[run_num]["pgvector"] = pg_res["results"]
                ground_truths[run_num]["faiss"] = fa_res["results"]
                ground_truths[run_num]["milvus"] = mil_res["results"]
                
                pg_recall, pg_prec, pg_f1, pg_mrr = 1.0, 1.0, 1.0, 1.0
                fa_recall, fa_prec, fa_f1, fa_mrr = 1.0, 1.0, 1.0, 1.0
                mil_recall, mil_prec, mil_f1, mil_mrr = 1.0, 1.0, 1.0, 1.0
            else:
                run_gt = ground_truths[run_num]
                
                pg_recall = calculate_recall(pg_res["results"], run_gt["pgvector"])
                pg_prec = calculate_precision(pg_res["results"], run_gt["pgvector"])
                pg_mrr = calculate_mrr(pg_res["results"], run_gt["pgvector"])
                pg_f1 = calculate_f1(pg_prec, pg_recall)
                
                fa_recall = calculate_recall(fa_res["results"], run_gt["faiss"])
                fa_prec = calculate_precision(fa_res["results"], run_gt["faiss"])
                fa_mrr = calculate_mrr(fa_res["results"], run_gt["faiss"])
                fa_f1 = calculate_f1(fa_prec, fa_recall)
                
                mil_recall = calculate_recall(mil_res["results"], run_gt["milvus"])
                mil_prec = calculate_precision(mil_res["results"], run_gt["milvus"])
                mil_mrr = calculate_mrr(mil_res["results"], run_gt["milvus"])
                mil_f1 = calculate_f1(mil_prec, mil_recall)
                
            all_runs_results[run_num - 1].append({
                "engine": "pgvector", "index_type": idx_type,
                "latency_ms": pg_res["avg_ms"], "std_ms": pg_res["std_ms"],
                "recall": pg_recall, "precision": pg_prec, "mrr": pg_mrr, "f1": pg_f1
            })
            all_runs_results[run_num - 1].append({
                "engine": "pgvector+FAISS", "index_type": idx_type,
                "latency_ms": fa_res["avg_ms"], "std_ms": fa_res["std_ms"],
                "recall": fa_recall, "precision": fa_prec, "mrr": fa_mrr, "f1": fa_f1
            })
            all_runs_results[run_num - 1].append({
                "engine": "Milvus", "index_type": idx_type,
                "latency_ms": mil_res["avg_ms"], "std_ms": mil_res["std_ms"],
                "recall": mil_recall, "precision": mil_prec, "mrr": mil_mrr, "f1": mil_f1
            })
        
    print("\n[5/5] Aggregating results and generating graphs...")
    aggregated_results = aggregate_runs(all_runs_results)
    
    # Plot results
    plot_latency(aggregated_results, RESULTS_DIR)
    plot_recall(aggregated_results, RESULTS_DIR)
    plot_speedup(aggregated_results, RESULTS_DIR)
    plot_summary_table(aggregated_results, RESULTS_DIR)
    
    # Save raw json output
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"compare_all_engines_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "queries": len(QUERIES),
            "runs": NUM_RUNS,
            "results": aggregated_results
        }, f, indent=2)
        
    print("\n" + "=" * 90)
    print("FINAL SUMMARY (Aggregated over runs)")
    print("=" * 90)
    print(f"{'Engine':<16} {'Index':<6} {'Latency (ms)':<16} {'Recall':<8} {'Precision':<10} {'F1':<8}")
    print("-" * 70)
    for r in aggregated_results:
        std_val = r.get("std_ms", 0)
        lat_str = f"{r['latency_ms']:.2f} ± {std_val:.2f}"
        print(f"{r['engine']:<16} {r['index_type']:<6} {lat_str:<16} {r['recall']:.2%}    {r['precision']:.2%}    {r['f1']:.2%}")
    print(f"\nGraphs saved to: {RESULTS_DIR}/")
    print(f"JSON data saved to: {out_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
