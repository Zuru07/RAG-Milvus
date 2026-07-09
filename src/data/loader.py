"""Dataset loading from HuggingFace."""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from datasets import load_dataset

from src.config import DATASET_NAME, EMBEDDING_CONFIG


def load_sample_data(
    num_samples: int = 10000,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load pre-generated embeddings from cache or generate if missing.

    Args:
        num_samples: Number of samples to load.
        cache_dir: Directory for cached embeddings.

    Returns:
        Tuple of (embeddings array, document ids).
    """
    cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
    embeddings_path = cache_dir / "embeddings.npy"
    ids_path = cache_dir / "ids.npy"

    if embeddings_path.exists() and ids_path.exists():
        embeddings = np.load(embeddings_path)
        ids = np.load(ids_path).tolist()
        if len(embeddings) >= num_samples:
            return embeddings[:num_samples], ids[:num_samples]

    return generate_embeddings(num_samples, cache_dir)


def generate_embeddings(
    num_samples: int = 10000,
    cache_dir: Optional[str] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Generate embeddings from HuggingFace dataset.

    Args:
        num_samples: Number of samples to process.
        cache_dir: Directory to save embeddings.

    Returns:
        Tuple of (embeddings array, document ids).
    """
    from sentence_transformers import SentenceTransformer

    cache_dir = Path(cache_dir) if cache_dir else Path("data/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")

    texts = [dataset[i]["abstract"] for i in range(min(num_samples, len(dataset)))]

    print(f"Loading embedding model: {EMBEDDING_CONFIG.model_name}...")
    import torch
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    model = SentenceTransformer(EMBEDDING_CONFIG.model_name, device=device)

    print(f"Generating embeddings for {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        batch_size=EMBEDDING_CONFIG.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    ids = list(range(len(embeddings)))

    embeddings_path = cache_dir / "embeddings.npy"
    ids_path = cache_dir / "ids.npy"
    np.save(embeddings_path, embeddings)
    np.save(ids_path, np.array(ids))

    print(f"Saved {len(embeddings)} embeddings to {cache_dir}")

    return embeddings, ids


def load_raw_documents(
    num_samples: int = 10000,
) -> List[dict]:
    """Load raw documents from dataset.

    Args:
        num_samples: Number of samples to load.

    Returns:
        List of document dictionaries.
    """
    dataset = load_dataset(DATASET_NAME, split="train")

    documents = []
    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        documents.append({
            "id": i,
            "content": item["abstract"],
            "title": item.get("title", ""),
            "authors": item.get("authors", []),
        })

    return documents
