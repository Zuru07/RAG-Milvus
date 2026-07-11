"""Milvus/pymilvus operations with indexing strategies."""

import os
import json
from typing import List, Optional, Generator, Tuple
from dataclasses import dataclass
import numpy as np
from pymilvus import MilvusClient, DataType

from src.config import EMBEDDING_CONFIG
from src.db.pgvector import SearchResult
from src.exceptions import DatabaseConnectionError


class MilvusDB:
    """Milvus database wrapper with indexing support using Milvus Lite."""

    def __init__(self, db_path: str = "data/cache/milvus_rag.db", collection_name: str = "arxiv_papers"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.dimension = EMBEDDING_CONFIG.dimension
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
            self.client = MilvusClient(uri=self.db_path)
            # Auto-load the collection if it already exists
            if self.client.has_collection(self.collection_name):
                self.client.load_collection(self.collection_name)
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to initialize Milvus client: {e}") from e

    def create_collection(self, dimension: int = EMBEDDING_CONFIG.dimension) -> None:
        """Create collection with schemas matching the pgvector document table."""
        self.dimension = dimension
        
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=True)
        
        # Primary Key (INT64, auto_id=False to control sequence IDs manually)
        schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
        # Content field
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        # Vector embedding field
        schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)
        # Metadata fields
        schema.add_field(field_name="author", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="date", datatype=DataType.VARCHAR, max_length=50)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=255)
        schema.add_field(field_name="metadata", datatype=DataType.JSON)

        # Create collection
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema
        )

    def create_indexes(self, index_type: str = "ivfflat", nlist: int = 100, ef_construction: int = 100, ef_search: int = 50) -> None:
        """Create vector index on the embedding field with specified type."""
        if not self.client.has_collection(self.collection_name):
            raise ValueError(f"Collection {self.collection_name} does not exist. Call create_collection() first.")

        # Release collection to modify indexes
        try:
            self.client.release_collection(self.collection_name)
        except Exception:
            pass

        # Drop index if it exists
        try:
            self.client.drop_index(self.collection_name, index_name="embedding")
        except Exception:
            pass

        # Prepare index parameters
        index_params = self.client.prepare_index_params()
        
        norm_idx = index_type.lower()
        if norm_idx in ["flat", "flat_l2"]:
            idx_type = "FLAT"
            params = {}
        elif norm_idx in ["ivfflat", "ivf", "ivf_flat"]:
            idx_type = "IVF_FLAT"
            params = {"nlist": nlist}
        elif norm_idx in ["hnsw"]:
            idx_type = "HNSW"
            params = {"M": 16, "efConstruction": ef_construction}
        else:
            raise ValueError(f"Unknown index type for Milvus: {index_type}")

        # Standardizing metric_type as L2 to match FAISS/pgvector flat benchmark
        index_params.add_index(
            field_name="embedding",
            index_name="embedding",
            index_type=idx_type,
            metric_type="L2",
            params=params
        )

        self.client.create_index(
            collection_name=self.collection_name,
            index_params=index_params
        )

        # Load the collection back into memory for searches
        self.client.load_collection(self.collection_name)

    def insert_batch(
        self,
        documents: List[str],
        embeddings: np.ndarray,
        authors: Optional[List[str]] = None,
        dates: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        tags: Optional[List[List[str]]] = None,
        sources: Optional[List[str]] = None,
        metadata_list: Optional[List[dict]] = None,
        ids: Optional[List[int]] = None,
    ) -> None:
        """Insert documents with embeddings and metadata in batch."""
        n = len(documents)
        if authors is None:
            authors = [""] * n
        if dates is None:
            dates = [""] * n
        if categories is None:
            categories = [""] * n
        if tags is None:
            tags = [[]] * n
        if sources is None:
            sources = [""] * n
        if metadata_list is None:
            metadata_list = [{}] * n
        if ids is None:
            # Generate sequential IDs based on current collection size
            current_size = self.count()
            ids = list(range(current_size, current_size + n))

        data = []
        for i in range(n):
            doc_id = int(ids[i])
            emb = embeddings[i]
            # Build clean metadata matching the schema
            meta_dict = metadata_list[i].copy() if metadata_list[i] else {}
            if tags[i]:
                meta_dict["tags"] = tags[i]

            data.append({
                "id": doc_id,
                "content": documents[i][:65000],
                "embedding": emb.tolist() if isinstance(emb, np.ndarray) else emb,
                "author": authors[i] if authors[i] is not None else "",
                "date": str(dates[i]) if dates[i] is not None else "",
                "category": categories[i] if categories[i] is not None else "",
                "source": sources[i] if sources[i] is not None else "",
                "metadata": meta_dict
            })

        self.client.insert(collection_name=self.collection_name, data=data)

    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filters: Optional[dict] = None,
        search_params: Optional[dict] = None,
    ) -> List[SearchResult]:
        """Vector search with optional metadata filtering."""
        if not self.client.has_collection(self.collection_name):
            return []
            
        try:
            self.client.load_collection(self.collection_name)
        except Exception:
            pass
            
        filter_exprs = []
        if filters:
            if "author" in filters:
                filter_exprs.append(f"author == '{filters['author']}'")
            if "category" in filters:
                filter_exprs.append(f"category == '{filters['category']}'")
            if "date_from" in filters:
                filter_exprs.append(f"date >= '{filters['date_from']}'")
            if "date_to" in filters:
                filter_exprs.append(f"date <= '{filters['date_to']}'")
            if "source" in filters:
                filter_exprs.append(f"source == '{filters['source']}'")
            if "tags" in filters and filters["tags"]:
                tag_val = filters["tags"]
                filter_exprs.append(f"JSON_CONTAINS(metadata, '{tag_val}', '$.tags')")

        filter_expr = " and ".join(filter_exprs) if filter_exprs else None

        # Execute search
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_embedding],
            limit=limit,
            filter=filter_expr,
            search_params=search_params,
            output_fields=["id", "content", "metadata"]
        )

        if not results or len(results) == 0:
            return []

        # results[0] is the hits list for the single query vector
        search_results = []
        for hit in results[0]:
            entity = hit.get("entity", {})
            search_results.append(
                SearchResult(
                    id=hit["id"],
                    content=entity.get("content", ""),
                    distance=hit["distance"],
                    metadata=entity.get("metadata", {})
                )
            )

        return search_results

    def count(self) -> int:
        """Return total document count."""
        try:
            stats = self.client.get_collection_stats(collection_name=self.collection_name)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0

    def get_document_by_id(self, doc_id: int) -> Optional[SearchResult]:
        """Get a document by its ID."""
        if not self.client.has_collection(self.collection_name):
            return None
            
        try:
            self.client.load_collection(self.collection_name)
        except Exception:
            pass

        try:
            res = self.client.get(
                collection_name=self.collection_name,
                ids=[int(doc_id)],
                output_fields=["id", "content", "metadata"]
            )
            if res and len(res) > 0:
                doc = res[0]
                return SearchResult(
                    id=doc["id"],
                    content=doc.get("content", ""),
                    distance=0.0,
                    metadata=doc.get("metadata", {})
                )
        except Exception:
            pass
        return None

    def get_categories(self, limit: int = 50) -> List[str]:
        """Get list of unique categories in the database."""
        try:
            res = self.client.query(
                collection_name=self.collection_name,
                filter="category != ''",
                output_fields=["category"],
                limit=1000
            )
            categories = list(set([item["category"] for item in res if "category" in item]))
            return categories[:limit]
        except Exception:
            return []

    def get_all_embeddings(self, limit: int = 10000) -> Tuple[np.ndarray, List[int]]:
        """Get embeddings and IDs for benchmarking."""
        res = self.client.query(
            collection_name=self.collection_name,
            filter="",
            output_fields=["id", "embedding"],
            limit=limit
        )
        
        ids = [item["id"] for item in res]
        embeddings = np.array([item["embedding"] for item in res], dtype="float32")
        return embeddings, ids

    def drop_collection(self) -> None:
        """Drop collection."""
        if self.client.has_collection(self.collection_name):
            self.client.drop_collection(self.collection_name)
