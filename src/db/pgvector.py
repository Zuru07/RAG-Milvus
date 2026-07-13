"""PostgreSQL/pgvector operations with indexing strategies."""

import json
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Generator, List, Optional

import numpy as np
import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

from src.config import DB_CONFIG, EMBEDDING_CONFIG
from src.exceptions import DatabaseConnectionError


@dataclass
class SearchResult:
    id: int
    content: str
    distance: float
    metadata: dict


class PGVectorDB:
    """PostgreSQL/pgvector database wrapper with indexing support."""

    def __init__(self, config: DB_CONFIG = DB_CONFIG):
        self.config = config
        self.dimension = EMBEDDING_CONFIG.dimension

    @contextmanager
    def connection(self, timeout: int = 30) -> Generator[psycopg2.extensions.connection, None, None]:
        """Context manager for database connections."""
        try:
            conn = psycopg2.connect(
                dbname=self.config.name,
                user=self.config.user,
                password=self.config.password,
                host=self.config.host,
                port=self.config.port,
                connect_timeout=timeout,
            )
            yield conn
        except psycopg2.OperationalError as e:
            raise DatabaseConnectionError(f"Failed to connect: {e}") from e

    def create_table(self, dimension: int = EMBEDDING_CONFIG.dimension) -> None:
        """Create documents table with metadata columns and vector index."""
        self.dimension = dimension
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

                cur.execute("""
                    CREATE TABLE IF NOT EXISTS documents (
                        id SERIAL PRIMARY KEY,
                        content TEXT NOT NULL,
                        embedding vector(%s),
                        author VARCHAR(255),
                        date DATE,
                        category VARCHAR(100),
                        tags TEXT[],
                        source VARCHAR(255),
                        metadata JSONB DEFAULT '{}'::jsonb,
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """, (dimension,))

                conn.commit()

    def create_indexes(self, index_type: str = "ivfflat", nlist: int = 100, ef_construction: int = 100, ef_search: int = 50) -> None:
        """Create vector index with specified type."""
        drop_sql = "DROP INDEX IF EXISTS idx_documents_embedding;"
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SET statement_timeout = 0;")
                cur.execute(drop_sql)

                if index_type == "flat":
                    cur.execute("""
                        CREATE INDEX idx_documents_embedding 
                        ON documents USING ivfflat (embedding vector_l2_ops)
                        WITH (lists = 1);
                    """)
                elif index_type == "ivfflat":
                    cur.execute(sql.SQL("""
                        CREATE INDEX idx_documents_embedding 
                        ON documents USING ivfflat (embedding vector_l2_ops)
                        WITH (lists = {});
                    """).format(sql.Literal(nlist)))
                elif index_type == "hnsw":
                    cur.execute(sql.SQL("""
                        CREATE INDEX idx_documents_embedding 
                        ON documents USING hnsw (embedding vector_cosine_ops)
                        WITH (m = 16, ef_construction = {});
                    """).format(sql.Literal(ef_construction)))
                conn.commit()

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
    ) -> None:
        """Insert documents with embeddings and metadata in batch."""
        n = len(documents)
        if authors is None:
            authors = [None] * n
        if dates is None:
            dates = [None] * n
        if categories is None:
            categories = [None] * n
        if tags is None:
            tags = [None] * n
        if sources is None:
            sources = [None] * n
        if metadata_list is None:
            metadata_list = [{}] * n

        data = [
            (
                doc,
                emb.tolist() if isinstance(emb, np.ndarray) else emb,
                author,
                date_str if date_str else None,
                category,
                tag_list,
                source,
                json.dumps(meta),
            )
            for doc, emb, author, date_str, category, tag_list, source, meta
            in zip(documents, embeddings, authors, dates, categories, tags, sources, metadata_list)
        ]

        with self.connection() as conn:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """INSERT INTO documents 
                       (content, embedding, author, date, category, tags, source, metadata)
                       VALUES %s""",
                    data,
                )
                conn.commit()

    def search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        filters: Optional[dict] = None,
    ) -> List[SearchResult]:
        """Vector search with optional metadata filtering."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                filter_parts = []
                filter_params = []
                params: List[Any] = []

                if filters:
                    if "author" in filters:
                        filter_parts.append("author = %s")
                        filter_params.append(filters["author"])
                    if "category" in filters:
                        filter_parts.append("category = %s")
                        filter_params.append(filters["category"])
                    if "date_from" in filters:
                        filter_parts.append("date >= %s")
                        filter_params.append(filters["date_from"])
                    if "date_to" in filters:
                        filter_parts.append("date <= %s")
                        filter_params.append(filters["date_to"])
                    if "tags" in filters:
                        filter_parts.append("%s = ANY(tags)")
                        filter_params.append(filters["tags"])
                    if "source" in filters:
                        filter_parts.append("source = %s")
                        filter_params.append(filters["source"])

                where_clause = ""
                if filter_parts:
                    where_clause = "WHERE " + " AND ".join(filter_parts)

                # Order: distance calc, WHERE filters, ORDER BY, LIMIT
                params = [query_embedding] + filter_params + [query_embedding, limit]

                sql_query = f"""
                    SELECT id, content,
                           embedding <-> %s::vector AS distance,
                           metadata
                    FROM documents
                    {where_clause}
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                """
                cur.execute(sql_query, params)
                rows = cur.fetchall()

                return [
                    SearchResult(
                        id=row[0],
                        content=row[1],
                        distance=row[2],
                        metadata=row[3] or {},
                    )
                    for row in rows
                ]

    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        limit: int = 5,
    ) -> List[SearchResult]:
        """Combine vector search with full-text search using RRF."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                vec_literal = "[" + ",".join(map(str, query_embedding)) + "]"

                sql_query = """
                    WITH semantic_search AS (
                        SELECT id, content,
                               RANK() OVER (ORDER BY embedding <-> %s::vector) AS rank,
                               metadata
                        FROM documents
                        ORDER BY embedding <-> %s::vector
                        LIMIT 20
                    ),
                    keyword_search AS (
                        SELECT id, content,
                               RANK() OVER (ORDER BY ts_rank_cd(
                                   to_tsvector('english', content),
                                   plainto_tsquery('english', %s)
                               ) DESC) AS rank,
                               metadata
                        FROM documents
                        WHERE to_tsvector('english', content) @@ 
                              plainto_tsquery('english', %s)
                        LIMIT 20
                    )
                    SELECT 
                        COALESCE(s.id, k.id) AS id,
                        COALESCE(s.content, k.content) AS content,
                        (COALESCE(1.0 / (60 + s.rank), 0.0) + 
                         COALESCE(1.0 / (60 + k.rank), 0.0)) AS rrf_score,
                        COALESCE(s.metadata, k.metadata) AS metadata
                    FROM semantic_search s
                    FULL OUTER JOIN keyword_search k ON s.id = k.id
                    ORDER BY rrf_score DESC
                    LIMIT %s
                """

                cur.execute(sql_query, (
                    vec_literal, vec_literal,
                    query, query,
                    limit
                ))
                rows = cur.fetchall()

                return [
                    SearchResult(
                        id=row[0],
                        content=row[1],
                        distance=float(row[2]),
                        metadata=row[3] or {},
                    )
                    for row in rows
                ]

    def count(self) -> int:
        """Return total document count."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents")
                return cur.fetchone()[0]

    def get_all_embeddings(self, limit: int = 10000) -> tuple[np.ndarray, List[int]]:
        """Get embeddings and IDs for benchmarking (limited for speed)."""
        import io
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT id, embedding::text FROM documents ORDER BY id LIMIT {limit}")
                rows = cur.fetchall()
                ids = [row[0] for row in rows]
                
                embeddings_list = []
                for row in rows:
                    emb_str = row[1].strip('[]')
                    emb = np.fromstring(emb_str, sep=',', dtype='float32')
                    embeddings_list.append(emb)
                
                embeddings = np.vstack(embeddings_list)
                return embeddings, ids

    def drop_table(self) -> None:
        """Drop documents table."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS documents CASCADE")
                conn.commit()

    def get_categories(self, limit: int = 50) -> List[str]:
        """Get list of unique categories in the database."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT category FROM documents WHERE category IS NOT NULL LIMIT %s"
                )
                return [row[0] for row in cur.fetchall()]

    def get_document_by_id(self, doc_id: int) -> Optional[SearchResult]:
        """Get a document by its ID."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, content, metadata FROM documents WHERE id = %s",
                    (doc_id,),
                )
                row = cur.fetchone()
                if row:
                    return SearchResult(
                        id=row[0],
                        content=row[1],
                        distance=0.0,
                        metadata=row[2] or {},
                    )
                return None
