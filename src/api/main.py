"""FastAPI application for RAG system."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os

from src.db.pgvector import PGVectorDB, SearchResult

try:
    from src.db.faiss_index import FAISSIndex
    faiss_available = True
except ImportError:
    faiss_available = False

from src.db.milvus import MilvusDB
from src.rag.generator import RAGPipeline


FAISS_INDEX_PATH = Path("data/cache/faiss_index")


class SearchRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=100)
    filters: Optional[dict] = None


class SearchResult(BaseModel):
    id: int
    content: str
    distance: float
    metadata: dict


class RAGRequest(BaseModel):
    query: str
    limit: int = Field(default=5, ge=1, le=20)
    use_hybrid: bool = True
    use_faiss: bool = False
    use_milvus: bool = False
    filters: Optional[dict] = None
    stream: bool = False


class RAGResponse(BaseModel):
    answer: str
    documents: List[SearchResult]
    query: str
    retrieval_engine: str


class StatsResponse(BaseModel):
    total_documents: int
    total_milvus_documents: int
    categories: List[str]
    has_faiss_index: bool
    has_milvus_collection: bool


class HealthResponse(BaseModel):
    status: str
    database: str
    faiss_loaded: bool
    milvus_loaded: bool
    model_loaded: bool


db: Optional[PGVectorDB] = None
faiss_index: Optional[FAISSIndex] = None
milvus_db: Optional[MilvusDB] = None
rag_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, faiss_index, milvus_db, rag_pipeline
    
    print("Starting up RAG API...")
    print("Connecting to PostgreSQL...")
    db = PGVectorDB()
    
    print("Loading FAISS index...")
    faiss_index = None
    if faiss_available:
        try:
            if FAISS_INDEX_PATH.exists():
                faiss_index = FAISSIndex.load(str(FAISS_INDEX_PATH))
                print(f"FAISS index loaded: {faiss_index.total_vectors} vectors")
            else:
                print("FAISS index not found at", FAISS_INDEX_PATH)
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")
    else:
        print("FAISS library not installed. Skipping local FAISS loading.")

    print("Connecting to Milvus...")
    try:
        milvus_db = MilvusDB()
        print(f"Milvus connected: {milvus_db.count()} documents")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        milvus_db = None
    
    print("Loading RAG pipeline...")
    rag_pipeline = RAGPipeline(db=db)
    
    print("RAG API ready!")
    yield
    
    print("Shutting down RAG API...")


app = FastAPI(
    title="RAG Vector Search API",
    description="High-performance RAG pipeline with pgvector and FAISS",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        database="connected" if db is not None else "disconnected",
        faiss_loaded=faiss_index is not None and faiss_index.total_vectors > 0,
        milvus_loaded=milvus_db is not None and milvus_db.count() > 0,
        model_loaded=rag_pipeline is not None,
    )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        categories = db.get_categories()
    except Exception:
        categories = []
        
    milvus_count = 0
    if milvus_db is not None:
        try:
            milvus_count = milvus_db.count()
        except Exception:
            pass
    
    return StatsResponse(
        total_documents=db.count(),
        total_milvus_documents=milvus_count,
        categories=categories,
        has_faiss_index=faiss_index is not None and faiss_index.total_vectors > 0,
        has_milvus_collection=milvus_db is not None and milvus_count > 0,
    )


@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not loaded")
    
    query_embedding = rag_pipeline.get_query_embedding(request.query)
    results = db.search(query_embedding, limit=request.limit, filters=request.filters)
    
    return [
        SearchResult(
            id=r.id,
            content=r.content,
            distance=r.distance,
            metadata=r.metadata or {},
        )
        for r in results
    ]


@app.post("/search/faiss", response_model=List[SearchResult])
async def search_faiss(request: SearchRequest):
    if faiss_index is None or faiss_index.total_vectors == 0:
        raise HTTPException(status_code=503, detail="FAISS index not loaded")
    
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not loaded")
    
    query_embedding = rag_pipeline.get_query_embedding(request.query)
    results = faiss_index.search(query_embedding, k=request.limit)
    
    return [
        SearchResult(
            id=r["id"],
            content=r["content"],
            distance=r["distance"],
            metadata=r.get("metadata", {}),
        )
        for r in results
    ]


@app.post("/search/milvus", response_model=List[SearchResult])
async def search_milvus(request: SearchRequest):
    if milvus_db is None or milvus_db.count() == 0:
        raise HTTPException(status_code=503, detail="Milvus database not connected or empty")
    
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not loaded")
    
    query_embedding = rag_pipeline.get_query_embedding(request.query)
    results = milvus_db.search(query_embedding, limit=request.limit, filters=request.filters)
    
    return [
        SearchResult(
            id=r.id,
            content=r.content,
            distance=r.distance,
            metadata=r.metadata or {},
        )
        for r in results
    ]


@app.post("/rag", response_model=RAGResponse)
async def rag_query(request: RAGRequest):
    if rag_pipeline is None or db is None:
        raise HTTPException(status_code=503, detail="RAG pipeline or database not loaded")
    
    retrieval_engine = "pgvector"
    
    if request.use_milvus and milvus_db is not None and milvus_db.count() > 0:
        query_embedding = rag_pipeline.get_query_embedding(request.query)
        docs = milvus_db.search(query_embedding, limit=request.limit, filters=request.filters)
        retrieval_engine = "milvus"
    elif request.use_faiss and faiss_index is not None and faiss_index.total_vectors > 0:
        query_embedding = rag_pipeline.get_query_embedding(request.query)
        _, _, faiss_results = faiss_index.search(query_embedding, k=request.limit)
        
        docs = []
        for r in faiss_results:
            pg_result = db.get_document_by_id(r["id"])
            if pg_result:
                pg_result.distance = r["distance"]
                docs.append(pg_result)
        
        retrieval_engine = "faiss"
    else:
        docs = rag_pipeline.retrieve(
            request.query,
            limit=request.limit,
            use_hybrid=request.use_hybrid,
            filters=request.filters,
        )
    
    if not docs:
        return RAGResponse(
            answer="No documents found matching your query.",
            documents=[],
            query=request.query,
            retrieval_engine=retrieval_engine,
        )
    
    answer = rag_pipeline.generate(request.query, docs, stream=False)
    
    return RAGResponse(
        answer=answer,
        documents=[
            SearchResult(
                id=d.id,
                content=d.content,
                distance=d.distance,
                metadata=d.metadata or {},
            )
            for d in docs
        ],
        query=request.query,
        retrieval_engine=retrieval_engine,
    )


@app.get("/")
async def root():
    return {
        "name": "RAG Vector Search API",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
