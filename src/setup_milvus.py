"""Setup script - run this ONCE to populate the Milvus database."""

import sys
from src.data.loader import load_sample_data, load_raw_documents
from src.db.milvus import MilvusDB

def main():
    print("=" * 50)
    print("MILVUS DATABASE SETUP")
    print("=" * 50)
    
    # Check if a command line arg is provided, else prompt
    if len(sys.argv) > 1:
        try:
            num_samples = int(sys.argv[1])
        except ValueError:
            num_samples = 1000
    else:
        num_samples = int(input("How many documents? (default: 1000): ") or "1000")

    print(f"\n1. Loading/generating embeddings for {num_samples} documents...")
    embeddings, ids = load_sample_data(num_samples)
    print(f"   Done: {len(embeddings)} docs, {embeddings.shape[1]}D")
    
    print("\n2. Setting up Milvus collection...")
    db = MilvusDB()
    
    print("   Dropping old collection...")
    db.drop_collection()
    
    print("   Creating new collection...")
    db.create_collection(dimension=embeddings.shape[1])
    
    print("   Loading raw documents...")
    documents = load_raw_documents(num_samples)
    
    print("   Inserting documents...")
    batch_size = 1000  # Milvus handles larger batch size efficiently
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        db.insert_batch(
            documents=[d["content"] for d in batch],
            embeddings=embeddings[i:i+batch_size],
            authors=["arXiv"] * len(batch),
            categories=["cs.AI"] * len(batch),
            ids=[d["id"] for d in batch],
        )
        print(f"   {min(i+batch_size, len(documents))}/{len(documents)}")
    
    print("\n3. Building indexes (this creates the HNSW index as default)...")
    # Milvus requires index creation before loaded to search. We'll default to FLAT first
    # and then build others. 
    db.create_indexes("flat")
    db.create_indexes("ivfflat", 100)
    db.create_indexes("hnsw", 16)
    
    print(f"\n4. Total documents in Milvus: {db.count()}")
    print("\n" + "=" * 50)
    print("MILVUS SETUP COMPLETE!")
    print("=" * 50)

if __name__ == "__main__":
    main()
