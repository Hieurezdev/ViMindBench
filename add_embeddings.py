

import os
from pymongo import MongoClient
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

def add_embeddings_to_mongodb():
    """Add embedding vectors to all documents in MongoDB."""
    
    # Connect to MongoDB
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME", "Data")
    collection_name = os.getenv("MONGO_COLLECTION_NAME", "mental")
    
    client = MongoClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]
    
    # Initialize local embedding client
    embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:1234/v1")
    embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b")
    
    print(f"Using embedding model: {embedding_model}")
    print(f"Endpoint: {embedding_base_url}")
    
    embedding_client = OpenAI(
        base_url=embedding_base_url,
        api_key="dummy"
    )
    
    # Get all documents
    total_docs = collection.count_documents({})
    print(f"Processing {total_docs} documents...")
    
    # Process documents in batches
    batch_size = 100
    cursor = collection.find({}).batch_size(batch_size)
    
    updated_count = 0
    error_count = 0
    
    for doc in tqdm(cursor, total=total_docs):
        # Skip if embedding already exists
        if 'embedding' in doc:
            continue
        
        # Generate embedding from content or summary
        text = doc.get('content', doc.get('summary', ''))
        if not text:
            continue
        
        try:
            # Call local embedding API
            response = embedding_client.embeddings.create(
                model=embedding_model,
                input=text  # Limit text length
            )
            embedding = response.data[0].embedding
            
            # Update document
            collection.update_one(
                {"_id": doc["_id"]},
                {"$set": {"embedding": embedding}}
            )
            updated_count += 1
            
        except Exception as e:
            print(f"\nError processing doc {doc.get('_id')}: {e}")
            error_count += 1
            continue
    
    print(f"\n✓ Added embeddings to {updated_count} documents")
    if error_count > 0:
        print(f"⚠ {error_count} errors occurred")
    client.close()

if __name__ == "__main__":
    add_embeddings_to_mongodb()
