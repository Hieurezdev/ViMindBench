from typing import List, Dict, Any
import os
from pymongo import MongoClient
from openai import OpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
import numpy as np

load_dotenv()

class MongoDBRetriever:
    """
    MongoDB-based retriever using:
    1. Atlas Vector Search (if available)
    2. Text search fallback
    3. Local embedding model via OpenAI API
    """
    
    def __init__(self, 
                 embedding_base_url: str = None,
                 embedding_model: str = None):
        """
        Initialize MongoDB retriever with local embedding model.
        
        Args:
            embedding_base_url: Base URL for local embedding server (default from env)
            embedding_model: Model name for embeddings (default from env)
        """
        # MongoDB connection
        self.mongo_uri = os.getenv("MONGO_URI")
        self.db_name = os.getenv("MONGO_DB_NAME", "Data")
        self.collection_name = os.getenv("MONGO_COLLECTION_NAME", "mental")
        
        if not self.mongo_uri:
            raise ValueError("MONGO_URI environment variable not set")
        
        self.client = MongoClient(self.mongo_uri)
        self.db = self.client[self.db_name]
        self.collection = self.db[self.collection_name]
        
        # Embedding client (local OpenAI-compatible API)
        self.embedding_base_url = embedding_base_url or os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:1234/v1")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "text-embedding-qwen3-embedding-0.6b")
        
        self.embedding_client = OpenAI(
            base_url=self.embedding_base_url,
            api_key="dummy"  # Local models don't need real key
        )
        
        print(f"✓ Connected to MongoDB: {self.db_name}.{self.collection_name}")
        print(f"✓ Using local embedding model: {self.embedding_model}")
        
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector using local OpenAI-compatible API."""
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Warning: Embedding generation failed: {e}")
            # Return zero vector as fallback (Qwen embedding model uses 1024 dimensions)
            return [0.0] * 1024
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Add documents to MongoDB with embeddings.
        
        Note: In production, you might want to batch this and run async.
        For now, we assume documents are already in MongoDB.
        """
        print(f"MongoDB retriever uses existing collection: {self.collection_name}")
        print(f"Total documents in collection: {self.collection.count_documents({})}")
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search MongoDB for relevant documents.
        
        Strategy:
        1. Try Atlas Vector Search (requires vector index)
        2. Fallback to text search
        3. Fallback to keyword matching
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of LangChain Document objects
        """
        try:
            # Strategy 1: Try Atlas Vector Search (requires atlas_vector_search index)
            return self._vector_search(query, k)
        except Exception as e:
            print(f"Vector search failed: {e}, falling back to text search")
            try:
                # Strategy 2: MongoDB text search (requires text index)
                return self._text_search(query, k)
            except Exception as e2:
                print(f"Text search failed: {e2}, falling back to keyword search")
                # Strategy 3: Simple keyword search
                return self._keyword_search(query, k)
    
    def _vector_search(self, query: str, k: int) -> List[Document]:
        """
        Perform Atlas Vector Search.
        
        Requires: Vector search index on 'embedding' field in Atlas
        """
        query_embedding = self._generate_embedding(query)
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",  # User specified index name
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": k * 10,
                    "limit": k
                }
            },
            {
                "$project": {
                    "content": 1,
                    "summary": 1,
                    "keywords": 1,
                    "uuid": 1,
                    "headers": 1,
                    "type": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(self.collection.aggregate(pipeline))
        return self._format_results(results)
    
    def _text_search(self, query: str, k: int) -> List[Document]:
        """
        Fallback to Mongo Atlas Text Search.
        
        Requires: Atlas Search index named 'atlas_index'
        """
        pipeline = [
            {
                "$search": {
                    "index": "atlas_index",  # User specified index name
                    "text": {
                        "query": query,
                        "path": ["content", "summary", "keywords"]
                    }
                }
            },
            {
                "$limit": k
            },
            {
                "$project": {
                    "content": 1,
                    "summary": 1,
                    "keywords": 1,
                    "uuid": 1,
                    "headers": 1,
                    "type": 1,
                    "score": {"$meta": "searchScore"}
                }
            }
        ]
        
        try:
            results = list(self.collection.aggregate(pipeline))
            return self._format_results(results)
        except Exception as e:
            print(f"Atlas text search failed: {e}. Trying standard text search fallback.")
            # Fallback to standard Mongo $text search if Atlas Search fails (e.g. index not found)
            return self._standard_text_search(query, k)

    def _standard_text_search(self, query: str, k: int) -> List[Document]:
        """
        Standard MongoDB text search ($text).
        """
        results = self.collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(k)
        
        return self._format_results(list(results))
    
    def _keyword_search(self, query: str, k: int) -> List[Document]:
        """
        Simple keyword-based search using regex.
        """
        # Split query into keywords
        keywords = query.lower().split()
        
        # Search in content, summary, and keywords fields
        regex_patterns = [{"$regex": kw, "$options": "i"} for kw in keywords]
        
        results = self.collection.find({
            "$or": [
                {"content": {"$in": regex_patterns}},
                {"summary": {"$in": regex_patterns}},
                {"keywords": {"$in": keywords}}
            ]
        }).limit(k)
        
        return self._format_results(list(results))
    
    def _format_results(self, results: List[Dict]) -> List[Document]:
        """Convert MongoDB results to LangChain Documents."""
        documents = []
        
        for result in results:
            # Use 'content' as page_content
            page_content = result.get('content', result.get('summary', ''))
            
            # Metadata
            metadata = {
                'uuid': result.get('uuid'),
                'headers': result.get('headers', []),
                'summary': result.get('summary'),
                'keywords': result.get('keywords', []),
                'type': result.get('type'),
                'score': result.get('score', 0.0)
            }
            
            documents.append(Document(
                page_content=page_content,
                metadata=metadata
            ))
        
        return documents
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            print("MongoDB connection closed")

# Alias for backward compatibility
Retriever = MongoDBRetriever
