from typing import List, Dict, Any
import os
from pymongo import MongoClient
from openai import OpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

load_dotenv()

class MongoDBRetriever:
    """
    MongoDB-based retriever using:
    1. Atlas Vector Search (if available)
    2. Text search fallback
    3. Local embedding model via sentence-transformers
    """
    
    def __init__(self, 
                 embedding_base_url: str = None,
                 embedding_model: str = None,
                 use_local_embedding: bool = None):
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
            print("Warning: MONGO_URI not set. Retriever will fail if used.")

        self.collection = None
        if self.mongo_uri:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            print(f"✓ Connected to MongoDB: {self.db_name}.{self.collection_name}")

        # Embedding configuration
        if use_local_embedding is None:
            self.use_local_embedding = os.getenv("USE_LOCAL_EMBEDDING", "true").lower() in ("1", "true", "yes")
        else:
            self.use_local_embedding = use_local_embedding

        self.embedding_base_url = embedding_base_url or os.getenv("EMBEDDING_BASE_URL", "http://127.0.0.1:1234/v1")
        self.embedding_model = embedding_model or os.getenv("EMBEDDING_MODEL", "namdp-ptit/ViDense")

        self.local_model = None
        self.embedding_client = None

        if self.use_local_embedding:
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not installed. Please install it or set USE_LOCAL_EMBEDDING=false")
            print(f"✓ Initializing Local Embedding Model: {self.embedding_model}")
            self.local_model = SentenceTransformer(self.embedding_model, trust_remote_code=True)
            print("✓ Local model loaded successfully.")
        else:
            self.embedding_client = OpenAI(
                base_url=self.embedding_base_url,
                api_key="dummy"
            )
            print(f"✓ Using API embedding model: {self.embedding_model} at {self.embedding_base_url}")
        
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector using local model or API."""
        try:
            if self.use_local_embedding and self.local_model:
                vector = self.local_model.encode(text, normalize_embeddings=True)
                return vector.tolist()

            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Warning: Embedding generation failed: {e}")
            # Fallback size for common embedding dims in this pipeline
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
        if self.collection is None:
            print("Error: MongoDB collection not initialized. Cannot search.")
            return []

        if self.use_local_embedding and self.local_model is None:
            print("Error: Local embedding model not initialized. Cannot search.")
            return []

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
        if self.collection is None:
            return []

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
                    "title": 1,
                    "summary": 1,
                    "tags": 1,
                    "keywords": 1,
                    "uuid": 1,
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
                        "path": ["content", "title", "summary", "keywords"]
                    }
                }
            },
            {
                "$limit": k
            },
            {
                "$project": {
                    "content": 1,
                    "title": 1,
                    "summary": 1,
                    "tags": 1,
                    "keywords": 1,
                    "uuid": 1,
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
                {"title": {"$in": regex_patterns}},
                {"summary": {"$in": regex_patterns}},
                {"keywords": {"$in": regex_patterns}}
            ]
        }).limit(k)
        
        return self._format_results(list(results))
    
    def _format_results(self, results: List[Dict]) -> List[Document]:
        """Convert MongoDB results to LangChain Documents."""
        documents = []
        
        for result in results:
            title = result.get('title', '')
            summary = result.get('summary', '')
            content = result.get('content', '')

            page_content = f"Tiêu đề: {title}\nTóm tắt: {summary}\nNội dung:\n{content}"
            
            metadata = {
                'uuid': result.get('uuid'),
                'title': title,
                'summary': summary,
                'tags': result.get('tags', []),
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
        if hasattr(self, 'client') and self.client:
            self.client.close()
            print("MongoDB connection closed")

# Alias for backward compatibility
Retriever = MongoDBRetriever
