from collections import OrderedDict
from threading import RLock
from typing import List, Dict, Any
import os
import logging
from pymongo import MongoClient
from openai import OpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

load_dotenv()

logger = logging.getLogger("mcq.retriever")


class MongoDBRetriever:
    """
    MongoDB-based retriever using:
    1. Atlas Vector Search (if available)
    2. Text search fallback
    3. Local embedding model via sentence-transformers
    """

    def __init__(
        self,
        embedding_base_url: str = None,
        embedding_model: str = None,
        use_local_embedding: bool = None,
    ):
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
        self.tier1_mongo_uri = os.getenv("TIER1_MONGO_URI")
        self.tier1_db_name = os.getenv("TIER1_MONGO_DB_NAME", "gtrinh")
        self.tier1_collection_name = os.getenv("TIER1_MONGO_COLLECTION_NAME", "gtrinh")

        if not self.mongo_uri:
            print("Warning: MONGO_URI not set. Retriever will fail if used.")

        self.collection = None
        self.tier1_client = None
        self.tier1_collection = None
        if self.mongo_uri:
            self.client = MongoClient(self.mongo_uri)
            self.db = self.client[self.db_name]
            self.collection = self.db[self.collection_name]
            print(f"✓ Connected to MongoDB: {self.db_name}.{self.collection_name}")

        if self.tier1_mongo_uri:
            self.tier1_client = MongoClient(self.tier1_mongo_uri)
            tier1_db = self.tier1_client[self.tier1_db_name]
            self.tier1_collection = tier1_db[self.tier1_collection_name]
            print(
                f"✓ Connected to Tier 1 MongoDB: "
                f"{self.tier1_db_name}.{self.tier1_collection_name}"
            )
        else:
            logger.warning(
                "TIER1_MONGO_URI is not set; retrieval will use Tier 2 only."
            )

        # Embedding configuration
        if use_local_embedding is None:
            self.use_local_embedding = os.getenv(
                "USE_LOCAL_EMBEDDING", "true"
            ).lower() in ("1", "true", "yes")
        else:
            self.use_local_embedding = use_local_embedding

        self.embedding_base_url = embedding_base_url or os.getenv(
            "EMBEDDING_BASE_URL", "http://127.0.0.1:1234/v1"
        )
        self.embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-m3"
        )
        self.embedding_device = os.getenv("EMBEDDING_DEVICE", "auto").strip()

        self.local_model = None
        self.embedding_client = None
        self.cache_size = max(0, int(os.getenv("RETRIEVER_CACHE_SIZE", "512")))
        self._cache_lock = RLock()
        self._embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
        self._search_cache: OrderedDict[tuple[str, int], List[Document]] = OrderedDict()
        self._dsm5_search_cache: OrderedDict[tuple[str, int], List[Document]] = OrderedDict()

        if self.use_local_embedding:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers not installed. Please install it or set USE_LOCAL_EMBEDDING=false"
                )
            model_kwargs: Dict[str, str] = {}
            if self.embedding_device and self.embedding_device.lower() != "auto":
                model_kwargs["device"] = self.embedding_device
            print(
                f"✓ Initializing Local Embedding Model: {self.embedding_model} "
                f"(device={self.embedding_device or 'auto'})"
            )
            self.local_model = SentenceTransformer(
                self.embedding_model, trust_remote_code=True, **model_kwargs
            )
            print("✓ Local model loaded successfully.")
        else:
            self.embedding_client = OpenAI(
                base_url=self.embedding_base_url, api_key="dummy"
            )
            print(
                f"✓ Using API embedding model: {self.embedding_model} at {self.embedding_base_url}"
            )

    @staticmethod
    def _cache_key(text: str) -> str:
        """Normalize semantically identical whitespace-only query variants."""
        return " ".join(text.lower().split())

    def _cache_get(self, cache: OrderedDict, key: Any) -> Any:
        if not self.cache_size:
            return None
        with self._cache_lock:
            value = cache.get(key)
            if value is not None:
                cache.move_to_end(key)
            return value

    def _cache_set(self, cache: OrderedDict, key: Any, value: Any) -> None:
        if not self.cache_size:
            return
        with self._cache_lock:
            cache[key] = value
            cache.move_to_end(key)
            while len(cache) > self.cache_size:
                cache.popitem(last=False)

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector using local model or API."""
        try:
            if self.use_local_embedding and self.local_model:
                vector = self.local_model.encode(text, normalize_embeddings=True)
                return vector.tolist()

            response = self.embedding_client.embeddings.create(
                model=self.embedding_model, input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Warning: Embedding generation failed: {e}")
            # Fallback size for common embedding dims in this pipeline
            return [0.0] * 1024

    def generate_embedding(self, text: str) -> List[float]:
        """
        Public method to generate embedding for a given text.
        Same as _generate_embedding but accessible from outside.
        """
        key = self._cache_key(text)
        cached = self._cache_get(self._embedding_cache, key)
        if cached is not None:
            return list(cached)
        vector = self._generate_embedding(text)
        self._cache_set(self._embedding_cache, key, list(vector))
        return vector

    def add_documents(self, chunks: List[Dict[str, Any]]):
        """
        Add documents to MongoDB with embeddings.

        Note: In production, you might want to batch this and run async.
        For now, we assume documents are already in MongoDB.
        """
        print(f"MongoDB retriever uses existing collection: {self.collection_name}")
        print(f"Total documents in collection: {self.collection.count_documents({})}")

    def search_dsm5(self, query_embedding: List[float], k: int = 5) -> List[Document]:
        """
        Search DSM-5 collection using vector search with provided embedding.
        Uses only embedding search (vector_index).

        Args:
            query_embedding: Pre-computed embedding vector
            k: Number of results to return (default 5)

        Returns:
            List of LangChain Document objects from DSM-5 collection
        """
        if self.collection is None:
            print("Error: MongoDB connection not initialized. Cannot search DSM-5.")
            return []

        dsm5_collection_name = os.getenv("MONGO_DSM5_COLLECTION_NAME", "DSM-5")
        dsm5_collection = self.db[dsm5_collection_name]

        try:
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": "vector_index",  # Same index name as main collection
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,
                        "limit": k,
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
                        "chunk_id": 1,
                        "tier": 1,
                        "source_tier": 1,
                        "type": 1,
                        "disease_name": 1,
                        "code": 1,
                        "differential_diagnosis": 1,
                        "_id": 1,
                        "score": {"$meta": "vectorSearchScore"},
                    }
                },
            ]

            results = list(dsm5_collection.aggregate(pipeline))
            return self._format_results(results)
        except Exception as e:
            print(f"DSM-5 vector search failed: {e}")
            return []

    def search_dsm5_by_query(self, query: str, k: int = 5) -> List[Document]:
        """Cache DSM-5 retrieval by query while reusing the embedding cache."""
        key = (self._cache_key(query), k)
        cached = self._cache_get(self._dsm5_search_cache, key)
        if cached is not None:
            return list(cached)
        results = self.search_dsm5(self.generate_embedding(query), k=k)
        self._cache_set(self._dsm5_search_cache, key, list(results))
        return results

    def search(
        self, query: str, k: int = 5, *, tier1_k: int | None = None
    ) -> List[Document]:
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

        # Tier-1 depth changes the Tier-2 expansion query, so it must be part
        # of the cache key; otherwise a hard request could reuse a shallow
        # easy/medium retrieval result for the same query.
        cache_key = (self._cache_key(query), k, tier1_k)
        cached = self._cache_get(self._search_cache, cache_key)
        if cached is not None:
            return list(cached)

        if getattr(self, "tier1_collection", None) is not None:
            results = self.search_tiered(query, k=k, tier1_k=tier1_k)
        else:
            results = self._search_tier2(query, k)
        self._cache_set(self._search_cache, cache_key, list(results))
        return results

    def _search_tier2(self, query: str, k: int) -> List[Document]:
        """Search the existing mental collection using its normal fallbacks."""
        try:
            # Strategy 1: Try Atlas Vector Search (requires atlas_vector_search index)
            results = self._vector_search(query, k)
        except Exception as e:
            print(f"Vector search failed: {e}, falling back to text search")
            try:
                # Strategy 2: MongoDB text search (requires text index)
                results = self._text_search(query, k)
            except Exception as e2:
                print(f"Text search failed: {e2}, falling back to keyword search")
                # Strategy 3: Simple keyword search
                results = self._keyword_search(query, k)
        return results

    @staticmethod
    def _tier1_query_context(documents: List[Document]) -> str:
        """Build a bounded Tier 1-derived query expansion for Tier 2 retrieval."""
        per_chunk = max(120, int(os.getenv("TIER1_QUERY_CONTEXT_CHARS", "900")))
        return "\n".join(document.page_content[:per_chunk] for document in documents)

    @staticmethod
    def _deduplicate_documents(documents: List[Document]) -> List[Document]:
        unique: List[Document] = []
        seen = set()
        for document in documents:
            metadata = document.metadata or {}
            key = str(metadata.get("chunk_id") or metadata.get("uuid") or document.page_content)
            if key not in seen:
                seen.add(key)
                unique.append(document)
        return unique

    @staticmethod
    def _set_source_tier(documents: List[Document], tier: str) -> List[Document]:
        for document in documents:
            document.metadata["tier"] = tier
            document.metadata["source_tier"] = tier
        return documents

    def search_tiered(
        self, query: str, k: int = 5, *, tier1_k: int | None = None
    ) -> List[Document]:
        """Retrieve Tier 1 textbooks first, then retrieve related Tier 2 evidence.

        Tier 2 is never queried from the original question alone when Tier 1
        material is available: the selected textbook passages expand the query
        so secondary material is anchored to the primary source.
        """
        requested_tier1_k = (
            tier1_k
            if tier1_k is not None
            else int(os.getenv("TIER1_RETRIEVAL_K", "2"))
        )
        tier1_k = min(k, max(1, requested_tier1_k))
        tier1_docs = self._search_collection(
            self.tier1_collection,
            query,
            k=tier1_k,
            vector_index=os.getenv("TIER1_MONGO_VECTOR_INDEX", "vector_index"),
            text_index=os.getenv("TIER1_MONGO_TEXT_INDEX", "atlas_index"),
            source_tier="tier_1",
        )
        tier1_docs = self._set_source_tier(tier1_docs, "tier_1")
        related_query = query
        if tier1_docs:
            related_query = f"{query}\n\nGiáo trình liên quan:\n{self._tier1_query_context(tier1_docs)}"
        else:
            logger.warning("Tier 1 retrieval returned no documents; using original query for Tier 2")

        tier2_k = max(0, k - len(tier1_docs))
        tier2_docs = self._set_source_tier(
            self._search_tier2(related_query, tier2_k) if tier2_k else [], "tier_2"
        )
        results = self._deduplicate_documents([*tier1_docs, *tier2_docs])[:k]
        logger.info(
            "Tiered retrieval: requested_tier1=%s tier1=%s tier2=%s returned=%s",
            tier1_k,
            len(tier1_docs), len(tier2_docs), len(results),
        )
        return results

    def _search_collection(
        self,
        collection: Any,
        query: str,
        *,
        k: int,
        vector_index: str,
        text_index: str,
        source_tier: str,
    ) -> List[Document]:
        """Search another Mongo collection without changing the Tier 2 client."""
        if collection is None or k < 1:
            return []
        projection = {
            "content": 1, "text": 1, "page_content": 1, "body": 1,
            "title": 1, "headers": 1, "summary": 1, "tags": 1, "keywords": 1,
            "uuid": 1, "chunk_id": 1, "type": 1, "source": 1,
            "source_sha256": 1, "chunk_index": 1, "start_char": 1,
            "end_char": 1, "source_char_count": 1, "structure_path": 1, "_id": 1,
            "score": {"$meta": "vectorSearchScore"},
        }
        try:
            vector_pipeline = [
                {"$vectorSearch": {
                    "index": vector_index, "path": "embedding",
                    "queryVector": self.generate_embedding(query),
                    "numCandidates": k * 10, "limit": k,
                }},
                {"$project": projection},
            ]
            return self._format_results(list(collection.aggregate(vector_pipeline)), force_tier=source_tier)
        except Exception as vector_error:
            logger.info("Tier 1 vector search unavailable (%s); trying text search", type(vector_error).__name__)
        try:
            text_pipeline = [
                {"$search": {"index": text_index, "text": {"query": query, "path": ["content", "text", "title", "summary", "keywords"]}}},
                {"$limit": k},
                {"$project": {**projection, "score": {"$meta": "searchScore"}}},
            ]
            return self._format_results(list(collection.aggregate(text_pipeline)), force_tier=source_tier)
        except Exception as text_error:
            logger.info("Tier 1 Atlas text search unavailable (%s); trying keyword search", type(text_error).__name__)
        keywords = [keyword for keyword in query.split() if len(keyword) > 1]
        if not keywords:
            return []
        patterns = [{"$regex": keyword, "$options": "i"} for keyword in keywords]
        results = collection.find(
            {"$or": [
                {"content": {"$in": patterns}}, {"text": {"$in": patterns}},
                {"title": {"$in": patterns}}, {"summary": {"$in": patterns}},
                {"keywords": {"$in": patterns}},
            ]}
        ).limit(k)
        return self._format_results(list(results), force_tier=source_tier)

    def _vector_search(self, query: str, k: int) -> List[Document]:
        """
        Perform Atlas Vector Search.

        Requires: Vector search index on 'embedding' field in Atlas
        """
        if self.collection is None:
            return []

        query_embedding = self.generate_embedding(query)

        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",  # User specified index name
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": k * 10,
                    "limit": k,
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
                    "chunk_id": 1,
                    "tier": 1,
                    "source_tier": 1,
                    "type": 1,
                    "disease_name": 1,
                    "code": 1,
                    "differential_diagnosis": 1,
                    "_id": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
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
                        "path": ["content", "title", "summary", "keywords"],
                    },
                }
            },
            {"$limit": k},
            {
                "$project": {
                    "content": 1,
                    "title": 1,
                    "summary": 1,
                    "tags": 1,
                    "keywords": 1,
                    "uuid": 1,
                    "chunk_id": 1,
                    "tier": 1,
                    "source_tier": 1,
                    "type": 1,
                    "disease_name": 1,
                    "code": 1,
                    "differential_diagnosis": 1,
                    "_id": 1,
                    "score": {"$meta": "searchScore"},
                }
            },
        ]

        try:
            results = list(self.collection.aggregate(pipeline))
            return self._format_results(results)
        except Exception as e:
            print(
                f"Atlas text search failed: {e}. Trying standard text search fallback."
            )
            # Fallback to standard Mongo $text search if Atlas Search fails (e.g. index not found)
            return self._standard_text_search(query, k)

    def _standard_text_search(self, query: str, k: int) -> List[Document]:
        """
        Standard MongoDB text search ($text).
        """
        results = (
            self.collection.find(
                {"$text": {"$search": query}}, {"score": {"$meta": "textScore"}}
            )
            .sort([("score", {"$meta": "textScore"})])
            .limit(k)
        )

        return self._format_results(list(results))

    def _keyword_search(self, query: str, k: int) -> List[Document]:
        """
        Simple keyword-based search using regex.
        """
        # Split query into keywords
        keywords = query.lower().split()

        # Search in content, summary, and keywords fields
        regex_patterns = [{"$regex": kw, "$options": "i"} for kw in keywords]

        results = self.collection.find(
            {
                "$or": [
                    {"content": {"$in": regex_patterns}},
                    {"title": {"$in": regex_patterns}},
                    {"summary": {"$in": regex_patterns}},
                    {"keywords": {"$in": regex_patterns}},
                ]
            }
        ).limit(k)

        return self._format_results(list(results))

    def _format_results(
        self, results: List[Dict], *, force_tier: str | None = None
    ) -> List[Document]:
        """Convert MongoDB results to LangChain Documents."""
        documents = []

        for result in results:
            # Handle both regular and DSM-5 collection formats
            disease_name = result.get("disease_name")
            code = result.get("code")
            differential_diagnosis = result.get("differential_diagnosis", [])

            headers = result.get("headers", [])
            fallback_title = " / ".join(headers) if isinstance(headers, list) else str(headers or "")
            source = str(result.get("source") or "")
            title = result.get("title") or disease_name or fallback_title or source
            summary = result.get("summary", "")
            content = (
                result.get("content")
                or result.get("text")
                or result.get("page_content")
                or result.get("body")
                or ""
            )

            # Format page_content with available fields
            if disease_name:
                # DSM-5 format
                dsm5_info = f"Bệnh: {disease_name}"
                if code:
                    dsm5_info += f"\nMã ICD: {code}"
                if differential_diagnosis:
                    if isinstance(differential_diagnosis, list):
                        dsm5_info += f"\nChẩn đoán phân biệt: {', '.join(differential_diagnosis)}"
                    else:
                        dsm5_info += f"\nChẩn đoán phân biệt: {differential_diagnosis}"
                page_content = f"{dsm5_info}\n\nNội dung:\n{content}"
            else:
                # Regular format
                page_content = (
                    f"Tiêu đề: {title}\nTóm tắt: {summary}\nNội dung:\n{content}"
                )

            metadata = {
                "uuid": result.get("uuid", result.get("_id", "")),
                "chunk_id": result.get(
                    "chunk_id", result.get("uuid", result.get("_id", ""))
                ),
                "tier": force_tier or result.get("tier", result.get("source_tier", "")),
                "source_tier": force_tier or result.get("source_tier", result.get("tier", "")),
                "title": title or disease_name or "",
                "summary": summary or "",
                "tags": result.get("tags", []),
                "keywords": result.get("keywords", []),
                "type": result.get("type", "DSM-5" if disease_name else ""),
                "score": result.get("score", 0.0),
                "code": code or "",
                "disease_name": disease_name or "",
                "source": source,
                "source_sha256": result.get("source_sha256", ""),
                "chunk_index": result.get("chunk_index"),
                "start_char": result.get("start_char"),
                "end_char": result.get("end_char"),
                "source_char_count": result.get("source_char_count"),
                "structure_path": result.get("structure_path", []),
            }

            documents.append(Document(page_content=page_content, metadata=metadata))

        return documents

    def close(self):
        """Close MongoDB connection."""
        if hasattr(self, "client") and self.client:
            self.client.close()
            print("MongoDB connection closed")
        if getattr(self, "tier1_client", None):
            self.tier1_client.close()
            print("Tier 1 MongoDB connection closed")


# Alias for backward compatibility
Retriever = MongoDBRetriever
