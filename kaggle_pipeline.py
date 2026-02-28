
!uv pip install pymongo openai langchain-core langchain-openai langgraph python-dotenv numpy sentence-transformers vllm

import os
import sys
import time
import subprocess
import requests
import json
import random
import re
import builtins
from typing import List, Dict, Any, TypedDict, Optional
# from dotenv import load_dotenv # Removed
from pymongo import MongoClient
import numpy as np

def start_vllm_background():
    """
    Start vLLM server in a background subprocess and wait for it to be ready.
    """
    cmd = Config.VLLM_CMD
    print(f"Starting vLLM with command: {cmd}")
    
    # Start process
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    print(f"vLLM process started with PID: {process.pid}")
    
    # Wait for server to be ready
    api_url = "http://localhost:8000/v1/models"
    print("Waiting for vLLM server to be ready...")
    
    max_retries = 60 # Wait up to 10 minutes (60 * 10s)
    for i in range(max_retries):
        try:
            response = requests.get(api_url, timeout=5)
            if response.status_code == 200:
                print("\n✓ vLLM Server is READY!")
                return process
        except requests.exceptions.ConnectionError:
            pass
        
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(10)
        
        # Check if process died
        if process.poll() is not None:
            stderr_output = process.stderr.read().decode('utf-8')
            print(f"\nERROR: vLLM process exited unexpectedly with code {process.returncode}")
            print(f"STDERR:\n{stderr_output}")
            sys.exit(1)

    print("\nTimeout waiting for vLLM server.")
    sys.exit(1)

def check_and_raise_503(e):
    """
    Check if the exception is a 503 Service Unavailable error.
    If so, print a critical error message and exit the script.
    """
    error_msg = str(e)
    if "503" in error_msg:
        print("\n" + "="*50)
        print("CRITICAL ERROR: 503 Service Unavailable detected.")
        print("The server is overloaded or down. Stopping pipeline immediately.")
        print("="*50 + "\n")
        sys.exit(1)

# LangChain / LangGraph imports
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

# OpenAI
from openai import OpenAI

# Sentence Transformers (for local usage)
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# ==========================================
# 0. Configuration
# ==========================================

class Config:
    # MongoDB Configuration
    MONGO_URI = "mongodb+srv://nguyenphamtrunghieu277_db_user:hieudevdut@cluster0.91ny3vc.mongodb.net/"
    MONGO_DB_NAME = "thuvien"
    MONGO_COLLECTION_NAME = "thuvienphapluat"
    
    # OpenAI / LLM Configuration
    USE_LOCAL_VLLM = True # Set to True to use local vLLM
    
    # Note: If USE_LOCAL_VLLM is True, OPENAI_BASE_URL will be overwritten to localhost
    OPENAI_BASE_URL = "https://silver-pots-jog.loca.lt/v1" 
    OPENAI_API_KEY = "EMPTY"
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    # vLLM Command (for local deployment)
    VLLM_CMD = 'uv run vllm serve "Qwen/Qwen3-30B-A3B-Instruct-2507" --dtype auto --gpu-memory-utilization 0.85 --max-model-len 12000 --host 0.0.0.0 --port 8000 --trust-remote-code'
    
    # Embedding Configuration
    USE_LOCAL_EMBEDDING = True
    EMBEDDING_BASE_URL = "http://127.0.0.1:1234/v1" # Fallback if local is False
    # Using a valid Hugging Face model ID for sentence-transformers
    EMBEDDING_MODEL = "namdp-ptit/ViDense" 
    
    # Pipeline Settings
    START_INDEX = 15068
    END_INDEX = 0
    NUM_QA_PAIRS = 5000

# Global context for retriever
builtins.RETRIEVER = None

# ==========================================
# 1. State Definition (src/graph/state.py)
# ==========================================

class AgentState(TypedDict):
    # Input Data
    # all_chunks: List[Dict[str, Any]]  # Removed to optimize memory - use direct DB query
    used_anchor_ids: List[str]  # IDs of chunks already used as anchors
    
    # Current Iteration State
    iteration_count: int        # Number of QA pairs generated so far
    max_iterations: int         # Maximum QA pairs to generate
    
    # Flow Selection
    is_reasoning_flow: bool     # True = Reasoning QA, False = Simple QA
    
    # Anchor and Context
    anchor: Optional[Dict[str, Any]]      # Randomly selected anchor
    query: str                  # Query derived from anchor
    context_docs: List[Document]# Retrieved documents (3-5 related/opposing)
    
    # Outputs
    simple_qa: Optional[Dict[str, str]]     # Normal QA output
    reasoning_qa: Optional[Dict[str, str]]  # Reasoning QA output
    
    # Reasoning Step-by-Step Verification (New)
    reasoning_raw_output: str   # Raw LLM output with <think> tags
    reasoning_steps: List[str]  # Parsed individual steps
    current_step_index: int     # Current step being verified (0-indexed)
    step_retry_count: int       # Number of retries for current step
    step_verification_results: List[bool]  # Per-step verification results
    
    # Logs and Flags
    verification_passed: bool   # Overall verification flag
    reasoning_logs: List[str]   # Logs of verification/refinement
    format_check_passed: bool   # Flag for output format validation
    final_output_ready: bool    # Ready to output
    
    # Output Collection
    all_outputs: List[Dict[str, Any]]  # Collected QA pairs across iterations
    last_saved_count: int               # Index of the last saved QA pair


# ==========================================
# 2. Retriever Definition (src/retriever.py)
# ==========================================

class MongoDBRetriever:
    """
    MongoDB-based retriever using:
    1. Atlas Vector Search (if available)
    2. Text search fallback
    3. Local embedding model via OpenAI API
    """
    
    def __init__(self):
        """
        Initialize MongoDB retriever with local embedding model.
        """
        # MongoDB connection
        self.mongo_uri = Config.MONGO_URI
        self.db_name = Config.MONGO_DB_NAME
        self.collection_name = Config.MONGO_COLLECTION_NAME
        
        if not self.mongo_uri:
            print("Warning: MONGO_URI not set in Config. Retriever will fail if used.")
        
        if self.mongo_uri:
            try:
                self.client = MongoClient(self.mongo_uri)
                self.db = self.client[self.db_name]
                self.collection = self.db[self.collection_name]
                print(f"✓ Connected to MongoDB: {self.db_name}.{self.collection_name}")
            except Exception as e:
                print(f"Error connecting to MongoDB: {e}")
                self.collection = None
        
        # Embedding client configuration
        self.use_local = Config.USE_LOCAL_EMBEDDING
        self.embedding_base_url = Config.EMBEDDING_BASE_URL
        self.embedding_model = Config.EMBEDDING_MODEL
        
        if self.use_local:
            print(f"✓ Initializing Local Embedding Model: {self.embedding_model}")
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not installed. Please pip install it.")
                
            # Initialize local model
            try:
                self.local_model = SentenceTransformer(self.embedding_model, trust_remote_code=True)
                print("✓ Local model loaded successfully.")
            except Exception as e:
                print(f"Error loading local model: {e}")
                self.local_model = None
        else:
            self.local_model = None
            self.embedding_client = OpenAI(
                base_url=self.embedding_base_url,
                api_key="dummy"
            )
            print(f"✓ Using API embedding model: {self.embedding_model} at {self.embedding_base_url}")
        
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector using local OpenAI-compatible API or Local HF Model."""
        try:
            if self.use_local and self.local_model:
                # Use local sentence-transformers model
                # encode returns ndarray, convert to list
                vector = self.local_model.encode(text, normalize_embeddings=True)
                return vector.tolist()
            else:
                # Use OpenAI API compatible endpoint
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return response.data[0].embedding
        except Exception as e:
            print(f"Warning: Embedding generation failed: {e}")
            # Return zero vector as fallback
            return [0.0] * 1024
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search MongoDB for relevant documents.
        """
        if not hasattr(self, 'collection') or self.collection is None:
            print("Error: MongoDB collection not initialized. Cannot search.")
            return []
            
        if self.use_local and self.local_model is None:
             print("Error: Local model failed to load. Cannot search.")
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
        """
        if not hasattr(self, 'collection') or self.collection is None:
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
                    "noi_dung": 1,
                    "title": 1,
                    "tom_tat": 1,
                    "so_hieu": 1,
                    "loai_van_ban": 1,
                    "ngay_ban_hanh": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(self.collection.aggregate(pipeline))
        return self._format_results(results)
    
    def _text_search(self, query: str, k: int) -> List[Document]:
        """
        Fallback to Mongo Atlas Text Search.
        """
        if not hasattr(self, 'collection') or self.collection is None:
            return []

        pipeline = [
            {
                "$search": {
                    "index": "atlas_index",  # User specified index name
                    "text": {
                        "query": query,
                        "path": ["noi_dung", "title", "tom_tat", "so_hieu"]
                    }
                }
            },
            {
                "$limit": k
            },
            {
                "$project": {
                    "noi_dung": 1,
                    "title": 1,
                    "tom_tat": 1,
                    "so_hieu": 1,
                    "loai_van_ban": 1,
                    "ngay_ban_hanh": 1,
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
            return self._standard_text_search(query, k)

    def _standard_text_search(self, query: str, k: int) -> List[Document]:
        """
        Standard MongoDB text search ($text).
        """
        if not hasattr(self, 'collection') or self.collection is None:
            return []

        results = self.collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(k)
        
        return self._format_results(list(results))
    
    def _keyword_search(self, query: str, k: int) -> List[Document]:
        """
        Simple keyword-based search using regex.
        """
        if not hasattr(self, 'collection') or self.collection is None:
            return []

        # Split query into keywords
        keywords = query.lower().split()
        
        # Search in content, summary, and keywords fields
        regex_patterns = [{"$regex": kw, "$options": "i"} for kw in keywords]
        
        results = self.collection.find({
            "$or": [
                {"noi_dung": {"$in": regex_patterns}},
                {"title": {"$in": regex_patterns}},
                {"tom_tat": {"$in": regex_patterns}}
            ]
        }).limit(k)
        
        return self._format_results(list(results))
    
    def _format_results(self, results: List[Dict]) -> List[Document]:
        """Convert MongoDB results to LangChain Documents."""
        documents = []
        
        for result in results:
            # Construct page_content from legal fields
            title = result.get('title', '')
            so_hieu = result.get('so_hieu', '')
            loai_van_ban = result.get('loai_van_ban', '')
            noi_dung = result.get('noi_dung', '')
            
            # Combine into a structured string for the LLM
            page_content = f"Văn bản: {title}\nSố hiệu: {so_hieu}\nLoại: {loai_van_ban}\nNội dung:\n{noi_dung}"
            
            # Metadata
            metadata = {
                'uuid': result.get('uuid'),
                'title': title,
                'so_hieu': so_hieu,
                'loai_van_ban': loai_van_ban,
                'ngay_ban_hanh': result.get('ngay_ban_hanh'),
                'tom_tat': result.get('tom_tat'),
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


openai_client = OpenAI(
    base_url=Config.OPENAI_BASE_URL,
    api_key=Config.OPENAI_API_KEY
)

MODEL_NAME = Config.MODEL_NAME

def first_filter_node(state: AgentState) -> Dict[str, Any]:
    """
    First Filter: Randomly decide if this iteration generates Simple QA or Reasoning QA.
    """
    # 50/50 split or custom logic
    is_reasoning = state['is_reasoning_flow']
    return {"is_reasoning_flow": is_reasoning}

def select_anchor_node(state: AgentState) -> Dict[str, Any]:
    """
    Randomly select an anchor directly from MongoDB, excluding used IDs.
    """
    # Connect to MongoDB (quick connection, lightweight)
    mongo_uri = Config.MONGO_URI
    db_name = Config.MONGO_DB_NAME
    collection_name = Config.MONGO_COLLECTION_NAME
    
    if not mongo_uri:
        print("Error: MONGO_URI not set for anchor selection.")
        return {"anchor": None}

    try:
        client = MongoClient(mongo_uri)
        collection = client[db_name][collection_name]
    except Exception as e:
        print(f"Error connecting to MongoDB in select_anchor: {e}")
        return {"anchor": None}
    
    used_ids = state.get('used_anchor_ids', [])
    is_reasoning = state.get('is_reasoning_flow', False)
    
    if not is_reasoning:
        # Simple QA: Sequential selection based on global index range
        start_index = Config.START_INDEX
        end_index = Config.END_INDEX
        
        # Calculate which document to fetch: Start + current_iteration
        target_offset = start_index + state.get('iteration_count', 0)
        
        if end_index > 0 and target_offset >= end_index:
            print(f"Reached END_INDEX ({end_index}). Stop selection.")
            client.close()
            return {"anchor": None}

        print(f"Selecting Anchor (Sequential for SimpleQA) - Offset {target_offset}...")
        pipeline = [
            {"$sort": {"_id": 1}},
            {"$skip": target_offset},
            {"$limit": 1}
        ]
    else:
        # Reasoning QA: Random selection
        print("Selecting Anchor (Random for ReasoningQA)...")
        pipeline = [
            {"$match": {"uuid": {"$nin": used_ids}}},
            {"$sample": {"size": 1}}
        ]
    
    results = list(collection.aggregate(pipeline))
    client.close()
    
    if not results:
        # Fallback: if all used, clear used list and try again (or stop)
        print("Warning: All anchors used. Resetting cycle.")
        
        if not is_reasoning:
             pipeline_fallback = [
                {"$sort": {"_id": 1}},
                {"$skip": target_offset},
                {"$limit": 1}
            ]
        else:
             pipeline_fallback = [{"$sample": {"size": 1}}]
             
        # Reconnect for fallback
        client = MongoClient(mongo_uri)
        collection = client[db_name][collection_name]
        results = list(collection.aggregate(pipeline_fallback))
        client.close()
        
        if not results:
            print("Error: No data in MongoDB.")
            return {"anchor": None}

    anchor_doc = results[0]
    new_id = anchor_doc.get('uuid')
    
    print(f"\n[{state.get('iteration_count', 0) + 1}] Anchor Selected ({'Simple' if not is_reasoning else 'Reasoning'}): {anchor_doc.get('title', '')[:100]}...")

    # Map to metadata format expected by pipeline
    title = anchor_doc.get('title', '')
    so_hieu = anchor_doc.get('so_hieu', '')
    loai_van_ban = anchor_doc.get('loai_van_ban', '')
    
    metadata = {
        'uuid': anchor_doc.get('uuid'),
        'title': title,
        'so_hieu': so_hieu,
        'loai_van_ban': loai_van_ban,
        'ngay_ban_hanh': anchor_doc.get('ngay_ban_hanh'),
        'tom_tat': anchor_doc.get('tom_tat'),
        'type': anchor_doc.get('type')
    }
    
    updates = {
        "anchor": metadata,
        "query": f"{title} {so_hieu} {anchor_doc.get('tom_tat', '')}",
        "used_anchor_ids": list(set(used_ids) | {new_id})
    }
    
    # If Simple QA, we skip retrieval, so we must provide the anchor content as context
    if not is_reasoning:
        content = anchor_doc.get('noi_dung', '')
        page_content = f"Văn bản: {title}\nSố hiệu: {so_hieu}\nLoại: {loai_van_ban}\nNội dung:\n{content}"
        # Create a Document object for the anchor
        anchor_as_doc = Document(page_content=page_content, metadata=metadata)
        updates["context_docs"] = [anchor_as_doc]
        
    return updates

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves 3-5 related/opposing documents using BM25 + Vector Search.
    """
    retriever = getattr(builtins, 'RETRIEVER', None)
    
    if not retriever:
        print("Warning: Retriever not found.")
        return {"context_docs": []}

    query = state['query']
    print(f"Retrieving docs for query: {query[:50]}...")
    docs = retriever.search(query, k=random.randint(3, 5))
    print(f"Found {len(docs)} documents.")
    return {"context_docs": docs}

def generate_simple_qa_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates Normal QA without reasoning.
    """
    print("Generating Simple QA...")
    context_text = "\n\n".join([d.page_content for d in state['context_docs']])
    
    prompt = f"""Bạn là chuyên gia pháp luật Việt Nam. Dựa vào văn bản pháp luật sau, tạo một cặp Câu hỏi - Câu trả lời chính xác.
    
Ngữ cảnh:
{context_text[:3000]}

Văn bản: {state['anchor'].get('title', '')}
Số hiệu: {state['anchor'].get('so_hieu', '')}

Format JSON: {{"question": "...", "answer": "..."}}
"""
    
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=7000,
            timeout=120
        )
        
        result = response.choices[0].message.content
        qa = json.loads(result.replace("```json", "").replace("```", "").strip())
        
        # Save to output collection
        output_entry = {
            "type": "simple_qa",
            "anchor_id": state['anchor']['uuid'],
            "question": qa.get('question'),
            "answer": qa.get('answer')
        }
        
        return {
            "simple_qa": qa,
            "all_outputs": state.get('all_outputs', []) + [output_entry],
            "iteration_count": state['iteration_count'] + 1
        }
    except Exception as e:
        check_and_raise_503(e)
        print(f"Error in simple_qa: {e}")
        return {"iteration_count": state['iteration_count'] + 1}

def generate_reasoning_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates Reasoning QA with <think> and <step> tags.
    """
    if not state.get('anchor'):
        print("Error: No anchor found in state. Skipping generation.")
        return {
             "reasoning_raw_output": "",
             "current_step_index": 0,
             "step_retry_count": 0,
             "step_verification_results": []
        }

    print("Generating Reasoning QA (this may take a while)...")
    context_text = "\n\n".join([d.page_content for d in state.get('context_docs', [])])
    
    prompt = f"""Bạn là luật sư/chuyên gia pháp luật uy tín với khả năng suy luận pháp lý chặt chẽ.

Ngữ cảnh (Các văn bản pháp luật liên quan):
{context_text}

Văn bản chính: {state['anchor'].get('title', '')}
Số hiệu: {state['anchor'].get('so_hieu', '')}

Nhiệm vụ:
1. Tạo một câu hỏi pháp lý phức tạp, tình huống cụ thể hoặc yêu cầu so sánh/áp dụng luật.
2. Suy nghĩ từng bước trong thẻ <think>. Mỗi bước suy luận đặt trong thẻ <step>.
   - Phân tích căn cứ pháp lý.
   - Trích dẫn điều luật (nếu có trong ngữ cảnh).
   - Suy luận logic từ quy định đến kết luận.
3. Đưa ra câu trả lời cuối cùng trong <answer>

Ví dụ format bắt buộc (bạn PHẢI tuân thủ cấu trúc này):
Question: [Câu hỏi pháp lý]
<think>
<step> ...</step>
<step> ...</step>
<step> ..</step>
</think>
<answer>Câu trả lời hoàn chỉnh và chính xác</answer>

LƯU Ý QUAN TRỌNG:
- BẮT BUỘC dùng thẻ <think>, <step>, <answer>.
- KHÔNG được bỏ qua bất kỳ thẻ nào.
- Nội dung trong <step> phải là suy nghĩ chi tiết.
"""
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4096,
            timeout=300
        )
        
        result = response.choices[0].message.content
        print(f"RAW OUTPUT:\n{result}\n-------------------")
        
        return {
            "reasoning_raw_output": result,
            "current_step_index": 0,
            "step_retry_count": 0,
            "step_verification_results": []
        }
    except Exception as e:
        check_and_raise_503(e)
        print(f"Error in generate_reasoning_node: {e}")
        return {
             "reasoning_raw_output": "",
             "current_step_index": 0,
             "step_retry_count": 0,
             "step_verification_results": []
        }

def parse_steps_node(state: AgentState) -> Dict[str, Any]:
    """
    Parse individual <step> tags from the reasoning output.
    """
    raw = state['reasoning_raw_output']
    
    # Extract steps
    step_pattern = r'<step>(.*?)</step>'
    steps = re.findall(step_pattern, raw, re.DOTALL)
    
    if not steps:
        print("Warning: No <step> tags found. Treating as single step.")
        # Fallback: treat as one step
        steps = [raw]
    
    print(f"Parsed {len(steps)} steps for verification.")
    
    return {
        "reasoning_steps": steps,
        "step_verification_results": [False] * len(steps)  # Initialize all as not verified
    }

def verify_single_step_node(state: AgentState) -> Dict[str, Any]:
    """
    Verify the current step only.
    """
    current_idx = state['current_step_index']
    if current_idx >= len(state['reasoning_steps']):
        return {"verification_passed": True}  # All done
    
    current_step = state['reasoning_steps'][current_idx]
    print(f"Verifying Step {current_idx}...")
    
    prompt = f"""Kiểm tra logic của bước suy luận sau:

{current_step}

Nếu bước này ĐÚNG logic, trả lời "ĐÚNG".
Nếu SAI, giải thích ngắn gọn lỗi.
"""
    
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,
            timeout=120
        )
        
        review = response.choices[0].message.content
        is_valid = "ĐÚNG" in review.upper() or "VALID" in review.upper()
        print(f"  > Verification Result: {'PASSED' if is_valid else 'FAILED'}")
        
        # Update verification result for this step
        results = state['step_verification_results'].copy()
        results[current_idx] = is_valid
        
        return {
            "step_verification_results": results,
            "reasoning_logs": state.get('reasoning_logs', []) + [f"Step {current_idx}: {review}"]
        }
    except Exception as e:
        check_and_raise_503(e)
        print(f"Error in verify_single_step_node: {e}")
        # Assuming pass on error to keep pipeline moving or fail. Let's assume fail.
        results = state['step_verification_results'].copy()
        results[current_idx] = False
        return {
            "step_verification_results": results,
             "reasoning_logs": state.get('reasoning_logs', []) + [f"Step {current_idx}: Error during verification - {e}"]
        }

def refine_single_step_node(state: AgentState) -> Dict[str, Any]:
    """
    Refine the current step that failed verification.
    """
    current_idx = state['current_step_index']
    print(f"  > Refining Step {current_idx} (Attempt {state.get('step_retry_count', 0) + 1})...")
    current_step = state['reasoning_steps'][current_idx]
    feedback = state['reasoning_logs'][-1]
    
    prompt = f"""Bước suy luận sau có lỗi:

{current_step}

Phản hồi: {feedback}

Viết lại bước này cho ĐÚNG.
"""
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800
        )
        
        refined = response.choices[0].message.content
        
        # Update the step
        steps = state['reasoning_steps'].copy()
        steps[current_idx] = refined
        
        return {
            "reasoning_steps": steps,
            "step_retry_count": state.get('step_retry_count', 0) + 1
        }
    except Exception as e:
        check_and_raise_503(e)
        print(f"Error in refine_single_step_node: {e}")
        return {"step_retry_count": state.get('step_retry_count', 0) + 1}

def increment_step_node(state: AgentState) -> Dict[str, Any]:
    """
    Increment step index to move to the next reasoning step.
    """
    return {
        "current_step_index": state['current_step_index'] + 1,
        "step_retry_count": 0
    }

def check_more_questions_node(state: AgentState) -> Dict[str, Any]:
    """
    After completing one QA, check if more iterations needed.
    Also moves to next step index if in reasoning flow.
    """
    # If in reasoning flow and current step verified, move to next
    if state.get('is_reasoning_flow') and state['current_step_index'] < len(state['reasoning_steps']):
        # Check if current step is verified OR max retries reached
        # Actually logic is handled by route: we only reach here if step is done.
        # But we need to check if we are at the last step to save.
        
        # Save the reasoning QA if we are at the last step
        # Note: 'current_step_index' is technically pointing to the *just verified* step here, 
        # but the logic in route increments it? No, route logic sends to 'check_more' when 
        # current_idx + 1 == len. So current_idx is the last step index.
        
        if state['current_step_index'] == len(state['reasoning_steps']) - 1:
             # Construct final reasoning QA
            question_match = re.search(r'Question:\s*(.*?)(?=<think>|<step>|<tool_call>|$)', state['reasoning_raw_output'], re.DOTALL)
            answer_match = re.search(r'(?:<answer>|\(answer\)|Answer:|\(answer>)\s*(.*?)(?:</answer>|$)', state['reasoning_raw_output'], re.DOTALL | re.IGNORECASE)
            
            # If thinking failed to parse correctly (i.e. it contains the question), try to clean it
            raw_thinking = "\n".join(state['reasoning_steps'])
            if question_match and question_match.group(1).strip() in raw_thinking:
                 # The raw output was probably just "Question: ..." without tags, so it got dumped into 'steps'
                 # We can try to separate if there are newlines, but it's risky.
                 pass

            qa_entry = {
                "type": "reasoning_qa",
                "anchor_id": state['anchor']['uuid'],
                "question": question_match.group(1).strip() if question_match else "Error parsing question",
                "thinking": raw_thinking,
                "answer": answer_match.group(1).strip() if answer_match else "Error parsing answer - Model failed to output <answer> tag"
            }
            
            # Only save if format check passed (or if check skipped/legacy)
            if state.get('format_check_passed', True):
                 state['all_outputs'] = state.get('all_outputs', []) + [qa_entry]
                 state['iteration_count'] += 1
            else:
                 print("Skipping invalid entry (Format Check Failed)")
            
            # Continue to save check logic below
        else:
            # Move to next step
            return {
                "current_step_index": state['current_step_index'] + 1,
                "step_retry_count": 0
            }
    
    # Save Logic: Every 100 iterations (or whatever batch size)
    # Note: SimpleQA increments iteration_count in its node, so we just check here.
    # ReasoningQA increments above.
    
    current_iter = state['iteration_count']
    if current_iter > 0 and current_iter % 5 == 0:
        last_saved = state.get('last_saved_count', 0)
        new_items = state.get('all_outputs', [])[last_saved:]
        
        if new_items:
            output_path = "/kaggle/working/generated_reasoning_qa.jsonl"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"Saving batch of {len(new_items)} items to {output_path}...")
            
            with open(output_path, 'a', encoding='utf-8') as f:
                for entry in new_items:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            state['last_saved_count'] = len(state.get('all_outputs', []))

    # Return valid state update
    # Note: For ReasoningQA, we manually updated state dict above, so we must return it or keys
    if state.get('is_reasoning_flow'):
         # If we just finished a reasoning QA
         if state['current_step_index'] == len(state['reasoning_steps']) - 1:
             return {
                 "all_outputs": state['all_outputs'],
                 "iteration_count": state['iteration_count'],
                 "current_step_index": state['current_step_index'] + 1,
                 "step_retry_count": 0,
                 "last_saved_count": state.get('last_saved_count', 0)
             }
         
    # Default for SimpleQA (iteration already incremented) or just passing through
    return {"last_saved_count": state.get('last_saved_count', 0)}


def check_format_node(state: AgentState) -> Dict[str, Any]:
    """
    Validates and auto-fixes the format of the reasoning output.
    Strictly requires <answer>...</answer> but will attempt to convert:
    Strictly requires <answer>...</answer> but will attempt to convert:
    - (answer) ...
    - (answer> ...
    - Answer: ...
    to <answer>...</answer> if found.
    """
    if not state.get('is_reasoning_flow'):
        return {"format_check_passed": True}

    raw_output = state.get('reasoning_raw_output', "")
    
    # 1. Check for basic "Question:" presence
    if "Question:" not in raw_output:
        msg = "Format Check Failed: Missing 'Question:'"
        print(f"X {msg}")
        return {
            "format_check_passed": False, 
            "reasoning_logs": state.get('reasoning_logs', []) + [msg]
        }

    # 2. Check for strictly valid <answer> tag
    # We want to ensure there is an <answer>... content ...</answer> block.
    strict_answer_pattern = r'<answer>.*?</answer>'
    if re.search(strict_answer_pattern, raw_output, re.DOTALL | re.IGNORECASE):
        # Already perfect
        return {"format_check_passed": True}

    # 3. Attempt Auto-Fix
    # Look for convertible patterns
    # Note: re.DOTALL means . matches newlines
    convertible_pattern = r'(?:<answer>|\(answer\)|Answer:|\(answer>)\s*(.*?)(?:</answer>|$)'
    match = re.search(convertible_pattern, raw_output, re.DOTALL | re.IGNORECASE)
    
    if match:
        answer_content = match.group(1).strip()
        # If the content is empty, that's also a failure of sorts, but let's assume valid content if matched.
        if not answer_content:
             msg = "Format Check Failed: Empty answer content"
             print(f"X {msg}")
             return {
                "format_check_passed": False, 
                "reasoning_logs": state.get('reasoning_logs', []) + [msg]
            }

        start_idx = match.start()
        end_idx = match.end()
        
        # Construct new output: Prefix + <answer>Content</answer> + Suffix
        # Note: raw_output[end_idx:] preserves anything after the answer block
        fixed_output = raw_output[:start_idx] + f"<answer>{answer_content}</answer>" + raw_output[end_idx:]
        
        log_msg = "Format Auto-Fixed: Converted answer format to <answer>...</answer>"
        print(f"✓ {log_msg}")
        return {
            "format_check_passed": True,
            "reasoning_raw_output": fixed_output,
            "reasoning_logs": state.get('reasoning_logs', []) + [log_msg]
        }

    # 4. Fail if no answer found
    msg = "Format Check Failed: No valid answer tag found"
    print(f"X {msg}")
    return {
        "format_check_passed": False, 
        "reasoning_logs": state.get('reasoning_logs', []) + [msg]
    }


# ==========================================
# 4. Workflow Definition (src/graph/workflow.py)
# ==========================================

def create_graph():
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("first_filter", first_filter_node)
    workflow.add_node("select_anchor", select_anchor_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("simple_qa", generate_simple_qa_node)
    workflow.add_node("generate_reasoning", generate_reasoning_node)
    workflow.add_node("parse_steps", parse_steps_node)
    workflow.add_node("verify_step", verify_single_step_node)
    workflow.add_node("refine_step", refine_single_step_node)
    workflow.add_node("check_more", check_more_questions_node)
    workflow.add_node("check_format", check_format_node)
    workflow.add_node("increment_step", increment_step_node)

    
    # Define Flow
    workflow.set_entry_point("first_filter")
    
    # First Filter: Decide between Normal QA and Reasoning QA
    def route_after_filter(state: AgentState):
        if state['is_reasoning_flow']:
            return "select_anchor"  # Reasoning path
        else:
            return "select_anchor"  # Both paths need anchor selection
    
    workflow.add_conditional_edges(
        "first_filter",
        route_after_filter,
        {
            "select_anchor": "select_anchor"
        }
    )
    
    # After anchor selection, route based on flow
    def route_after_anchor(state: AgentState):
        if state['is_reasoning_flow']:
             return "retrieve"
        else:
             return "simple_qa"

    workflow.add_conditional_edges(
        "select_anchor",
        route_after_anchor,
        {
            "retrieve": "retrieve",
            "simple_qa": "simple_qa"
        }
    )
    
    # After retrieval, go to reasoning generator
    # (Simple QA is now routed directly from select_anchor, so retrieve only goes to reasoning)
    workflow.add_edge("retrieve", "generate_reasoning")
    
    # Simple QA goes directly to check_more
    workflow.add_edge("simple_qa", "check_more")
    
    # Reasoning path: Generate → Parse Steps → Verify
    workflow.add_edge("generate_reasoning", "parse_steps")
    workflow.add_edge("parse_steps", "verify_step")
    
    # Conditional after single step verification
    def route_after_step_verification(state: AgentState):
        current_idx = state['current_step_index']
        retry_count = state.get('step_retry_count', 0)
        
        # Check if current step passed
        if not state['step_verification_results'][current_idx]:
            # If failed, check retries
            if retry_count < 3:
                return "refine_step"  # Retry
            else:
                # Max retries reached, just accept it and move on
                pass
        
        # Step passed OR max retries reached
        if current_idx + 1 < len(state['reasoning_steps']):
            return "increment_step"  # Go to next step (via increment node)
        else:
            return "check_format"  # All steps verified, format check
    
    workflow.add_conditional_edges(
        "verify_step",
        route_after_step_verification,
        {
            "refine_step": "refine_step",
            "increment_step": "increment_step",
            "check_format": "check_format"
        }
    )
    
    # After refinement, update retry count
    workflow.add_edge("refine_step", "verify_step")
    
    # After incrementing step, verify the new step
    workflow.add_edge("increment_step", "verify_step")
    
    # After step verification loop, run format check
    workflow.add_edge("check_format", "check_more")
    
    # Check if more questions need to be generated
    def route_after_check_more(state: AgentState):
        if state['iteration_count'] < state['max_iterations']:
            return "first_filter"  # Generate another question
        else:
            return END  # Done
    
    workflow.add_conditional_edges(
        "check_more",
        route_after_check_more,
        {
            "first_filter": "first_filter",
            END: END
        }
    )
    
    return workflow.compile()


# ==========================================
# 5. Main Execution (main.py)
# ==========================================

def main():
    # 1. Initialize Retriever (just for connectivity check and context for nodes)
    # logic moved: we don't load all data anymore.
    print("Initializing pipeline...")
    
    # Set global retriever for nodes to access
    # Note: Retriever now connects to Mongo internally, no need to add_documents
    try:
        retriever = MongoDBRetriever()
        builtins.RETRIEVER = retriever
    except Exception as e:
        print(f"Retriever initialization failed: {e}")
        print("Pipeline might fail if it relies on retrieval.")
    
    # 4. Initialize Graph
    app = create_graph()
    
    # 5. Start Local vLLM if configured
    if Config.USE_LOCAL_VLLM:
        print("Configured to use Local vLLM. Starting server...")
        start_vllm_background()
        # Override OpenAI Base URL to local
        Config.OPENAI_BASE_URL = "http://localhost:8000/v1"
        # Re-initialize client with new URL
        global openai_client
        openai_client = OpenAI(
            base_url=Config.OPENAI_BASE_URL,
            api_key=Config.OPENAI_API_KEY
        )
        print(f"✓ OpenAI Client updated to use: {Config.OPENAI_BASE_URL}")

    # 6. Run Pipeline with iterations
    num_qa_pairs = Config.NUM_QA_PAIRS
    
    print(f"Running pipeline to generate {num_qa_pairs} QA pairs...")
    
    initial_state = {
        "iteration_count": 0,
        "max_iterations": num_qa_pairs,
        "is_reasoning_flow": True,
        "anchor": None,
        "query": "",
        "context_docs": [],
        "simple_qa": None,
        "reasoning_qa": None,
        "reasoning_raw_output": "",
        "reasoning_steps": [],
        "current_step_index": 0,
        "step_retry_count": 0,
        "step_verification_results": [],
        "verification_passed": False,
        "reasoning_logs": [],
        "final_output_ready": False,
        "all_outputs": [],
        "used_anchor_ids": [],
        "last_saved_count": 0,
        "format_check_passed": True
    }
    
    # Run the graph once - it will iterate internally
    final_state = app.invoke(initial_state)
    
    # 6. Save Output (Final save for any remaining items)
    os.makedirs("data/output", exist_ok=True)
    output_path = "/kaggle/working/generated_reasoning_qa.jsonl"
    
    # We might have saved incrementally, so let's just save whatever is new/remaining or double check
    # The check_more_questions_node handles batch saving. 
    # But let's save everything to be sure if the last batch wasn't saved.
    
    last_saved = final_state.get('last_saved_count', 0)
    remaining_items = final_state['all_outputs'][last_saved:]
    
    if remaining_items:
        print(f"Saving final batch of {len(remaining_items)} items...")
        with open(output_path, 'a', encoding='utf-8') as f:
            for entry in remaining_items:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Done. Generated {len(final_state['all_outputs'])} QA pairs in total.")
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
