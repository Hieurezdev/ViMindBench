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
    MONGO_URI = "mongodb+srv://nguyenphamtrunghieu277_db_user:hieudevdut@cluster0.kf8srgq.mongodb.net/"
    MONGO_DB_NAME = "Data"
    MONGO_COLLECTION_NAME = "mental"
    
    # OpenAI / LLM Configuration
    USE_LOCAL_VLLM = True # Set to True to use local vLLM
    
    # Note: If USE_LOCAL_VLLM is True, OPENAI_BASE_URL will be overwritten to localhost
    OPENAI_BASE_URL = "https://silver-pots-jog.loca.lt/v1" 
    OPENAI_API_KEY = "EMPTY"
    MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    
    # vLLM Command (for local deployment)
    VLLM_CMD = 'uv run vllm serve "Qwen/Qwen3-30B-A3B-Instruct-2507" --dtype auto --gpu-memory-utilization 0.85 --max-model-len 14000 --host 0.0.0.0 --port 8000 --trust-remote-code'
    
    # Embedding Configuration
    USE_LOCAL_EMBEDDING = True
    EMBEDDING_BASE_URL = "http://127.0.0.1:1234/v1" # Fallback if local is False
    # Using a valid Hugging Face model ID for sentence-transformers
    EMBEDDING_MODEL = "namdp-ptit/ViDense" 
    
    # Pipeline Settings
    START_INDEX = 0
    END_INDEX = 0  # 0 means no limit
    NUM_QA_PAIRS = 5000

# Global context for retriever
builtins.RETRIEVER = None

# ==========================================
# 1. State Definition
# ==========================================

class AgentState(TypedDict):
    # Input Data
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
    
    # Reasoning Step-by-Step Verification
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
# 2. Retriever Definition
# ==========================================

class MongoDBRetriever:
    """
    MongoDB-based retriever for Psychology data using:
    1. Atlas Vector Search (if available)
    2. Text search fallback
    3. Local embedding model via sentence-transformers
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
        """Generate embedding vector using local model or API."""
        try:
            if self.use_local and self.local_model:
                # Use local sentence-transformers model
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
        Search MongoDB for relevant psychology documents.
        """
        if not hasattr(self, 'collection') or self.collection is None:
            print("Error: MongoDB collection not initialized. Cannot search.")
            return []
            
        if self.use_local and self.local_model is None:
             print("Error: Local model failed to load. Cannot search.")
             return []

        try:
            # Strategy 1: Try Atlas Vector Search
            return self._vector_search(query, k)
        except Exception as e:
            print(f"Vector search failed: {e}, falling back to text search")
            try:
                # Strategy 2: MongoDB text search
                return self._text_search(query, k)
            except Exception as e2:
                print(f"Text search failed: {e2}, falling back to keyword search")
                # Strategy 3: Simple keyword search
                return self._keyword_search(query, k)
    
    def _vector_search(self, query: str, k: int) -> List[Document]:
        """
        Perform Atlas Vector Search for psychology data.
        """
        if not hasattr(self, 'collection') or self.collection is None:
            return []

        query_embedding = self._generate_embedding(query)
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
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
                    "uuid": 1,
                    "type": 1,
                    "tags": 1,
                    "keywords": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        
        results = list(self.collection.aggregate(pipeline))
        return self._format_results(results)
    
    def _text_search(self, query: str, k: int) -> List[Document]:
        """
        Fallback to Mongo Atlas Text Search for psychology data.
        """
        if not hasattr(self, 'collection') or self.collection is None:
            return []

        pipeline = [
            {
                "$search": {
                    "index": "atlas_index",
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
                    "uuid": 1,
                    "type": 1,
                    "tags": 1,
                    "keywords": 1,
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
        Simple keyword-based search using regex for psychology data.
        """
        if not hasattr(self, 'collection') or self.collection is None:
            return []

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
        """Convert MongoDB results to LangChain Documents for psychology data."""
        documents = []
        
        for result in results:
            # Construct page_content from psychology fields
            title = result.get('title', '')
            summary = result.get('summary', '')
            content = result.get('content', '')
            
            # Combine into a structured string for the LLM
            page_content = f"Tiêu đề: {title}\nTóm tắt: {summary}\nNội dung:\n{content}"
            
            # Metadata
            metadata = {
                'uuid': result.get('uuid'),
                'title': title,
                'summary': summary,
                'type': result.get('type'),
                'tags': result.get('tags', []),
                'keywords': result.get('keywords', []),
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

# ==========================================
# 3. Graph Nodes
# ==========================================

def first_filter_node(state: AgentState) -> Dict[str, Any]:
    """
    First Filter: Randomly decide if this iteration generates Simple QA or Reasoning QA.
    """
    is_reasoning = state['is_reasoning_flow']
    return {"is_reasoning_flow": is_reasoning}

def select_anchor_node(state: AgentState) -> Dict[str, Any]:
    """
    Randomly select an anchor directly from MongoDB, excluding used IDs.
    """
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
        # Fallback: if all used, clear used list and try again
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
    summary = anchor_doc.get('summary', '')
    
    metadata = {
        'uuid': anchor_doc.get('uuid'),
        'title': title,
        'summary': summary,
        'type': anchor_doc.get('type'),
        'tags': anchor_doc.get('tags', []),
        'keywords': anchor_doc.get('keywords', [])
    }
    
    updates = {
        "anchor": metadata,
        "query": f"{title} {summary}",
        "used_anchor_ids": list(set(used_ids) | {new_id})
    }
    
    # If Simple QA, we skip retrieval, so we must provide the anchor content as context
    if not is_reasoning:
        content = anchor_doc.get('content', '')
        page_content = f"Tiêu đề: {title}\nTóm tắt: {summary}\nNội dung:\n{content}"
        # Create a Document object for the anchor
        anchor_as_doc = Document(page_content=page_content, metadata=metadata)
        updates["context_docs"] = [anchor_as_doc]
        
    return updates

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves 3-5 related/opposing documents using Vector Search.
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
    Generates Normal Psychology QA without reasoning.
    """
    print("Generating Simple QA...")
    context_text = "\n\n".join([d.page_content for d in state['context_docs']])
    
    num_options = random.choice([4, 5])
    if num_options == 4:
        options_format = "A. [Đáp án A]\\nB. [Đáp án B]\\nC. [Đáp án C]\\nD. [Đáp án D]"
        answer_format = "[Đáp án đúng: A, B, C hoặc D]"
        req_text = "4 đáp án (A, B, C, D)"
    else:
        options_format = "A. [Đáp án A]\\nB. [Đáp án B]\\nC. [Đáp án C]\\nD. [Đáp án D]\\nE. [Đáp án E]"
        answer_format = "[Đáp án đúng: A, B, C, D hoặc E]"
        req_text = "5 đáp án (A, B, C, D, E)"

    prompt = f"""Bạn là chuyên gia tâm lý học. Dựa vào ngữ cảnh sau, tạo một câu hỏi trắc nghiệm và câu trả lời về tâm lý. Câu hỏi phải gồm {req_text} và chỉ có 1 đáp án đúng.
    
Ngữ cảnh:
{context_text[:6000]}

Chủ đề: {state['anchor'].get('title', '')}
Tóm tắt: {state['anchor'].get('summary', '')}

**Format JSON:** {{"question": "[Câu hỏi]\\n{options_format}", "answer": "{answer_format} - [Giải thích]"}}

**tuyệt đối: Không được lặp lại đáp án nhiều lần trong câu hỏi**
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
    Generates Reasoning Psychology QA with <think> and <step> tags.
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
    
    num_options = random.choice([4, 5])
    if num_options == 4:
        options_format = "A. [Đáp án A]\nB. [Đáp án B]\nC. [Đáp án C]\nD. [Đáp án D]"
        answer_format = "[Chỉ ghi đáp án đúng: A, B, C hoặc D]"
        req_text = "4 đáp án (A, B, C, D)"
    else:
        options_format = "A. [Đáp án A]\nB. [Đáp án B]\nC. [Đáp án C]\nD. [Đáp án D]\nE. [Đáp án E]"
        answer_format = "[Chỉ ghi đáp án đúng: A, B, C, D hoặc E]"
        req_text = "5 đáp án (A, B, C, D, E)"

    prompt = f"""Bạn là chuyên gia tâm lý với khả năng suy luận sâu sắc.

Ngữ cảnh (Các tài liệu tâm lý liên quan):
{context_text}

Chủ đề: {state['anchor'].get('title', '')}
Tóm tắt: {state['anchor'].get('summary', '')}

Nhiệm vụ:
1. Tạo một câu hỏi trắc nghiệm phức tạp, cần sự suy luận nhiều hoặc các câu hỏi có sự so sánh về tâm lý học. Câu hỏi phải đi kèm {req_text} và chỉ có 1 đáp án đúng.
2. Suy nghĩ từng bước trong thẻ <think>. Mỗi bước suy luận đặt trong thẻ <step>.
   - Bạn có thể suy nghĩ theo cách tự nhiên nhất của mình
   - Hãy phân tích câu hỏi, phân tích từng đáp án, loại trừ đáp án sai và chứng minh đáp án đúng
   - Không cần cứng nhắc các step như ví dụ, hãy linh hoạt
   - Mỗi <step> có thể là phân tích, so sánh, kết nối ý tưởng, etc.
3. Đưa ra đáp án cuối cùng trong <answer>

**format bắt buộc (bạn PHẢI tuân thủ cấu trúc này):**
Question: [Câu hỏi trắc nghiệm tâm lý?]
{options_format}

<think>
<step>[Ý nghĩ đầu tiên khi suy nghĩ chi tiết về kiến thức tâm lý liên quan đến chủ đề này...]</step>
<step>[Dòng suy nghĩ tiếp theo, lật lại vấn đề, so sánh các ý tưởng, đánh giá độ khó...]</step>
... (tùy ý thêm số lượng step. Hãy viết dòng suy nghĩ một cách liền mạch, chân thật và lộn xộn như não người đang tư duy thực sự, tự phản biện. KHÔNG dùng các tiêu đề mẫu cứng nhắc như "Phân tích:", "Xem xét:")
</think>

<answer>{answer_format}</answer>

LƯU Ý QUAN TRỌNG:
- BẮT BUỘC dùng thẻ <think>, <step>, <answer>.
- TUYỆT ĐỐI KHÔNG nhắc đến các từ như "ngữ cảnh", "tài liệu đã cho", "đoạn văn". Hãy hành xử như thể toàn bộ nội dung kiến thức là do BẠN TỰ BIẾT và đang nhớ lại từ kho tàng tâm lý học của chính mình.
- CÁC THẺ <step> LÀ NƠI TƯ DUY TỰ DO Nội bộ. Đừng biến nó thành một cái dàn ý hay bài văn báo cáo (như "Bước 1: Phân tích"). Hãy viết như đang độc thoại nội tâm.
- Câu hỏi PHẢI có đúng {req_text} và chỉ 1 đáp án đúng. KHÔNG THÊM HAY BỚT ĐÁP ÁN.
- KHÔNG được bỏ qua bất kỳ thẻ nào.
"""
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=8196,
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
        steps = [raw]
    
    print(f"Parsed {len(steps)} steps for verification.")
    
    return {
        "reasoning_steps": steps,
        "step_verification_results": [False] * len(steps)
    }

def verify_single_step_node(state: AgentState) -> Dict[str, Any]:
    """
    Verify the current step only.
    """
    current_idx = state['current_step_index']
    if current_idx >= len(state['reasoning_steps']):
        return {"verification_passed": True}
    
    current_step = state['reasoning_steps'][current_idx]
    print(f"Verifying Step {current_idx}...")
    
    prompt = f"""Kiểm tra logic của bước suy luận tâm lý sau:

{current_step}

Nếu bước này ĐÚNG logic, chỉ trả lời duy nhất: "ĐÚNG".
Nếu SAI, chỉ giải thích ngắn gọn lỗi.
TUYỆT ĐỐI KHÔNG thêm bất kỳ câu thừa nào như "Sửa lỗi như sau", "Đề xuất sửa", v.v...
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
        
        results = state['step_verification_results'].copy()
        results[current_idx] = is_valid
        
        return {
            "step_verification_results": results,
            "reasoning_logs": state.get('reasoning_logs', []) + [f"Step {current_idx}: {review}"]
        }
    except Exception as e:
        check_and_raise_503(e)
        print(f"Error in verify_single_step_node: {e}")
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
    
    prompt = f"""Bước suy luận tâm lý sau có lỗi:

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
    """
    # If in reasoning flow and current step verified, save the reasoning QA
    if state.get('is_reasoning_flow') and state['current_step_index'] == len(state['reasoning_steps']) - 1:
        # Construct final reasoning QA
        question_match = re.search(r'Question:\s*(.*?)(?=<think>|<step>|<tool_call>|$)', state['reasoning_raw_output'], re.DOTALL)
        answer_match = re.search(r'(?:<answer>|\(answer\)|Answer:|\(answer>)\s*(.*?)(?:</answer>|$)', state['reasoning_raw_output'], re.DOTALL | re.IGNORECASE)
        
        raw_thinking = "\n".join(state['reasoning_steps'])

        qa_entry = {
            "type": "reasoning_qa",
            "anchor_id": state['anchor']['uuid'],
            "question": question_match.group(1).strip() if question_match else "Error parsing question",
            "thinking": raw_thinking,
            "answer": answer_match.group(1).strip() if answer_match else "Error parsing answer"
        }
        
        # Only save if format check passed
        if state.get('format_check_passed', True):
             state['all_outputs'] = state.get('all_outputs', []) + [qa_entry]
             state['iteration_count'] += 1
        else:
             print("Skipping invalid entry (Format Check Failed)")
    
    # Save Logic: Every 5 iterations
    current_iter = state['iteration_count']
    if current_iter > 0 and current_iter % 5 == 0:
        last_saved = state.get('last_saved_count', 0)
        new_items = state.get('all_outputs', [])[last_saved:]
        
        if new_items:
            output_path = "/content/drive/MyDrive/output/generated_psychology_multiple_choice.jsonl"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"Saving batch of {len(new_items)} items to {output_path}...")
            
            with open(output_path, 'a', encoding='utf-8') as f:
                for entry in new_items:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            state['last_saved_count'] = len(state.get('all_outputs', []))

    # Return state update
    if state.get('is_reasoning_flow'):
         if state['current_step_index'] == len(state['reasoning_steps']) - 1:
             return {
                 "all_outputs": state['all_outputs'],
                 "iteration_count": state['iteration_count'],
                 "current_step_index": state['current_step_index'] + 1,
                 "step_retry_count": 0,
                 "last_saved_count": state.get('last_saved_count', 0)
             }
         
    return {"last_saved_count": state.get('last_saved_count', 0)}

def check_format_node(state: AgentState) -> Dict[str, Any]:
    """
    Validates and auto-fixes the format of the reasoning output.
    """
    if not state.get('is_reasoning_flow'):
        return {"format_check_passed": True}

    raw_output = state.get('reasoning_raw_output', "")
    
    # Check for basic "Question:" presence
    if "Question:" not in raw_output:
        msg = "Format Check Failed: Missing 'Question:'"
        print(f"X {msg}")
        return {
            "format_check_passed": False, 
            "reasoning_logs": state.get('reasoning_logs', []) + [msg]
        }

    # Check for strictly valid <answer> tag
    strict_answer_pattern = r'<answer>.*?</answer>'
    if re.search(strict_answer_pattern, raw_output, re.DOTALL | re.IGNORECASE):
        return {"format_check_passed": True}

    # Attempt Auto-Fix
    convertible_pattern = r'(?:<answer>|\(answer\)|Answer:|\(answer>)\s*(.*?)(?:</answer>|$)'
    match = re.search(convertible_pattern, raw_output, re.DOTALL | re.IGNORECASE)
    
    if match:
        answer_content = match.group(1).strip()
        if not answer_content:
             msg = "Format Check Failed: Empty answer content"
             print(f"X {msg}")
             return {
                "format_check_passed": False, 
                "reasoning_logs": state.get('reasoning_logs', []) + [msg]
            }

        start_idx = match.start()
        end_idx = match.end()
        
        fixed_output = raw_output[:start_idx] + f"<answer>{answer_content}</answer>" + raw_output[end_idx:]
        
        log_msg = "Format Auto-Fixed: Converted answer format to <answer>...</answer>"
        print(f"✓ {log_msg}")
        return {
            "format_check_passed": True,
            "reasoning_raw_output": fixed_output,
            "reasoning_logs": state.get('reasoning_logs', []) + [log_msg]
        }

    # Fail if no answer found
    msg = "Format Check Failed: No valid answer tag found"
    print(f"X {msg}")
    return {
        "format_check_passed": False, 
        "reasoning_logs": state.get('reasoning_logs', []) + [msg]
    }

# ==========================================
# 4. Workflow Definition
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
        return "select_anchor"
    
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
                return "refine_step"
            else:
                pass
        
        # Step passed OR max retries reached
        if current_idx + 1 < len(state['reasoning_steps']):
            return "increment_step"
        else:
            return "check_format"
    
    workflow.add_conditional_edges(
        "verify_step",
        route_after_step_verification,
        {
            "refine_step": "refine_step",
            "increment_step": "increment_step",
            "check_format": "check_format"
        }
    )
    
    # After refinement, verify again
    workflow.add_edge("refine_step", "verify_step")
    
    # After incrementing step, verify the new step
    workflow.add_edge("increment_step", "verify_step")
    
    # After format check, go to check_more
    workflow.add_edge("check_format", "check_more")
    
    # Check if more questions need to be generated
    def route_after_check_more(state: AgentState):
        if state['iteration_count'] < state['max_iterations']:
            return "first_filter"
        else:
            return END
    
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
# 5. Main Execution
# ==========================================

def main():
    print("Initializing Psychology QA Generation Pipeline...")
    
    # Set global retriever for nodes to access
    try:
        retriever = MongoDBRetriever()
        builtins.RETRIEVER = retriever
    except Exception as e:
        print(f"Retriever initialization failed: {e}")
        print("Pipeline might fail if it relies on retrieval.")
    
    # Initialize Graph
    app = create_graph()
    
    # Start Local vLLM if configured
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

    # Run Pipeline
    num_qa_pairs = Config.NUM_QA_PAIRS
    
    print(f"Running pipeline to generate {num_qa_pairs} Psychology QA pairs...")
    
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
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    # Final save
    output_path = "/content/drive/MyDrive/output/generated_psychology_multiple_choice.jsonl"
    
    last_saved = final_state.get('last_saved_count', 0)
    remaining_items = final_state['all_outputs'][last_saved:]
    
    if remaining_items:
        print(f"Saving final batch of {len(remaining_items)} items...")
        with open(output_path, 'a', encoding='utf-8') as f:
            for entry in remaining_items:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Done. Generated {len(final_state['all_outputs'])} Psychology QA pairs in total.")
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
