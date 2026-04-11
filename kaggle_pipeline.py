
!uv pip install pymongo openai langchain-core langchain-openai langgraph python-dotenv numpy sentence-transformers
!uv pip install vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly
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
    
    # CẬP NHẬT MODEL SANG QWEN 3.5
    MODEL_NAME = "Qwen/Qwen3.5-27B"
    
    # vLLM Command (for local deployment) - Cập nhật tên model
    VLLM_CMD = f'uv run vllm serve "{MODEL_NAME}" --dtype auto --gpu-memory-utilization 0.9 --max-model-len 14000 --host 0.0.0.0 --port 8000 --reasoning-parser qwen3 --trust-remote-code'
    
    # Embedding Configuration
    USE_LOCAL_EMBEDDING = True
    EMBEDDING_BASE_URL = "http://127.0.0.1:1234/v1" # Fallback if local is False
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
    primary_retrieval_query: str # Query used for main retrieval
    negative_retrieval_query: str # Query used for opposing retrieval
    context_docs: List[Document]# Retrieved documents (3-5 related/opposing)
    negative_docs: List[Document] # Retrieved documents opposing the anchor
    
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
    
    # QA Validation 
    qa_validation_passed: bool  
    qa_validation_attempts: int 
    
    # Grounding Validation 
    grounding_passed: bool      
    grounding_attempts: int     
    
    # Output Collection
    all_outputs: List[Dict[str, Any]]  # Collected QA pairs across iterations
    last_saved_count: int              # Index of the last saved QA pair


# ==========================================
# 2. Retriever Definition
# ==========================================

class MongoDBRetriever:
    """MongoDB-based retriever for Psychology data"""
    
    def __init__(self):
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
        
        self.use_local = Config.USE_LOCAL_EMBEDDING
        self.embedding_base_url = Config.EMBEDDING_BASE_URL
        self.embedding_model = Config.EMBEDDING_MODEL
        
        if self.use_local:
            print(f"✓ Initializing Local Embedding Model: {self.embedding_model}")
            if SentenceTransformer is None:
                raise ImportError("sentence-transformers not installed. Please pip install it.")
                
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
        try:
            if self.use_local and self.local_model:
                vector = self.local_model.encode(text, normalize_embeddings=True)
                return vector.tolist()
            else:
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=text
                )
                return response.data[0].embedding
        except Exception as e:
            print(f"Warning: Embedding generation failed: {e}")
            return [0.0] * 1024
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        if not hasattr(self, 'collection') or self.collection is None:
            return []
            
        if self.use_local and self.local_model is None:
             return []

        try:
            return self._vector_search(query, k)
        except Exception as e:
            try:
                return self._text_search(query, k)
            except Exception as e2:
                return self._keyword_search(query, k)
    
    def _vector_search(self, query: str, k: int) -> List[Document]:
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
                    "content": 1, "title": 1, "summary": 1, "uuid": 1,
                    "type": 1, "tags": 1, "keywords": 1,
                    "score": {"$meta": "vectorSearchScore"}
                }
            }
        ]
        results = list(self.collection.aggregate(pipeline))
        return self._format_results(results)
    
    def _text_search(self, query: str, k: int) -> List[Document]:
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
            {"$limit": k},
            {
                "$project": {
                    "content": 1, "title": 1, "summary": 1, "uuid": 1,
                    "type": 1, "tags": 1, "keywords": 1,
                    "score": {"$meta": "searchScore"}
                }
            }
        ]
        try:
            results = list(self.collection.aggregate(pipeline))
            return self._format_results(results)
        except Exception:
            return self._standard_text_search(query, k)

    def _standard_text_search(self, query: str, k: int) -> List[Document]:
        if not hasattr(self, 'collection') or self.collection is None:
            return []
        results = self.collection.find(
            {"$text": {"$search": query}},
            {"score": {"$meta": "textScore"}}
        ).sort([("score", {"$meta": "textScore"})]).limit(k)
        return self._format_results(list(results))
    
    def _keyword_search(self, query: str, k: int) -> List[Document]:
        if not hasattr(self, 'collection') or self.collection is None:
            return []
        keywords = query.lower().split()
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
                'type': result.get('type'),
                'tags': result.get('tags', []),
                'keywords': result.get('keywords', []),
                'score': result.get('score', 0.0)
            }
            documents.append(Document(page_content=page_content, metadata=metadata))
        return documents
    
    def close(self):
        if hasattr(self, 'client') and self.client:
            self.client.close()


openai_client = OpenAI(
    base_url=Config.OPENAI_BASE_URL,
    api_key=Config.OPENAI_API_KEY
)

MODEL_NAME = Config.MODEL_NAME

# ==========================================
# 3. Graph Nodes
# ==========================================

def _log_retrieved_docs(docs: List[Document], label: str) -> None:
    if not docs:
        print(f"[{label}] No documents retrieved.")
        return
    print(f"[{label}] Retrieved {len(docs)} document(s):")
    for idx, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        print(
            f"  - #{idx} | uuid={meta.get('uuid', '')} | "
            f"title={meta.get('title', '')[:120]} | "
            f"score={meta.get('score', 0.0)}"
        )

def _build_references(state: AgentState) -> List[Dict[str, Any]]:
    references: List[Dict[str, Any]] = []
    anchor = state.get('anchor')
    if anchor:
        references.append({
            "role": "anchor", "uuid": anchor.get('uuid'), "title": anchor.get('title', ''),
            "summary": anchor.get('summary', ''), "type": anchor.get('type'),
            "tags": anchor.get('tags', []), "keywords": anchor.get('keywords', [])
        })

    for doc in state.get('context_docs', []):
        meta = doc.metadata or {}
        references.append({
            "role": "retrieved_context", "query": state.get('primary_retrieval_query', state.get('query', '')),
            "uuid": meta.get('uuid'), "title": meta.get('title', ''), "summary": meta.get('summary', ''),
            "type": meta.get('type'), "tags": meta.get('tags', []), "keywords": meta.get('keywords', []),
            "score": meta.get('score', 0.0)
        })

    for doc in state.get('negative_docs', []):
        meta = doc.metadata or {}
        references.append({
            "role": "retrieved_negative", "query": state.get('negative_retrieval_query', ''),
            "uuid": meta.get('uuid'), "title": meta.get('title', ''), "summary": meta.get('summary', ''),
            "type": meta.get('type'), "tags": meta.get('tags', []), "keywords": meta.get('keywords', []),
            "score": meta.get('score', 0.0)
        })

    return references

def first_filter_node(state: AgentState) -> Dict[str, Any]:
    is_reasoning = state['is_reasoning_flow']
    return {"is_reasoning_flow": is_reasoning}

def select_anchor_node(state: AgentState) -> Dict[str, Any]:
    mongo_uri = Config.MONGO_URI
    db_name = Config.MONGO_DB_NAME
    collection_name = Config.MONGO_COLLECTION_NAME
    
    if not mongo_uri:
        return {"anchor": None}

    try:
        client = MongoClient(mongo_uri)
        collection = client[db_name][collection_name]
    except Exception as e:
        return {"anchor": None}
    
    used_ids = state.get('used_anchor_ids', [])
    is_reasoning = state.get('is_reasoning_flow', False)
    
    if not is_reasoning:
        start_index = Config.START_INDEX
        end_index = Config.END_INDEX
        target_offset = start_index + state.get('iteration_count', 0)
        
        if end_index > 0 and target_offset >= end_index:
            client.close()
            return {"anchor": None}

        pipeline = [{"$sort": {"_id": 1}}, {"$skip": target_offset}, {"$limit": 1}]
    else:
        pipeline = [{"$match": {"uuid": {"$nin": used_ids}}}, {"$sample": {"size": 1}}]
    
    results = list(collection.aggregate(pipeline))
    client.close()
    
    if not results:
        if not is_reasoning:
             pipeline_fallback = [{"$sort": {"_id": 1}}, {"$skip": target_offset}, {"$limit": 1}]
        else:
             pipeline_fallback = [{"$sample": {"size": 1}}]
             
        client = MongoClient(mongo_uri)
        collection = client[db_name][collection_name]
        results = list(collection.aggregate(pipeline_fallback))
        client.close()
        
        if not results:
            return {"anchor": None}

    anchor_doc = results[0]
    new_id = anchor_doc.get('uuid')
    
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
        "primary_retrieval_query": f"{title} {summary}",
        "used_anchor_ids": list(set(used_ids) | {new_id})
    }
    
    if not is_reasoning:
        content = anchor_doc.get('content', '')
        page_content = f"Tiêu đề: {title}\nTóm tắt: {summary}\nNội dung:\n{content}"
        anchor_as_doc = Document(page_content=page_content, metadata=metadata)
        updates["context_docs"] = [anchor_as_doc]
        
    return updates

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    retriever = getattr(builtins, 'RETRIEVER', None)
    if not retriever:
        return {"context_docs": []}

    query = state['query']
    docs = retriever.search(query, k=random.randint(3, 5))
    _log_retrieved_docs(docs, "retrieve")
    return {
        "context_docs": docs,
        "primary_retrieval_query": query
    }

def retrieve_negative_node(state: AgentState) -> Dict[str, Any]:
    retriever = getattr(builtins, 'RETRIEVER', None)
    if not retriever:
        return {"negative_docs": []}

    anchor = state.get('anchor', {})
    title       = anchor.get('title', '') if anchor else ''
    summary     = anchor.get('summary', '') if anchor else ''
    anchor_uuid = anchor.get('uuid', '') if anchor else ''

    llm_prompt = f"""Bạn là chuyên gia tâm lý học.

Tài liệu gốc (anchor):
- Tiêu đề: {title}
- Tóm tắt: {summary}

Hãy viết MỘT câu truy vấn ngắn (10-20 từ) để tìm tài liệu tâm lý học
mang quan điểm ĐỐI LẬP, TƯƠNG PHẢN hoặc PHÊ PHÁN với chủ đề trên.
Chỉ trả về câu truy vấn, không giải thích thêm."""

    generated_query = f"{title} {summary}"
    try:
        resp = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.7,
            max_tokens=64,
            timeout=60,
            # TẮT THINKING ĐỂ LẤY CÂU TRUY VẤN NHANH
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        generated = resp.choices[0].message.content.strip()
        if generated:
            generated_query = generated
    except Exception as e:
        check_and_raise_503(e)

    k = random.randint(2, 3)
    candidates = retriever.search(generated_query, k=k + 5)

    existing_uuids = {anchor_uuid}
    for doc in state.get('context_docs', []):
        existing_uuids.add(doc.metadata.get('uuid', ''))

    negative_docs = [
        doc for doc in candidates
        if doc.metadata.get('uuid', '') not in existing_uuids
    ][:k]

    return {
        "negative_docs": negative_docs,
        "negative_retrieval_query": generated_query
    }

def generate_simple_qa_node(state: AgentState) -> Dict[str, Any]:
    context_text = "\n\n".join([d.page_content for d in state['context_docs']])
    
    num_options = random.choice([4, 5])
    if num_options == 4:
        options_format = "A. [Đáp án A]\nB. [Đáp án B]\nC. [Đáp án C]\nD. [Đáp án D]"
        answer_format = "[Đáp án đúng: A, B, C hoặc D]"
        req_text = "4 đáp án (A, B, C, D)"
    else:
        options_format = "A. [Đáp án A]\nB. [Đáp án B]\nC. [Đáp án C]\nD. [Đáp án D]\nE. [Đáp án E]"
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
            timeout=120,
            # TẮT THINKING ĐỂ TRẢ VỀ JSON TRỰC TIẾP
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        
        result = response.choices[0].message.content
        qa = json.loads(result.replace("```json", "").replace("```", "").strip())
        
        output_entry = {
            "type": "simple_qa",
            "anchor_id": state['anchor']['uuid'],
            "question": qa.get('question'),
            "answer": qa.get('answer'),
            "references": _build_references(state)
        }
        
        return {
            "simple_qa": qa,
            "all_outputs": state.get('all_outputs', []) + [output_entry],
            "iteration_count": state['iteration_count'] + 1
        }
    except Exception as e:
        check_and_raise_503(e)
        return {"iteration_count": state['iteration_count'] + 1}

def generate_reasoning_node(state: AgentState) -> Dict[str, Any]:
    if not state.get('anchor'):
        return {
             "reasoning_raw_output": "", "current_step_index": 0,
             "step_retry_count": 0, "step_verification_results": []
        }

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

    question_types = [
        "so sánh hai lý thuyết/trường phái tâm lý học trái chiều hoặc tương đồng nhau",
        "tình huống lâm sàng: chẩn đoán hoặc lựa chọn can thiệp phù hợp",
        "phân tích nguyên nhân – hậu quả của một hiện tượng tâm lý",
        "nhận diện sai lầm nhận thức (cognitive bias) trong một mô tả",
        "ứng dụng lý thuyết tâm lý vào cuộc sống / công việc thực tế",
    ]
    question_type = random.choice(question_types)

    prompt = f"""Bạn là chuyên gia tâm lý với khả năng suy luận sâu sắc.

Ngữ cảnh:
{context_text}

Chủ đề: {state['anchor'].get('title', '')}
Tóm tắt: {state['anchor'].get('summary', '')}

Nhiệm vụ:
1. Tạo một câu hỏi trắc nghiệm với độ phức tạp khó dạng: **{question_type}**. Câu hỏi phải đi kèm {req_text} và chỉ có 1 đáp án đúng.
2. Suy nghĩ từng bước trong thẻ <think>. Mỗi bước suy luận đặt trong thẻ <step>.
3. Đưa ra đáp án cuối cùng trong <answer>

**format bắt buộc:**
Question: [Câu hỏi trắc nghiệm tâm lý?]
{options_format}

<think>
<step>[Ý nghĩ đầu tiên...]</step>
<step>[Dòng suy nghĩ tiếp theo...]</step>
</think>

<answer>{answer_format}</answer>
"""
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=8196,
            timeout=300,
            # BẬT THINKING ĐỂ YÊU CẦU MODEL SUY LUẬN SÂU (SINH RA THẺ <think>)
            extra_body={"chat_template_kwargs": {"enable_thinking": True}}
        )
        
        result = response.choices[0].message.content
        return {
            "reasoning_raw_output": result,
            "current_step_index": 0,
            "step_retry_count": 0,
            "step_verification_results": []
        }
    except Exception as e:
        check_and_raise_503(e)
        return {
             "reasoning_raw_output": "", "current_step_index": 0,
             "step_retry_count": 0, "step_verification_results": []
        }

def validate_qa_node(state: AgentState) -> Dict[str, Any]:
    is_reasoning = state.get('is_reasoning_flow', False)
    
    if is_reasoning:
        raw = state.get('reasoning_raw_output', '')
        question_block_match = re.search(r'(Question:.*?)(?=<think>|$)', raw, re.DOTALL)
        question_block = question_block_match.group(1).strip() if question_block_match else raw
    else:
        qa = state.get('simple_qa', {})
        question_block = qa.get('question', '') if qa else ''

    option_pattern = r'^([A-E])[\.\)]\s*(.+)$'
    options = re.findall(option_pattern, question_block, re.MULTILINE)
    
    num_opts = len(options)
    if num_opts not in (4, 5):
        return {
            "qa_validation_passed": False,
            "qa_validation_attempts": state.get('qa_validation_attempts', 0) + 1,
        }

    options_display = "\n".join([f"{label}. {text}" for label, text in options])
    question_line_match = re.search(r'(?:Question:|Câu hỏi:?)\s*(.+?)\n', question_block, re.IGNORECASE)
    question_text = question_line_match.group(1).strip() if question_line_match else question_block[:300]
    
    llm_prompt = f"""Đánh giá câu hỏi sau:

Câu hỏi: {question_text}

Các đáp án:
{options_display}

Hãy kiểm tra:
1. Câu hỏi rõ nghĩa?
2. Chỉ có 1 đáp án đúng?
3. Các đáp án phân biệt rõ ràng?

Trả lời "HỢP LỆ" nếu đạt cả 3, ngược lại nêu lý do sai."""

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.1,
            max_tokens=256,
            timeout=60,
            # TẮT THINKING ĐỂ CHECK NHANH
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        review = response.choices[0].message.content.strip()
        is_valid = "HỢP LỆ" in review.upper() or "VALID" in review.upper()
        
        return {
            "qa_validation_passed": is_valid,
            "qa_validation_attempts": state.get('qa_validation_attempts', 0) + (0 if is_valid else 1)
        }
    except Exception as e:
        check_and_raise_503(e)
        return {"qa_validation_passed": True, "qa_validation_attempts": state.get('qa_validation_attempts', 0)}

def verify_grounding_node(state: AgentState) -> Dict[str, Any]:
    is_reasoning = state.get('is_reasoning_flow', False)
    
    if is_reasoning:
        raw = state.get('reasoning_raw_output', '')
        q_match = re.search(r'(Question:.*?)(?=<think>|<step>|<tool_call>|$)', raw, re.DOTALL)
        question_block = q_match.group(1).strip() if q_match else raw
        a_match = re.search(r'(?:<answer>|\(answer\)|Answer:|\(answer>)\s*(.*?)(?:</answer>|$)', raw, re.DOTALL | re.IGNORECASE)
        answer_text = a_match.group(1).strip() if a_match else ""
    else:
        qa = state.get('simple_qa', {})
        question_block = qa.get('question', '') if qa else ''
        answer_text = qa.get('answer', '') if qa else ''

    context_text = "\n\n".join([d.page_content for d in state.get('context_docs', [])])
    if not context_text.strip():
        return {"grounding_passed": True, "grounding_attempts": state.get('grounding_attempts', 0)}

    llm_prompt = f"""Kiểm tra độ chính xác factual:

Tài Liệu:
{context_text[:6000]}

Câu hỏi:
{question_block}

Đáp án:
{answer_text}

Nếu đáp án đúng được hỗ trợ bởi tài liệu, trả lời "ĐẠT". Ngược lại giải thích lỗi."""

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.1,
            max_tokens=256,
            timeout=60,
            # TẮT THINKING ĐỂ CHECK NHANH
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        review = response.choices[0].message.content.strip()
        is_valid = "ĐẠT" in review.upper() or "PASSED" in review.upper()
        
        return {
            "grounding_passed": is_valid,
            "grounding_attempts": state.get('grounding_attempts', 0) + (0 if is_valid else 1)
        }
    except Exception as e:
        check_and_raise_503(e)
        return {"grounding_passed": True, "grounding_attempts": state.get('grounding_attempts', 0)}

def parse_steps_node(state: AgentState) -> Dict[str, Any]:
    raw = state['reasoning_raw_output']
    step_pattern = r'<step>(.*?)</step>'
    steps = re.findall(step_pattern, raw, re.DOTALL)
    
    if not steps:
        steps = [raw]
    
    return {
        "reasoning_steps": steps,
        "step_verification_results": [False] * len(steps)
    }

def verify_single_step_node(state: AgentState) -> Dict[str, Any]:
    current_idx = state['current_step_index']
    if current_idx >= len(state['reasoning_steps']):
        return {"verification_passed": True}
    
    current_step = state['reasoning_steps'][current_idx]
    
    prompt = f"""Kiểm tra logic của bước suy luận tâm lý sau:

{current_step}

Nếu bước này ĐÚNG logic, chỉ trả lời duy nhất: "ĐÚNG".
Nếu SAI, chỉ giải thích ngắn gọn lỗi."""
    
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,
            timeout=120,
            # TẮT THINKING VÌ ĐÂY LÀ TÁC VỤ ĐÁNH GIÁ CHUẨN
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
        )
        
        review = response.choices[0].message.content
        is_valid = "ĐÚNG" in review.upper() or "VALID" in review.upper()
        
        results = state['step_verification_results'].copy()
        results[current_idx] = is_valid
        
        return {"step_verification_results": results}
    except Exception as e:
        check_and_raise_503(e)
        results = state['step_verification_results'].copy()
        results[current_idx] = False
        return {"step_verification_results": results}

def refine_single_step_node(state: AgentState) -> Dict[str, Any]:
    current_idx = state['current_step_index']
    current_step = state['reasoning_steps'][current_idx]
    
    prompt = f"""Bước suy luận tâm lý sau có lỗi:

{current_step}

Viết lại bước này cho ĐÚNG logic."""
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=800,
            # TẮT THINKING KHI SỬA LỖI STEP ĐƠN LẺ
            extra_body={"chat_template_kwargs": {"enable_thinking": False}}
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
        return {"step_retry_count": state.get('step_retry_count', 0) + 1}

def increment_step_node(state: AgentState) -> Dict[str, Any]:
    return {
        "current_step_index": state['current_step_index'] + 1,
        "step_retry_count": 0
    }

def check_more_questions_node(state: AgentState) -> Dict[str, Any]:
    if state.get('is_reasoning_flow') and state['current_step_index'] == len(state['reasoning_steps']) - 1:
        question_match = re.search(r'Question:\s*(.*?)(?=<think>|<step>|<tool_call>|$)', state['reasoning_raw_output'], re.DOTALL)
        answer_match = re.search(r'(?:<answer>|\(answer\)|Answer:|\(answer>)\s*(.*?)(?:</answer>|$)', state['reasoning_raw_output'], re.DOTALL | re.IGNORECASE)
        
        raw_thinking = "\n".join(state['reasoning_steps'])

        qa_entry = {
            "type": "reasoning_qa",
            "anchor_id": state['anchor']['uuid'],
            "question": question_match.group(1).strip() if question_match else "Error parsing question",
            "thinking": raw_thinking,
            "answer": answer_match.group(1).strip() if answer_match else "Error parsing answer",
            "references": _build_references(state)
        }
        
        if state.get('format_check_passed', True):
             state['all_outputs'] = state.get('all_outputs', []) + [qa_entry]
             state['iteration_count'] += 1
    
    current_iter = state['iteration_count']
    if current_iter > 0 and current_iter % 5 == 0:
        last_saved = state.get('last_saved_count', 0)
        new_items = state.get('all_outputs', [])[last_saved:]
        
        if new_items:
            output_path = "/kaggle/working/generated_psychology_multiple_choice.jsonl"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'a', encoding='utf-8') as f:
                for entry in new_items:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            state['last_saved_count'] = len(state.get('all_outputs', []))

    updates = {
        "last_saved_count": state.get('last_saved_count', 0),
        "qa_validation_attempts": 0,
        "grounding_attempts": 0
    }

    if state.get('is_reasoning_flow'):
         if state['current_step_index'] == len(state['reasoning_steps']) - 1:
             updates.update({
                 "all_outputs": state['all_outputs'],
                 "iteration_count": state['iteration_count'],
                 "current_step_index": state['current_step_index'] + 1,
                 "step_retry_count": 0
             })
             return updates
         
    return updates

def check_format_node(state: AgentState) -> Dict[str, Any]:
    if not state.get('is_reasoning_flow'):
        return {"format_check_passed": True}

    raw_output = state.get('reasoning_raw_output', "")
    
    if "Question:" not in raw_output:
        return {"format_check_passed": False}

    strict_answer_pattern = r'<answer>.*?</answer>'
    if re.search(strict_answer_pattern, raw_output, re.DOTALL | re.IGNORECASE):
        return {"format_check_passed": True}

    convertible_pattern = r'(?:<answer>|\(answer\)|Answer:|\(answer>)\s*(.*?)(?:</answer>|$)'
    match = re.search(convertible_pattern, raw_output, re.DOTALL | re.IGNORECASE)
    
    if match:
        answer_content = match.group(1).strip()
        if not answer_content:
             return {"format_check_passed": False}

        start_idx = match.start()
        end_idx = match.end()
        fixed_output = raw_output[:start_idx] + f"<answer>{answer_content}</answer>" + raw_output[end_idx:]
        
        return {
            "format_check_passed": True,
            "reasoning_raw_output": fixed_output
        }

    return {"format_check_passed": False}


# ==========================================
# 4. Workflow Definition
# ==========================================

def create_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("first_filter", first_filter_node)
    workflow.add_node("select_anchor", select_anchor_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("retrieve_negative", retrieve_negative_node)
    workflow.add_node("simple_qa", generate_simple_qa_node)
    workflow.add_node("generate_reasoning", generate_reasoning_node)
    workflow.add_node("validate_qa", validate_qa_node)
    workflow.add_node("verify_grounding", verify_grounding_node) 
    workflow.add_node("parse_steps", parse_steps_node)
    workflow.add_node("verify_step", verify_single_step_node)
    workflow.add_node("refine_step", refine_single_step_node)
    workflow.add_node("check_more", check_more_questions_node)
    workflow.add_node("check_format", check_format_node)
    workflow.add_node("increment_step", increment_step_node)

    workflow.set_entry_point("first_filter")
    
    def route_after_filter(state: AgentState):
        return "select_anchor"
    
    workflow.add_conditional_edges("first_filter", route_after_filter, {"select_anchor": "select_anchor"})
    
    def route_after_anchor(state: AgentState):
        if not state['is_reasoning_flow']:
            return "simple_qa"
        return "retrieve" if random.random() < 0.5 else "retrieve_negative"

    workflow.add_conditional_edges("select_anchor", route_after_anchor, {
        "retrieve": "retrieve",
        "retrieve_negative": "retrieve_negative",
        "simple_qa": "simple_qa"
    })

    workflow.add_edge("retrieve", "generate_reasoning")
    workflow.add_edge("retrieve_negative", "generate_reasoning")
    workflow.add_edge("simple_qa", "validate_qa")
    workflow.add_edge("generate_reasoning", "validate_qa")
    
    def route_after_validate(state: AgentState):
        passed = state.get('qa_validation_passed', False)
        attempts = state.get('qa_validation_attempts', 0)
        is_reasoning = state.get('is_reasoning_flow', False)
        
        if passed:
            return "verify_grounding"
        else:
            if attempts < 2:
                return "generate_reasoning" if is_reasoning else "simple_qa"
            else:
                return "check_more"
    
    workflow.add_conditional_edges("validate_qa", route_after_validate, {
        "verify_grounding": "verify_grounding",
        "check_more": "check_more",
        "generate_reasoning": "generate_reasoning",
        "simple_qa": "simple_qa"
    })

    def route_after_grounding(state: AgentState):
        passed = state.get('grounding_passed', False)
        attempts = state.get('grounding_attempts', 0)
        is_reasoning = state.get('is_reasoning_flow', False)
        
        if passed:
            if is_reasoning:
                return "parse_steps"
            else:
                return "check_more"
        else:
            if attempts < 2:
                return "generate_reasoning" if is_reasoning else "simple_qa"
            else:
                return "check_more"

    workflow.add_conditional_edges("verify_grounding", route_after_grounding, {
        "parse_steps": "parse_steps",
        "check_more": "check_more",
        "generate_reasoning": "generate_reasoning",
        "simple_qa": "simple_qa"
    })
    
    workflow.add_edge("parse_steps", "verify_step")
    
    def route_after_step_verification(state: AgentState):
        current_idx = state['current_step_index']
        retry_count = state.get('step_retry_count', 0)
        
        if not state['step_verification_results'][current_idx]:
            if retry_count < 3:
                return "refine_step"
        
        if current_idx + 1 < len(state['reasoning_steps']):
            return "increment_step"
        else:
            return "check_format"
    
    workflow.add_conditional_edges("verify_step", route_after_step_verification, {
        "refine_step": "refine_step",
        "increment_step": "increment_step",
        "check_format": "check_format"
    })
    
    workflow.add_edge("refine_step", "verify_step")
    workflow.add_edge("increment_step", "verify_step")
    workflow.add_edge("check_format", "check_more")
    
    def route_after_check_more(state: AgentState):
        if state['iteration_count'] < state['max_iterations']:
            return "first_filter"
        else:
            return END
    
    workflow.add_conditional_edges("check_more", route_after_check_more, {
        "first_filter": "first_filter",
        END: END
    })
    
    return workflow.compile()


# ==========================================
# 5. Main Execution
# ==========================================

def main():
    print("Initializing QA Generation Pipeline with Qwen 3.5...")
    
    try:
        retriever = MongoDBRetriever()
        builtins.RETRIEVER = retriever
    except Exception as e:
        print(f"Retriever initialization failed: {e}")
    
    app = create_graph()
    
    if Config.USE_LOCAL_VLLM:
        start_vllm_background()
        Config.OPENAI_BASE_URL = "http://localhost:8000/v1"
        global openai_client
        openai_client = OpenAI(
            base_url=Config.OPENAI_BASE_URL,
            api_key=Config.OPENAI_API_KEY
        )
        print(f"✓ OpenAI Client updated to use: {Config.OPENAI_BASE_URL}")

    num_qa_pairs = Config.NUM_QA_PAIRS
    
    used_anchor_ids = []
    output_path = "/kaggle/working/generated_psychology_multiple_choice.jsonl"
    existing_output_files = [
        output_path,
        "/kaggle/input/datasets/hiudev/psychology/generated_psychology_multiple_choice-2.jsonl",
    ]
    for existing_file in existing_output_files:
        if os.path.exists(existing_file):
            with open(existing_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        aid = entry.get("anchor_id")
                        if aid and aid not in used_anchor_ids:
                            used_anchor_ids.append(aid)
                    except json.JSONDecodeError:
                        pass
    
    initial_state = {
        "iteration_count": 0, "max_iterations": num_qa_pairs, "is_reasoning_flow": True,
        "anchor": None, "query": "", "primary_retrieval_query": "", "negative_retrieval_query": "",
        "context_docs": [], "negative_docs": [], "simple_qa": None, "reasoning_qa": None,
        "reasoning_raw_output": "", "reasoning_steps": [], "current_step_index": 0,
        "step_retry_count": 0, "step_verification_results": [], "verification_passed": False,
        "reasoning_logs": [], "format_check_passed": True, "final_output_ready": False,
        "all_outputs": [], "used_anchor_ids": used_anchor_ids, "last_saved_count": 0,
        "qa_validation_passed": False, "qa_validation_attempts": 0,
        "grounding_passed": False, "grounding_attempts": 0
    }
    
    final_state = app.invoke(initial_state)
    
    last_saved = final_state.get('last_saved_count', 0)
    remaining_items = final_state['all_outputs'][last_saved:]
    
    if remaining_items:
        with open(output_path, 'a', encoding='utf-8') as f:
            for entry in remaining_items:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Done. Generated {len(final_state['all_outputs'])} Psychology QA pairs.")

if __name__ == "__main__":
    main()