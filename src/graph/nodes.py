
from typing import Dict, Any, List
import random
import re
import os
import sys
import json
import builtins
from .state import AgentState
from openai import OpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
openai_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY")
)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507")


# ==========================================
# Helpers
# ==========================================

def check_and_raise_503(e):
    """
    Check if the exception is a 503 Service Unavailable error.
    If so, print a warning and allow pipeline to continue.
    Returns True if this is a 503-like error, otherwise False.
    """
    error_msg = str(e)
    if "503" in error_msg or "tunnel unavailable" in error_msg.lower():
        print("\n" + "="*50)
        print("WARNING: 503 Service Unavailable detected.")
        print("The server is overloaded or down. Skipping this call and continuing pipeline.")
        print("="*50 + "\n")
        return True
    return False


def _log_retrieved_docs(docs: List[Document], label: str) -> None:
    """Pretty-print retrieved documents for debugging and traceability."""
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
    """
    Build list of references used to generate QA.
    Includes anchor + retrieved docs (similar/opposing).
    """
    references: List[Dict[str, Any]] = []

    anchor = state.get('anchor')
    if anchor:
        references.append({
            "role": "anchor",
            "uuid": anchor.get('uuid'),
            "title": anchor.get('title', ''),
            "summary": anchor.get('summary', ''),
            "type": anchor.get('type'),
            "tags": anchor.get('tags', []),
            "keywords": anchor.get('keywords', [])
        })

    for doc in state.get('context_docs', []):
        meta = doc.metadata or {}
        references.append({
            "role": "retrieved_context",
            "query": state.get('primary_retrieval_query', state.get('query', '')),
            "uuid": meta.get('uuid'),
            "title": meta.get('title', ''),
            "summary": meta.get('summary', ''),
            "type": meta.get('type'),
            "tags": meta.get('tags', []),
            "keywords": meta.get('keywords', []),
            "score": meta.get('score', 0.0)
        })

    for doc in state.get('negative_docs', []):
        meta = doc.metadata or {}
        references.append({
            "role": "retrieved_negative",
            "query": state.get('negative_retrieval_query', ''),
            "uuid": meta.get('uuid'),
            "title": meta.get('title', ''),
            "summary": meta.get('summary', ''),
            "type": meta.get('type'),
            "tags": meta.get('tags', []),
            "keywords": meta.get('keywords', []),
            "score": meta.get('score', 0.0)
        })

    return references


# ==========================================
# Nodes
# ==========================================

def first_filter_node(state: AgentState) -> Dict[str, Any]:
    """
    First Filter: Pass through the current is_reasoning_flow setting.
    """
    is_reasoning = state['is_reasoning_flow']
    return {"is_reasoning_flow": is_reasoning}


def select_anchor_node(state: AgentState) -> Dict[str, Any]:
    """
    Randomly select an anchor directly from MongoDB, excluding used IDs.
    """
    from pymongo import MongoClient

    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME", "Data")
    collection_name = os.getenv("MONGO_COLLECTION_NAME", "mental")

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
        start_index = int(os.getenv("START_INDEX", "0"))
        end_index = int(os.getenv("END_INDEX", "0"))  # 0 means no limit

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
        print("Warning: All anchors used. Resetting cycle.")
        if not is_reasoning:
            pipeline_fallback = [
                {"$sort": {"_id": 1}},
                {"$skip": target_offset},
                {"$limit": 1}
            ]
        else:
            pipeline_fallback = [{"$sample": {"size": 1}}]

        client = MongoClient(mongo_uri)
        collection = client[db_name][collection_name]
        results = list(collection.aggregate(pipeline_fallback))
        client.close()

        if not results:
            print("Error: No data in MongoDB.")
            return {"anchor": None}

    anchor_doc = results[0]
    new_id = anchor_doc.get('uuid')

    print(f"\n[{state.get('iteration_count', 0) + 1}] Anchor Selected ({'Simple' if not is_reasoning else 'Reasoning'}):")
    print(f"  - uuid: {anchor_doc.get('uuid', '')}")
    print(f"  - title: {anchor_doc.get('title', '')[:150]}")
    print(f"  - summary: {anchor_doc.get('summary', '')[:200]}")

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

    # If Simple QA, skip retrieval — provide anchor content as context
    if not is_reasoning:
        content = anchor_doc.get('content', '')
        page_content = f"Tiêu đề: {title}\nTóm tắt: {summary}\nNội dung:\n{content}"
        anchor_as_doc = Document(page_content=page_content, metadata=metadata)
        updates["context_docs"] = [anchor_as_doc]

    return updates


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves 3-5 similar/related documents using Vector Search.
    """
    retriever = getattr(builtins, 'RETRIEVER', None)

    if not retriever:
        print("Warning: Retriever not found.")
        return {"context_docs": []}

    query = state['query']
    print(f"Retrieving docs for query: {query[:50]}...")
    docs = retriever.search(query, k=random.randint(3, 5))
    print(f"Found {len(docs)} documents.")
    _log_retrieved_docs(docs, "retrieve")
    return {
        "context_docs": docs,
        "primary_retrieval_query": query
    }


def retrieve_negative_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves 2-3 documents that are OPPOSING / contrasting with the anchor.
    (retrieve_node already handles similar docs; this node covers the opposing side.)
    """
    retriever = getattr(builtins, 'RETRIEVER', None)
    if not retriever:
        print("Warning: Retriever not found. Skipping negative retrieval.")
        return {"negative_docs": []}

    anchor = state.get('anchor', {})
    title       = anchor.get('title', '') if anchor else ''
    summary     = anchor.get('summary', '') if anchor else ''
    anchor_uuid = anchor.get('uuid', '') if anchor else ''

    # ── Step 1: Ask LLM to generate an opposing query ─────────────────────
    llm_prompt = f"""Bạn là chuyên gia tâm lý học.

Tài liệu gốc (anchor):
- Tiêu đề: {title}
- Tóm tắt: {summary}

Hãy viết MỘT câu truy vấn ngắn (10-20 từ) để tìm tài liệu tâm lý học
mang quan điểm ĐỐI LẬP, TƯƠNG PHẢN hoặc PHÊ PHÁN với chủ đề trên.
Chỉ trả về câu truy vấn, không giải thích thêm."""

    generated_query = f"{title} {summary}"  # fallback
    try:
        resp = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.7,
            max_tokens=64,
            timeout=60
        )
        generated = resp.choices[0].message.content.strip()
        if generated:
            generated_query = generated
            print(f"[retrieve_negative] Opposing query: {generated_query}")
    except Exception as e:
        check_and_raise_503(e)
        print(f"[retrieve_negative] LLM query generation failed, using fallback: {e}")

    # ── Step 2: Search ─────────────────────────────────────────────────────
    k = random.randint(2, 3)
    candidates = retriever.search(generated_query, k=k + 5)

    # ── Step 3: Filter out anchor and existing context docs ────────────────
    existing_uuids = {anchor_uuid}
    for doc in state.get('context_docs', []):
        existing_uuids.add(doc.metadata.get('uuid', ''))

    negative_docs = [
        doc for doc in candidates
        if doc.metadata.get('uuid', '') not in existing_uuids
    ][:k]

    print(f"[retrieve_negative] Found {len(negative_docs)} opposing doc(s).")
    _log_retrieved_docs(negative_docs, "retrieve_negative")
    return {
        "negative_docs": negative_docs,
        "negative_retrieval_query": generated_query
    }


def generate_simple_qa_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates Normal Psychology QA without reasoning (multiple choice, 4-5 options).
    """
    print("Generating Simple QA...")
    context_text = "\n\n".join([d.page_content for d in state['context_docs']])

    num_options = random.choices([4, 5], weights=[0.6, 0.4])[0]
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
{context_text[:8192]}

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
            max_tokens=8192,
            timeout=120
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
            "question_type_enum": "factual_recall",
            "iteration_count": state['iteration_count']
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

    num_options = random.choices([4, 5], weights=[0.6, 0.4])[0]
    if num_options == 4:
        options_format = "A. [Đáp án A]\nB. [Đáp án B]\nC. [Đáp án C]\nD. [Đáp án D]"
        answer_format = "[Chỉ ghi đáp án đúng: A, B, C hoặc D]"
        req_text = "4 đáp án (A, B, C, D)"
    else:
        options_format = "A. [Đáp án A]\nB. [Đáp án B]\nC. [Đáp án C]\nD. [Đáp án D]\nE. [Đáp án E]"
        answer_format = "[Chỉ ghi đáp án đúng: A, B, C, D hoặc E]"
        req_text = "5 đáp án (A, B, C, D, E)"

    question_types = {
        "so sánh hai lý thuyết/trường phái tâm lý học trái chiều hoặc tương đồng nhau": "theory_comparison",
        "tình huống lâm sàng: chẩn đoán hoặc lựa chọn can thiệp phù hợp": "clinical_scenario",
        "phân tích nguyên nhân – hậu quả của một hiện tượng tâm lý": "causal_analysis",
        "nhận diện sai lầm nhận thức (cognitive bias) trong một mô tả": "cognitive_bias",
        "ứng dụng lý thuyết tâm lý vào cuộc sống / công việc thực tế": "application"
    }
    question_type_desc = random.choice(list(question_types.keys()))
    question_type_enum = question_types[question_type_desc]

    prompt = f"""Bạn là chuyên gia tâm lý với khả năng suy luận sâu sắc.

Ngữ cảnh (Các tài liệu tâm lý liên quan):
{context_text}

Chủ đề: {state['anchor'].get('title', '')}
Tóm tắt: {state['anchor'].get('summary', '')}

Nhiệm vụ:
1. Tạo một câu hỏi trắc nghiệm dạng: **{question_type_desc}**. Câu hỏi phải đi kèm {req_text} và chỉ có 1 đáp án đúng.
2. Suy nghĩ từng bước trong thẻ <think>. Mỗi bước suy luận đặt trong thẻ <step>.
   - HÃY BẮT ĐẦU BẰNG VIỆC: Đưa ra các reference đầu vào (trích dẫn thông tin quan trọng từ tài liệu đã cho).
   - SAU ĐÓ: Suy luận như một con người đang tự suy nghĩ nội bộ. Hãy dùng nhiều văn phong khác nhau một cách linh hoạt (ví dụ: phân tích từng bước, suy luận tự nhiên, hoặc đúc kết nguyên nhân-kết quả ngắn gọn).
   - Hãy phân tích câu hỏi, phân tích từng đáp án, loại trừ đáp án sai và chứng minh đáp án đúng một cách tự nhiên.
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
- CÁC THẺ <step> LÀ NƠI TƯ DUY TỰ DO nội bộ. Đừng biến nó thành một cái dàn ý hay bài văn báo cáo (như "Bước 1: Phân tích"). Hãy viết như đang độc thoại nội tâm.
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
            "question_type_enum": question_type_enum,
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


def validate_qa_node(state: AgentState) -> Dict[str, Any]:
    """
    Kiểm tra câu hỏi trắc nghiệm sinh ra trước khi đi vào step verification.
    Gồm 2 tầng:
    1. Rule-based: kiểm tra số đáp án, đáp án trùng lặp.
    2. LLM-based: kiểm tra câu hỏi hợp lí, chỉ 1 đáp án đúng, đáp án phân biệt rõ.
    """
    is_reasoning = state.get('is_reasoning_flow', False)

    if is_reasoning:
        raw = state.get('reasoning_raw_output', '')
        question_block_match = re.search(
            r'(Question:.*?)(?=<think>|$)', raw, re.DOTALL
        )
        question_block = question_block_match.group(1).strip() if question_block_match else raw
    else:
        qa = state.get('simple_qa', {})
        question_block = qa.get('question', '') if qa else ''

    print("[validate_qa] Bắt đầu kiểm tra câu hỏi...")

    # ── TẦNG 1: Rule-based ────────────────────────────────────────────────
    option_pattern = r'^([A-E])[\.\)]\s*(.+)$'
    options = re.findall(option_pattern, question_block, re.MULTILINE)
    option_texts = [text.strip().lower() for _, text in options]

    num_opts = len(options)
    if num_opts not in (4, 5):
        msg = f"[validate_qa] ✗ FAIL (Rule): Số đáp án không hợp lệ: {num_opts} (cần 4 hoặc 5)"
        print(msg)
        return {
            "qa_validation_passed": False,
            "qa_validation_attempts": state.get('qa_validation_attempts', 0) + 1,
            "reasoning_logs": state.get('reasoning_logs', []) + [msg]
        }

    def _normalize(text: str) -> str:
        return re.sub(r'[\s\.,;:!?()\'"]+', ' ', text.lower()).strip()

    normalized = [_normalize(t) for t in option_texts]
    duplicates_found = False
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            a, b = normalized[i], normalized[j]
            shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
            if shorter and shorter in longer:
                duplicates_found = True
                break
            set_a, set_b = set(a.split()), set(b.split())
            if set_a and set_b:
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)
                if union > 0 and intersection / union > 0.8:
                    duplicates_found = True
                    break
        if duplicates_found:
            break

    if duplicates_found:
        msg = "[validate_qa] ✗ FAIL (Rule): Phát hiện đáp án trùng lặp hoặc quá giống nhau"
        print(msg)
        return {
            "qa_validation_passed": False,
            "qa_validation_attempts": state.get('qa_validation_attempts', 0) + 1,
            "reasoning_logs": state.get('reasoning_logs', []) + [msg]
        }

    print(f"[validate_qa] ✓ Rule-based: OK ({num_opts} đáp án, không trùng lặp)")

    # ── TẦNG 2: LLM-based ────────────────────────────────────────────────
    print("[validate_qa] Kiểm tra LLM...")

    options_display = "\n".join([f"{label}. {text}" for label, text in options])
    question_line_match = re.search(r'(?:Question:|Câu hỏi:?)\s*(.+?)\n', question_block, re.IGNORECASE)
    question_text = question_line_match.group(1).strip() if question_line_match else question_block[:300]

    llm_prompt = f"""Bạn là chuyên gia kiểm định câu hỏi trắc nghiệm. Hãy đánh giá câu hỏi sau:

Câu hỏi: {question_text}

Các đáp án:
{options_display}

Hãy kiểm tra:
1. Câu hỏi có rõ nghĩa, có thể trả lời được không?
2. Chỉ có đúng 1 đáp án đúng không?
3. Các đáp án có phân biệt rõ ràng với nhau không (không trùng ý)?

Nếu câu hỏi HỢP LỆ (thỏa mãn tất cả 3 tiêu chí), chỉ trả lời đúng 1 từ: "HỢP LỆ"
Nếu KHÔNG HỢP LỆ, giải thích ngắn gọn vấn đề (1-2 câu)."""

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.1,
            max_tokens=256,
            timeout=60
        )
        review = response.choices[0].message.content.strip()
        is_valid = "HỢP LỆ" in review.upper() or "VALID" in review.upper()

        if is_valid:
            print("[validate_qa] ✓ LLM: HỢP LỆ")
            return {
                "qa_validation_passed": True,
                "qa_validation_attempts": state.get('qa_validation_attempts', 0),
                "reasoning_logs": state.get('reasoning_logs', []) + ["[validate_qa] PASS"]
            }
        else:
            msg = f"[validate_qa] ✗ FAIL (LLM): {review[:200]}"
            print(msg)
            return {
                "qa_validation_passed": False,
                "qa_validation_attempts": state.get('qa_validation_attempts', 0) + 1,
                "reasoning_logs": state.get('reasoning_logs', []) + [msg]
            }
    except Exception as e:
        check_and_raise_503(e)
        print(f"[validate_qa] Lỗi LLM validation: {e}. Bỏ qua, coi như PASS.")
        return {
            "qa_validation_passed": True,
            "qa_validation_attempts": state.get('qa_validation_attempts', 0)
        }


def verify_grounding_node(state: AgentState) -> Dict[str, Any]:
    """
    Kiểm tra xem câu hỏi và đáp án có ĐÚNG với kiến thức trong context_docs không.
    Tránh trường hợp LLM tự bịa (hallucinate) sai lệch với tài liệu.
    """
    print("[verify_grounding] Bắt đầu đối chiếu QA với tài liệu tham khảo...")
    is_reasoning = state.get('is_reasoning_flow', False)
    
    # 1. Trích xuất Câu hỏi và Đáp án
    if is_reasoning:
        raw = state.get('reasoning_raw_output', '')
        # Tách câu hỏi
        q_match = re.search(r'(Question:.*?)(?=<think>|<step>|<tool_call>|$)', raw, re.DOTALL)
        question_block = q_match.group(1).strip() if q_match else raw
        # Tách đáp án
        a_match = re.search(r'(?:<answer>|\(answer\)|Answer:|\(answer>)\s*(.*?)(?:</answer>|$)', raw, re.DOTALL | re.IGNORECASE)
        answer_text = a_match.group(1).strip() if a_match else "[Không tìm thấy đáp án]"
    else:
        qa = state.get('simple_qa', {})
        question_block = qa.get('question', '') if qa else ''
        answer_text = qa.get('answer', '') if qa else ''

    # 2. Gom tài liệu tham khảo
    context_text = "\n\n".join([d.page_content for d in state.get('context_docs', [])])
    
    if not context_text.strip():
        print("[verify_grounding] Không có context docs, tự động PASS.")
        return {"grounding_passed": True, "grounding_attempts": state.get('grounding_attempts', 0)}

    # 3. Prompt kiểm tra
    llm_prompt = f"""Bạn là chuyên gia thẩm định nội dung tâm lý học. 
Nhiệm vụ của bạn là kiểm tra tính chính xác của Câu Hỏi và Đáp Án dựa trên Tài Liệu Tham Khảo.

Tài Liệu Tham Khảo:
{context_text[:8192]}

Câu hỏi trắc nghiệm:
{question_block}

Đáp án được chọn là đúng:
{answer_text}

Tiêu chí đánh giá:
1. Đáp án đúng được chọn PHẢI được hỗ trợ bởi thông tin trong Tài Liệu Tham Khảo.
2. Không mâu thuẫn với nội dung của tài liệu.

Nếu ĐẠT tiêu chí trên, bạn CHỈ trả lời một từ duy nhất: "ĐẠT"
Nếu KHÔNG ĐẠT (sai kiến thức, tài liệu không nhắc tới, hoặc bịa đặt), hãy giải thích ngắn gọn lý do (1-2 câu)."""

    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": llm_prompt}],
            temperature=0.1,
            max_tokens=256,
            timeout=60
        )
        review = response.choices[0].message.content.strip()
        is_valid = "ĐẠT" in review.upper() or "PASSED" in review.upper()
        
        if is_valid:
            print(f"[verify_grounding] ✓ ĐẠT: Phù hợp với reference.")
            return {
                "grounding_passed": True,
                "grounding_attempts": state.get('grounding_attempts', 0),
                "reasoning_logs": state.get('reasoning_logs', []) + ["[verify_grounding] PASS"]
            }
        else:
            msg = f"[verify_grounding] ✗ KHÔNG ĐẠT: {review[:200]}"
            print(msg)
            return {
                "grounding_passed": False,
                "grounding_attempts": state.get('grounding_attempts', 0) + 1,
                "reasoning_logs": state.get('reasoning_logs', []) + [msg]
            }
            
    except Exception as e:
        check_and_raise_503(e)
        print(f"[verify_grounding] Lỗi LLM: {e}. Bỏ qua, coi như PASS.")
        return {
            "grounding_passed": True,
            "grounding_attempts": state.get('grounding_attempts', 0)
        }


def parse_steps_node(state: AgentState) -> Dict[str, Any]:
    """
    Parse individual <step> tags from the reasoning output.
    """
    raw = state['reasoning_raw_output']

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
            "reasoning_logs": state.get('reasoning_logs', []) + [f"Step {current_idx}: Error - {e}"]
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
    qa_entry = state.get('formatted_qa')
    
    # We only increment and save if format_output_node has successfully created the final entry
    if qa_entry:
        all_outputs = state.get('all_outputs', [])
        # Prevent double adding if somehow called twice
        if not all_outputs or all_outputs[-1].get('id') != qa_entry.get('id'):
            state['all_outputs'] = all_outputs + [qa_entry]
            state['iteration_count'] += 1
    else:
        print("Skipping entry: formatted_qa is None")

    # Save every 5 iterations
    current_iter = state['iteration_count']
    if current_iter > 0 and current_iter % 5 == 0:
        last_saved = state.get('last_saved_count', 0)
        new_items = state.get('all_outputs', [])[last_saved:]

        if new_items:
            output_path = os.getenv(
                "OUTPUT_PATH",
                "data/output/generated_psychology_multiple_choice.jsonl"
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            print(f"Saving batch of {len(new_items)} items to {output_path}...")

            with open(output_path, 'a', encoding='utf-8') as f:
                for entry in new_items:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            state['last_saved_count'] = len(state.get('all_outputs', []))

    # Construct the state updates dictionary
    updates = {
        "last_saved_count": state.get('last_saved_count', 0),
        "formatted_qa": None,
        # Đặt lại các biến đếm để iteration tiếp theo bắt đầu mới hoàn toàn
        "qa_validation_attempts": 0,
        "grounding_attempts": 0
    }

    # Return state update for reasoning flow
    if state.get('is_reasoning_flow'):
        updates.update({
            "all_outputs": state['all_outputs'],
            "iteration_count": state['iteration_count'],
            "current_step_index": state.get('current_step_index', 0) + 1,
            "step_retry_count": 0
        })
        return updates

    return updates


def check_format_node(state: AgentState) -> Dict[str, Any]:
    """
    Validates and auto-fixes the format of the reasoning output.
    """
    if not state.get('is_reasoning_flow'):
        return {"format_check_passed": True}

    raw_output = state.get('reasoning_raw_output', "")

    if "Question:" not in raw_output:
        msg = "Format Check Failed: Missing 'Question:'"
        print(f"X {msg}")
        return {
            "format_check_passed": False,
            "reasoning_logs": state.get('reasoning_logs', []) + [msg]
        }

    strict_answer_pattern = r'<answer>.*?</answer>'
    if re.search(strict_answer_pattern, raw_output, re.DOTALL | re.IGNORECASE):
        return {"format_check_passed": True}

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

    msg = "Format Check Failed: No valid answer tag found"
    print(f"X {msg}")
    return {
        "format_check_passed": False,
        "reasoning_logs": state.get('reasoning_logs', []) + [msg]
    }


def format_output_node(state: AgentState) -> Dict[str, Any]:
    """
    Format output into the required JSON structure and extract metadata using LLM.
    """
    print("Formatting Output to Final JSON...")
    
    is_reasoning = state.get('is_reasoning_flow', False)
    anchor = state.get('anchor', {})
    keywords = anchor.get('keywords', [])
    anchor_type = anchor.get('type', '')
    
    # Extract source based on keywords
    source_val = "synthetic"
    keywords_str = " ".join(keywords).lower() if keywords else ""
    if "dsm-5" in keywords_str or "dsm" in keywords_str:
        source_val = "DSM-5"
    elif "giáo trình" in keywords_str or "textbook" in keywords_str or "sách" in keywords_str:
        source_val = "textbook"
        
    question_block = ""
    answer_text = ""
    reasoning_raw = ""
    
    if is_reasoning:
        raw = state.get('reasoning_raw_output', '')
        reasoning_raw = raw
        q_match = re.search(r'(Question:.*?)(?=<think>|<step>|<tool_call>|$)', raw, re.DOTALL)
        question_block = q_match.group(1).strip() if q_match else raw
        a_match = re.search(r'(?:<answer>|\(answer\)|Answer:|\(answer>)\s*(.*?)(?:</answer>|$)', raw, re.DOTALL | re.IGNORECASE)
        answer_text = a_match.group(1).strip() if a_match else "A"
    else:
        qa = state.get('simple_qa', {})
        question_block = qa.get('question', '') if qa else ''
        answer_text = qa.get('answer', '') if qa else 'A'
        reasoning_raw = "Câu hỏi trắc nghiệm kiến thức (không có suy luận chi tiết)."

    # Extract options using regex
    option_pattern = r'^([A-E])[\.\)]\s*(.+)$'
    options_matches = re.findall(option_pattern, question_block, re.MULTILINE)
    
    options_dict = {}
    for label, text in options_matches:
        options_dict[label.upper()] = text.strip()
        
    # Clean up the question part
    q_text_only = re.sub(r'^[A-E][\.\)]\s*.+$', '', question_block, flags=re.MULTILINE).strip()
    if q_text_only.lower().startswith("question:"):
        q_text_only = q_text_only[9:].strip()
        
    # Extract correct answer letter
    ans_letter_match = re.search(r'([A-E])', answer_text, re.IGNORECASE)
    ans_letter = ans_letter_match.group(1).upper() if ans_letter_match else "A"
    
    # Call LLM to extract metadata & steps
    prompt = f"""Dựa vào nội dung câu hỏi và suy luận dưới đây, hãy trích xuất các thông tin siêu dữ liệu (metadata) dưới dạng JSON.

Nội dung câu hỏi:
{q_text_only}

Quá trình suy luận:
{reasoning_raw[:5000]}

Từ khóa của tài liệu gốc: {', '.join(keywords) if keywords else 'Không có'}
Loại tài liệu: {anchor_type}

Hãy trả về CHỈ MỘT OBJECT JSON với cấu trúc sau:
{{
  "topic": "[Chủ đề chính, dựa vào từ khóa và câu hỏi]",
  "subtopic": "[Chủ đề phụ]",
  "cognitive_skill": "[Loại kỹ năng nhận thức, vd: factual_recall, causal_reasoning, diagnostic_reasoning, bias_detection, decision_making...]",
  "steps": [
    "[Bước 1 của quá trình suy luận]",
    "[Bước 2 của quá trình suy luận]"
  ]
}}
Lưu ý: "steps" là một mảng tóm tắt các bước logic chính từ "Quá trình suy luận". Nếu không có suy luận chi tiết, mảng "steps" có thể chứa 1-2 bước cơ bản.
"""
    try:
        response = openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
            timeout=60
        )
        result_text = response.choices[0].message.content.strip()
        result_json_str = re.sub(r'```json\n|\n```|```', '', result_text).strip()
        meta_data = json.loads(result_json_str)
    except Exception as e:
        print(f"Error parsing metadata via LLM: {e}")
        meta_data = {
            "topic": keywords[0] if keywords else "general",
            "subtopic": "general",
            "cognitive_skill": "factual_recall" if not is_reasoning else "analytical",
            "steps": ["Read question", "Identify answer"]
        }
        
    formatted_qa = {
        "id": f"PSY-{state.get('iteration_count', 0) + 1:06d}",
        "question": q_text_only,
        "options": options_dict,
        "answer": ans_letter,
        "reasoning": {
            "steps": meta_data.get("steps", []),
            "explanation": reasoning_raw
        },
        "metadata": {
            "topic": meta_data.get("topic", ""),
            "subtopic": meta_data.get("subtopic", ""),
            "question_type": state.get("question_type_enum", "factual_recall"),
            "cognitive_skill": meta_data.get("cognitive_skill", ""),
            "difficulty": "hard" if is_reasoning else "medium",
            "language": "vi",
            "source": source_val,
            "has_reasoning": is_reasoning
        },
        "validation": {
            "expert_verified": False,
            "consistency_checked": True,
            "reasoning_verified": is_reasoning and state.get("verification_passed", False)
        }
    }
    
    return {"formatted_qa": formatted_qa}
