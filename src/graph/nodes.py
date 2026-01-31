
from typing import Dict, Any, List
import random
import re
import os
from .state import AgentState
from openai import OpenAI
import json
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client with custom endpoint support
openai_client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL"),
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY")
)

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-30B-A3B-Instruct-2507")

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
    from pymongo import MongoClient
    
    # Connect to MongoDB (quick connection, lightweight)
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME", "Data")
    collection_name = os.getenv("MONGO_COLLECTION_NAME", "mental")
    
    client = MongoClient(mongo_uri)
    collection = client[db_name][collection_name]
    
    used_ids = state.get('used_anchor_ids', [])
    is_reasoning = state.get('is_reasoning_flow', False)
    
    if not is_reasoning:
        # Simple QA: Sequential selection based on global index range
        start_index = int(os.getenv("START_INDEX", "6598"))
        end_index = int(os.getenv("END_INDEX", "0")) # 0 means no limit/all
        
        # Calculate which document to fetch: Start + current_iteration
        # Note: input 'iteration_count' starts at 0
        target_offset = start_index + state.get('iteration_count', 0)
        
        if end_index > 0 and target_offset >= end_index:
            print(f"Reached END_INDEX ({end_index}). Stop selection.")
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
        # For infinite loop support, we reset used_ids effectively by querying without exclusion
        print("Warning: All anchors used. Resetting cycle.")
        
        # Fallback logic also respects flow type
        if not is_reasoning:
             # Retry the same offset if missed? Or just fail. 
             # If empty at offset, it means we ran out of data.
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
    
    print(f"\n[{state.get('iteration_count', 0) + 1}] Anchor Selected ({'Simple' if not is_reasoning else 'Reasoning'}): {anchor_doc.get('summary')[:100]}...")

    # Map to metadata format expected by pipeline
    # Note: Chunker mapping logic simulated here
    metadata = {
        'uuid': anchor_doc.get('uuid'),
        'headers': anchor_doc.get('headers'),
        'summary': anchor_doc.get('summary'),
        'keywords': anchor_doc.get('keywords', []),
        'type': anchor_doc.get('type')
    }
    
    updates = {
        "anchor": metadata,
        "query": f"{metadata['summary']} {' '.join(metadata.get('keywords', []))}",
        "used_anchor_ids": list(set(used_ids) | {new_id})
    }
    
    # If Simple QA, we skip retrieval, so we must provide the anchor content as context
    if not is_reasoning:
        from langchain_core.documents import Document
        content = anchor_doc.get('content', '')
        # Create a Document object for the anchor
        anchor_as_doc = Document(page_content=content, metadata=metadata)
        updates["context_docs"] = [anchor_as_doc]
        
    return updates

def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieves 3-5 related/opposing documents using BM25 + Vector Search.
    """
    import builtins
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
    
    prompt = f"""Bạn là chuyên gia tâm lý học. Dựa vào ngữ cảnh sau, tạo một cặp Câu hỏi - Câu trả lời về tâm lý.
    
Ngữ cảnh:
{context_text[:2000]}

Chủ đề: {state['anchor'].get('summary', '')}

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
        print(f"Error in simple_qa: {e}")
        return {"iteration_count": state['iteration_count'] + 1}

def generate_reasoning_node(state: AgentState) -> Dict[str, Any]:
    """
    Generates Reasoning QA with <think> and <step> tags.
    """
    print("Generating Reasoning QA (this may take a while)...")
    context_text = "\n\n".join([d.page_content for d in state['context_docs']])
    
    prompt = f"""Bạn là chuyên gia tâm lý với khả năng suy luận sâu sắc.

Ngữ cảnh:
{context_text}

Chủ đề: {state['anchor'].get('summary', '')}

Nhiệm vụ:
1. Tạo một câu hỏi phức tạp cần suy luận về tâm lý học
2. Suy nghĩ từng bước trong thẻ <think>. Mỗi bước suy luận đặt trong thẻ <step>.
   - Bạn có thể suy nghĩ theo cách tự nhiên nhất của mình
   - Không cần theo format cứng nhắc, hãy viết như cách bạn thực sự suy nghĩ
   - Mỗi <step> có thể là phân tích, đặt câu hỏi, so sánh, kết nối ý tưởng, etc.
3. Đưa ra câu trả lời cuối cùng trong <answer>

Ví dụ format bắt buộc (bạn PHẢI tuân thủ cấu trúc này):
Question: [Câu hỏi của bạn]
<think>
<step> Phân tích...</step>
<step> Suy luận...</step>
<step> Tổng hợp...</step>
</think>
<answer>Câu trả lời hoàn chỉnh</answer>

LƯU Ý QUAN TRỌNG:
- BẮT BUỘC dùng thẻ <think>, <step>, <answer>.
- KHÔNG được bỏ qua bất kỳ thẻ nào.
- Nội dung trong <step> phải là suy nghĩ chi tiết.
"""
    
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=7000,
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
    
    response = openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=7000,
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
            question_match = re.search(r'Question:\s*(.*?)(?=<think>|$)', state['reasoning_raw_output'], re.DOTALL)
            answer_match = re.search(r'<answer>(.*?)</answer>', state['reasoning_raw_output'], re.DOTALL)
            
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
                
            state['all_outputs'] = state.get('all_outputs', []) + [qa_entry]
            state['iteration_count'] += 1
            
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
    import os
    import json
    
    current_iter = state['iteration_count']
    if current_iter > 0 and current_iter % 50 == 0:
        last_saved = state.get('last_saved_count', 0)
        new_items = state.get('all_outputs', [])[last_saved:]
        
        if new_items:
            output_path = "data/output/generated_simple_qa.jsonl"
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
