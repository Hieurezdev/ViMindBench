
import json
import os

FILE_PATH = "data/output/generated_reasoning_qa.jsonl"
TEMP_FILE = "data/output/generated_reasoning_qa.clean.jsonl"

def clean_file():
    if not os.path.exists(FILE_PATH):
        print(f"File not found: {FILE_PATH}")
        return

    removed_count = 0
    kept_count = 0

    print(f"Scanning {FILE_PATH}...")
    
    with open(FILE_PATH, 'r', encoding='utf-8') as fin, \
         open(TEMP_FILE, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # Check validation: ensure all 3 fields exist and are not empty
                question = data.get("question")
                thinking = data.get("thinking")
                answer = data.get("answer")

                if not (question and thinking and answer):
                    removed_count += 1
                    continue
                
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                kept_count += 1
                
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line[:50]}...")
                continue

    # Replace original file
    if removed_count > 0:
        os.replace(TEMP_FILE, FILE_PATH)
        print(f"Cleanup complete. Overwrote {FILE_PATH}")
        print(f"Removed {removed_count} lines missing fields.")
        print(f"Kept {kept_count} lines.")
    else:
        os.remove(TEMP_FILE)
        print("No lines deemed for removal. File unchanged.")
        print(f"Total lines scanned: {kept_count}")

if __name__ == "__main__":
    clean_file()
