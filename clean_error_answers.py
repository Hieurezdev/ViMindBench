
import json
import os

FILE_PATH = "data/output/generated_reasoning_qa.jsonl"
TEMP_FILE = "data/output/generated_reasoning_qa.clean.jsonl"
ERROR_STRING = "Error parsing answer - Model failed to output <answer> tag"

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
                # Check for the specific error message in answer
                answer = data.get("answer", "")
                if ERROR_STRING in answer:
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
        print(f"Removed {removed_count} lines containing error.")
        print(f"Kept {kept_count} lines.")
    else:
        os.remove(TEMP_FILE)
        print("No lines deemed for removal. File unchanged.")

if __name__ == "__main__":
    clean_file()
