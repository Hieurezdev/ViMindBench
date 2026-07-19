import re


def test_answer_parsing():
    # Helper to clean up output
    def parse(regex, text):
        match = re.search(regex, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else "Error parsing answer"

    # The NEW Regex from nodes.py
    # Includes: <answer>, (answer), Answer:, (answer>
    new_regex = r"(?:<answer>|\(answer\)|Answer:|\(answer>)\s*(.*?)(?:</answer>|$)"

    print(f"Testing Regex: {new_regex}\n")

    # Case 1: Issue case from data (using (answer) tag)
    case1 = """
<step> ... </step>
(answer)Câu trả lời hoàn chỉnh: 
Đây là câu trả lời.
</answer>"""
    ans1 = parse(new_regex, case1)
    print(f"Case 1 (answer): {'PASS' if ans1.startswith('Câu trả lời') else 'FAIL'}")
    if not ans1.startswith("Câu trả lời"):
        print(f"  Got: {ans1}")

    # Case 2: Standard <answer> tag
    case2 = """
<step> ... </step>
<answer>
Normal answer.
</answer>"""
    ans2 = parse(new_regex, case2)
    print(f"Case 2 <answer>: {'PASS' if ans2 == 'Normal answer.' else 'FAIL'}")

    # Case 3: The NEW requested case (answer>
    case3 = """
<step> ... </step>
(answer>
Đây là câu trả lời mới.
</answer>"""
    ans3 = parse(new_regex, case3)
    print(f"Case 3 (answer>: {'PASS' if ans3 == 'Đây là câu trả lời mới.' else 'FAIL'}")
    if ans3 != "Đây là câu trả lời mới.":
        print(f"  Got: {ans3}")

    # Case 4: Answer: variation
    case4 = """
<step> ... </step>
Answer: 
Final Answer here.
"""
    ans4 = parse(new_regex, case4)
    print(f"Case 4 Answer:: {'PASS' if ans4 == 'Final Answer here.' else 'FAIL'}")


if __name__ == "__main__":
    test_answer_parsing()
