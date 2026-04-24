"""
Judge AnswerBench results using GPT-5.2 (Azure OpenAI).
Extracts \boxed{} from model output, compares to gold answer via LLM.

Usage:
  export AZURE_OPENAI_KEY="your-key"
  python judge_answerbench.py --input outputs/qed-nano_answerbench_107q.jsonl \
                              --output outputs/qed-nano_answerbench_107q_judged.jsonl
"""
import json, os, re, time, argparse
from openai import AzureOpenAI

ENDPOINT = "https://msncompanioneu2.cognitiveservices.azure.com/"
DEPLOYMENT = "gpt-5-2"
API_VERSION = "2024-12-01-preview"
PROMPT_PATH = "configs/prompts/answerbench.txt"
PERSIST_DIR = "/scratch/azureml/cr/j/05c921e0efaa4264b68084c90beb0eaf/cap/data-capability/wd/INPUT_msndni/shares/UDC/RnR/Users/yuhangbai/QEDEval"

def find_last_boxed(text):
    """Extract last \\boxed{} content, handling nested braces."""
    matches = []
    i = 0
    while i < len(text):
        idx = text.find('\\boxed{', i)
        if idx == -1:
            break
        # Find matching closing brace
        depth = 0
        start = idx + 7  # len('\\boxed{')
        for j in range(start, len(text)):
            if text[j] == '{':
                depth += 1
            elif text[j] == '}':
                if depth == 0:
                    matches.append(text[start:j])
                    i = j + 1
                    break
                depth -= 1
        else:
            i = start
    return matches[-1].strip() if matches else None

def judge_one(client, prompt_template, problem, model_solution, gold_answer):
    """Judge a single answer."""
    extracted = find_last_boxed(model_solution)
    if not extracted:
        # Fallback: last 5 lines
        lines = model_solution.strip().split('\n')
        extracted = '...' + '\n'.join(lines[-5:]) if len(lines) >= 5 else model_solution[-500:]
    
    prompt = prompt_template.format(
        problem_statement=problem,
        student_answer=extracted,
        gold_answer=gold_answer
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=4096,
                model=DEPLOYMENT,
            )
            content = response.choices[0].message.content
            # Parse correctness: look for \boxed{correct} or \boxed{incorrect}
            judge_boxed = find_last_boxed(content)
            if judge_boxed:
                is_correct = "incorrect" not in judge_boxed.lower()
            else:
                is_correct = "correct" in content.lower() and "incorrect" not in content.lower()
            
            return {
                "extracted_answer": extracted,
                "is_correct": is_correct,
                "judge_response": content,
                "judge_tokens": response.usage.total_tokens if response.usage else 0,
            }
        except Exception as e:
            print(f"\n    Error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
            else:
                return {
                    "extracted_answer": extracted,
                    "is_correct": False,
                    "judge_response": str(e),
                    "judge_tokens": 0,
                }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    api_key = os.environ.get("AZURE_OPENAI_KEY")
    if not api_key:
        raise ValueError("AZURE_OPENAI_KEY not set")

    client = AzureOpenAI(
        azure_endpoint=ENDPOINT,
        api_key=api_key,
        api_version=API_VERSION,
    )

    with open(PROMPT_PATH) as f:
        prompt_template = f.read()

    with open(args.input) as f:
        data = [json.loads(l) for l in f]

    # Resume support
    existing = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            for l in f:
                d = json.loads(l)
                if 'is_correct' in d:
                    existing[d['question_id']] = d
        if existing:
            print(f"Resuming: {len(existing)} already judged")

    results = []
    correct = 0
    total = 0
    for i, d in enumerate(data):
        qid = d['question_id']
        
        if qid in existing:
            d = existing[qid]
            c = d.get('is_correct', False)
            print(f"  [{i+1}/{len(data)}] {qid:35s} ({d.get('category',''):15s}) → {'✓' if c else '✗'} (cached)")
            results.append(d)
            if c: correct += 1
            total += 1
            continue

        model_sol = d.get('model_solution', '')
        print(f"  [{i+1}/{len(data)}] {qid:35s} ({d.get('category',''):15s})", end=" ", flush=True)
        
        if not model_sol:
            print("→ ✗ (no output)")
            d['extracted_answer'] = ''
            d['is_correct'] = False
            d['judge_response'] = 'No model output'
        else:
            t0 = time.time()
            result = judge_one(client, prompt_template, d['problem'], model_sol, d['answer'])
            elapsed = time.time() - t0
            d.update(result)
            c = result['is_correct']
            print(f"→ {'✓' if c else '✗'} (gold={d['answer'][:20]}, extracted={result['extracted_answer'][:20]}) ({elapsed:.0f}s)")
            if c: correct += 1
        
        total += 1
        results.append(d)
        
        # Save after every problem
        with open(args.output, 'w') as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    # Final save to persistent dir
    os.makedirs(PERSIST_DIR, exist_ok=True)
    persist_path = os.path.join(PERSIST_DIR, os.path.basename(args.output))
    with open(persist_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n=== Summary ===")
    print(f"Correct: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Saved to {args.output}")
    print(f"Also saved to {persist_path}")

    # By category
    from collections import defaultdict
    by_cat = defaultdict(lambda: [0, 0])
    for r in results:
        cat = r.get('category', '?')
        by_cat[cat][1] += 1
        if r.get('is_correct', False):
            by_cat[cat][0] += 1
    print("\nBy category:")
    for cat in sorted(by_cat.keys()):
        c, t = by_cat[cat]
        print(f"  {cat:20s}: {c}/{t} ({c/t*100:.1f}%)")

if __name__ == "__main__":
    main()
