"""
Judge and compare all 3 models using GPT-5.2
Usage:
  export AZURE_OPENAI_KEY="your-key"
  python judge_and_compare.py
  
Saves:
  - Per-model judged JSONL (with scores per problem)
  - Final comparison summary
"""
import json, os, re, time, sys
from openai import AzureOpenAI
from collections import defaultdict

ENDPOINT = "https://msncompanioneu2.cognitiveservices.azure.com/"
DEPLOYMENT = "gpt-5-2"
API_VERSION = "2024-12-01-preview"
PROMPT_PATH = "configs/prompts/proofbench.txt"
PERSIST_DIR = "/scratch/azureml/cr/j/05c921e0efaa4264b68084c90beb0eaf/cap/data-capability/wd/INPUT_msndni/shares/UDC/RnR/Users/yuhangbai/QEDEval"

MODELS = {
    "qed-nano": "outputs/qed-nano_20q.jsonl",
    "qwen3-4b-thinking": "outputs/qwen3-4b-thinking_20q.jsonl",
    "qed-nano-sft": "outputs/qed-nano-sft_20q.jsonl",
}

def load_prompt():
    with open(PROMPT_PATH) as f:
        return f.read()

def judge_one(client, prompt_template, d):
    model_sol = d.get("model_solution", "")
    if not model_sol:
        return {"points": 0, "judge_response": "No proof generated", "judge_tokens": 0}
    
    prompt = prompt_template.replace("{problem_statement}", d["problem"])
    prompt = prompt.replace("{guidelines}", d.get("grading_guidelines", ""))
    prompt = prompt.replace("{student_answer}", model_sol)
    prompt = prompt.replace("{solution}", d.get("solution", ""))

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=16384,
                model=DEPLOYMENT,
            )
            content = response.choices[0].message.content
            match = re.search(r'<points>\s*(\d+)', content)
            points = int(match.group(1)) if match else -1
            return {
                "points": points,
                "judge_response": content,
                "judge_tokens": response.usage.total_tokens if response.usage else 0,
            }
        except Exception as e:
            print(f"\n    Error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time
                time.sleep(10 * (attempt + 1))  # backoff: 10s, 20s
            else:
                return {"points": -1, "judge_response": str(e), "judge_tokens": 0}

def judge_model(client, prompt_template, model_name, input_path):
    print(f"\n{'='*60}")
    print(f"  Judging: {model_name}")
    print(f"{'='*60}")
    
    with open(input_path) as f:
        lines = f.readlines()
    
    # Support resume: check if partial results exist
    output_path = input_path.replace(".jsonl", "_judged.jsonl")
    existing_results = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                d = json.loads(line)
                if "judge_points" in d:
                    existing_results[d["question_id"]] = d
        if existing_results:
            print(f"  Resuming: {len(existing_results)} problems already judged")
    
    results = []
    for i, line in enumerate(lines):
        d = json.loads(line)
        qid = d["question_id"]
        level = d.get("level", "unknown")
        has_proof = bool(d.get("model_solution", ""))
        
        # Skip if already judged
        if qid in existing_results:
            d = existing_results[qid]
            print(f"  [{i+1}/{len(lines)}] {qid:20s} ({level:12s}) → {d['judge_points']}/7 (cached)")
            results.append(d)
            continue
        
        print(f"  [{i+1}/{len(lines)}] {qid:20s} ({level:12s})", end=" ", flush=True)
        if not has_proof:
            print("→ 0/7 (no proof)")
            d["judge_points"] = 0
            d["judge_response"] = "No proof generated"
        else:
            t0 = time.time()
            result = judge_one(client, prompt_template, d)
            elapsed = time.time() - t0
            d["judge_points"] = result["points"]
            d["judge_response"] = result["judge_response"]
            d["judge_tokens"] = result["judge_tokens"]
            print(f"→ {result['points']}/7 ({elapsed:.0f}s)")
        results.append(d)
        
        # Save after EVERY problem (incremental save)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Final save to persistent dir
    os.makedirs(PERSIST_DIR, exist_ok=True)
    persist_path = os.path.join(PERSIST_DIR, os.path.basename(output_path))
    with open(persist_path, "w") as f:
        for d in results:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"  Saved to {persist_path}")
    
    return results

def compute_stats(results, model_name):
    """Compute per-difficulty and overall stats"""
    by_level = defaultdict(list)
    all_scores = []
    
    for d in results:
        pts = d.get("judge_points", 0)
        if pts < 0: pts = 0  # treat parse failures as 0
        level = d.get("level", "unknown")
        by_level[level].append(pts)
        all_scores.append(pts)
    
    total = sum(all_scores)
    n = len(all_scores)
    avg = total / n if n > 0 else 0
    pct = avg / 7 * 100
    
    stats = {
        "model": model_name,
        "n_problems": n,
        "total_score": total,
        "max_possible": n * 7,
        "avg_score": round(avg, 2),
        "score_pct": round(pct, 1),
        "by_level": {},
    }
    
    for level in ["pre-IMO", "IMO-easy", "IMO-medium", "IMO-hard"]:
        scores = by_level.get(level, [])
        if scores:
            lavg = sum(scores) / len(scores)
            stats["by_level"][level] = {
                "n": len(scores),
                "avg": round(lavg, 2),
                "pct": round(lavg / 7 * 100, 1),
                "scores": scores,
            }
    
    return stats

def print_comparison(all_stats):
    print(f"\n{'='*80}")
    print(f"  FINAL COMPARISON (GPT-5.2 as Judge)")
    print(f"{'='*80}")
    
    # Header
    print(f"\n{'Model':<25} {'Score%':>8} {'Avg/7':>8} {'Total':>8} {'Proofs':>8}")
    print("-" * 60)
    
    for s in all_stats:
        n_proofs = sum(1 for d in s.get("_results", []) if d.get("model_solution", ""))
        print(f"{s['model']:<25} {s['score_pct']:>7.1f}% {s['avg_score']:>7.2f} {s['total_score']:>5}/{s['max_possible']:<3} {n_proofs:>5}/{s['n_problems']}")
    
    # By difficulty
    print(f"\n{'--- By Difficulty ---':^60}")
    levels = ["pre-IMO", "IMO-easy", "IMO-medium", "IMO-hard"]
    
    print(f"\n{'Level':<15}", end="")
    for s in all_stats:
        print(f" {s['model']:>20}", end="")
    print()
    print("-" * (15 + 21 * len(all_stats)))
    
    for level in levels:
        print(f"{level:<15}", end="")
        for s in all_stats:
            bl = s["by_level"].get(level, {})
            if bl:
                print(f" {bl['pct']:>18.1f}%", end="")
            else:
                print(f" {'N/A':>19}", end="")
        print()
    
    # Paper comparison
    print(f"\n{'--- vs Paper (IMO-ProofBench, avg@3, 60 problems) ---':^60}")
    paper = {"qwen3-4b-thinking": 20.4, "qed-nano-sft": 39.5, "qed-nano": 40.0}
    print(f"\n{'Model':<25} {'Our Score%':>10} {'Paper Score%':>12} {'Note':>15}")
    print("-" * 65)
    for s in all_stats:
        p = paper.get(s["model"], "N/A")
        note = ""
        if isinstance(p, float):
            diff = s["score_pct"] - p
            note = f"({diff:+.1f}%)"
        print(f"{s['model']:<25} {s['score_pct']:>9.1f}% {str(p):>11}% {note:>15}")

def main():
    key = os.environ.get("AZURE_OPENAI_KEY")
    if not key:
        print("ERROR: Set AZURE_OPENAI_KEY environment variable")
        sys.exit(1)
    
    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=key,
    )
    
    prompt_template = load_prompt()
    all_stats = []
    
    for model_name, input_path in MODELS.items():
        if not os.path.exists(input_path):
            print(f"Skipping {model_name}: {input_path} not found")
            continue
        
        results = judge_model(client, prompt_template, model_name, input_path)
        stats = compute_stats(results, model_name)
        stats["_results"] = results
        all_stats.append(stats)
    
    print_comparison(all_stats)
    
    # Save comparison summary
    summary = [{k: v for k, v in s.items() if k != "_results"} for s in all_stats]
    summary_path = os.path.join(PERSIST_DIR, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()
