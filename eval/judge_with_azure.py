"""
Judge QED-Nano eval results using Azure OpenAI GPT-5.2
Usage:
  export AZURE_OPENAI_KEY="your-key"
  python judge_with_azure.py --input outputs/qed-nano_20q.jsonl --output outputs/qed-nano_20q_judged.jsonl
"""
import json, os, argparse, time
from openai import AzureOpenAI

# Azure OpenAI config
ENDPOINT = "https://msncompanioneu2.cognitiveservices.azure.com/"
DEPLOYMENT = "gpt-5-2"
API_VERSION = "2024-12-01-preview"

# Load grading prompt (from repo)
PROMPT_PATH = "configs/prompts/proofbench.txt"

def load_prompt():
    with open(PROMPT_PATH) as f:
        return f.read()

def judge_one(client, prompt_template, problem, model_solution, grading_guidelines, solution=""):
    """Judge a single proof using GPT-5.2"""
    prompt = prompt_template.replace("{problem_statement}", problem)
    prompt = prompt.replace("{guidelines}", grading_guidelines)
    prompt = prompt.replace("{student_answer}", model_solution)
    prompt = prompt.replace("{solution}", solution)

    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=16384,
            model=DEPLOYMENT,
            # GPT-5.2 is a reasoning model, temperature is fixed at 1
        )
        content = response.choices[0].message.content
        
        # Extract points from XML <points>X</points> or <points>X out of 7</points>
        import re
        match = re.search(r'<points>\s*(\d+)', content)
        points = int(match.group(1)) if match else -1
        
        return {
            "points": points,
            "judge_response": content,
            "judge_tokens": response.usage.total_tokens if response.usage else 0,
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {"points": -1, "judge_response": str(e), "judge_tokens": 0}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input JSONL from eval run")
    parser.add_argument("--output", required=True, help="Output JSONL with judge scores")
    args = parser.parse_args()

    key = os.environ.get("AZURE_OPENAI_KEY")
    if not key:
        print("ERROR: Set AZURE_OPENAI_KEY environment variable")
        return

    client = AzureOpenAI(
        api_version=API_VERSION,
        azure_endpoint=ENDPOINT,
        api_key=key,
    )

    prompt_template = load_prompt()

    with open(args.input) as f:
        lines = f.readlines()

    results = []
    total_points = 0
    total_judged = 0

    for i, line in enumerate(lines):
        d = json.loads(line)
        qid = d["question_id"]
        level = d.get("level", "unknown")
        model_sol = d.get("model_solution", "")

        if not model_sol:
            print(f"[{i+1}/{len(lines)}] {qid} ({level}): SKIPPED (no proof)")
            d["judge_points"] = 0
            d["judge_response"] = "No proof generated"
            results.append(d)
            continue

        print(f"[{i+1}/{len(lines)}] {qid} ({level}): judging...", end=" ", flush=True)
        t0 = time.time()

        result = judge_one(
            client, prompt_template,
            problem=d["problem"],
            model_solution=model_sol,
            grading_guidelines=d.get("grading_guidelines", ""),
            solution=d.get("solution", ""),
        )

        elapsed = time.time() - t0
        print(f"{result['points']}/7 ({elapsed:.0f}s)")

        d["judge_points"] = result["points"]
        d["judge_response"] = result["judge_response"]
        d["judge_tokens"] = result["judge_tokens"]
        results.append(d)

        if result["points"] >= 0:
            total_points += result["points"]
            total_judged += 1

    # Save results
    with open(args.output, "w") as f:
        for d in results:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Judged: {total_judged}/{len(lines)}")
    if total_judged > 0:
        avg = total_points / total_judged
        pct = avg / 7 * 100
        print(f"Average score: {avg:.2f}/7 ({pct:.1f}%)")
    print(f"Saved to {args.output}")

    # Also save to persistent dir
    persist_dir = "/scratch/azureml/cr/j/05c921e0efaa4264b68084c90beb0eaf/cap/data-capability/wd/INPUT_msndni/shares/UDC/RnR/Users/yuhangbai/QEDEval"
    os.makedirs(persist_dir, exist_ok=True)
    persist_file = os.path.join(persist_dir, os.path.basename(args.output))
    with open(persist_file, "w") as f:
        for d in results:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"Also saved to {persist_file}")

if __name__ == "__main__":
    main()
