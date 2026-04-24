"""
Run models on IMO-AnswerBench (subset or full).
Usage:
  # Start vLLM server first, then:
  VLLM_API_KEY=EMPTY MODEL_NAME=qed-nano python run_answerbench.py [--full]
"""
import yaml, json, time, os, csv, sys, random
from imobench.evaluation import run_bench

# === Configuration ===
MODEL_NAME = os.environ.get('MODEL_NAME', 'qed-nano')
DATA_PATH = os.environ.get('ANSWERBENCH_CSV', 'data/answerbench_v2.csv')
FULL_MODE = '--full' in sys.argv
SEED = 42

PERSIST_DIR = '/scratch/azureml/cr/j/05c921e0efaa4264b68084c90beb0eaf/cap/data-capability/wd/INPUT_msndni/shares/UDC/RnR/Users/yuhangbai/QEDEval'

# === Load data ===
with open(DATA_PATH, newline='') as f:
    all_problems = list(csv.DictReader(f))
print(f"Loaded {len(all_problems)} problems from {DATA_PATH}")

# === Sample or use all ===
if FULL_MODE:
    problems = all_problems
    suffix = "400q"
    print(f"Running FULL benchmark: {len(problems)} problems")
else:
    # Stratified sample by Category x Subcategory, ~25% per subcategory (min 1)
    random.seed(SEED)
    problems = []
    for cat in sorted(set(p['Category'] for p in all_problems)):
        cat_rows = [p for p in all_problems if p['Category'] == cat]
        by_sub = {}
        for p in cat_rows:
            by_sub.setdefault(p['Subcategory'], []).append(p)
        cat_sampled = 0
        for sub in sorted(by_sub.keys()):
            n = max(1, round(len(by_sub[sub]) * 0.25))
            picked = random.sample(by_sub[sub], n)
            problems.extend(picked)
            cat_sampled += n
        print(f"  {cat}: sampled {cat_sampled}/{len(cat_rows)}")
    
    suffix = f"{len(problems)}q"
    print(f"Running SUBSET: {len(problems)} problems (stratified by subcategory)")

# === Load model config & prompt ===
with open('configs/models/vllm/vllm-lm-provers-qed-nano.yaml') as f:
    model_config = yaml.safe_load(f)
# Keep max_tokens consistent with official run.py (uses YAML value: 229376)
# Override slightly below server max_model_len to leave room for input tokens
model_config['max_tokens'] = 229000

with open('configs/prompts/answerbench_run.txt') as f:
    prompt = f.read()

# === Run generation ===
questions = [p['Problem'] for p in problems]
question_ids = [p['Problem ID'] for p in problems]
answers = [p['Short Answer'] for p in problems]
categories_list = [p['Category'] for p in problems]

from collections import Counter
print(f"Categories: {dict(Counter(categories_list))}")
t0 = time.time()

output_file = f'outputs/{MODEL_NAME}_answerbench_{suffix}.jsonl'
results = run_bench(model_config, prompt, questions, overwrite=True,
    other_params={'output_path': output_file, 'n': 1, 'model_config_name': MODEL_NAME,
                  'benchmark': 'answerbench'},
    path='vllm/vllm-lm-provers-qed-nano')

elapsed = time.time() - t0
print(f'\nGeneration done in {elapsed/60:.1f} min')

# === Save results ===
os.makedirs('outputs', exist_ok=True)
os.makedirs(PERSIST_DIR, exist_ok=True)

rows_out = []
for i, (p, r) in enumerate(zip(problems, results)):
    row = {
        'question_id': p['Problem ID'],
        'problem': p['Problem'],
        'answer': p['Short Answer'],
        'category': p['Category'],
        'subcategory': p['Subcategory'],
        'source': p['Source'],
        'model_solution': r.get('response', '') or '',
        'cost_run': r['cost'],
    }
    rows_out.append(row)

with open(output_file, 'w') as f:
    for row in rows_out:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

persist_file = os.path.join(PERSIST_DIR, os.path.basename(output_file))
with open(persist_file, 'w') as f:
    for row in rows_out:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
print(f'Saved to {output_file}')
print(f'Also saved to {persist_file}')

# === Summary ===
total_tokens = sum(r['cost']['output_tokens'] for r in results)
completed = sum(1 for r in results if r.get('response', ''))
print(f'Completed: {completed}/{len(results)}')
print(f'Total tokens: {total_tokens:,}')
print(f'Avg tokens/problem: {total_tokens // len(results):,}')
for i, (p, r) in enumerate(zip(problems, results)):
    resp = r.get('response', '') or ''
    # Extract boxed answer
    import re
    boxed = re.findall(r'\\boxed\{([^}]*)\}', resp)
    extracted = boxed[-1] if boxed else '?'
    gold = p['Short Answer']
    print(f'  {p["Problem ID"]:35s} ({p["Category"]:15s}): tokens={r["cost"]["output_tokens"]:6d}, answer={extracted[:30]}, gold={gold[:30]}')
