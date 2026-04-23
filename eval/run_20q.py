import yaml, json, time, os
from datasets import load_dataset
from imobench.evaluation import run_bench

SELECTED_IDS = [
    'PB-Basic-002', 'PB-Basic-013', 'PB-Basic-017',
    'PB-Basic-001', 'PB-Advanced-019', 'PB-Basic-009', 'PB-Basic-025', 'PB-Advanced-028', 'PB-Basic-019', 'PB-Advanced-025',
    'PB-Basic-007', 'PB-Advanced-014', 'PB-Basic-028', 'PB-Advanced-020',
    'PB-Advanced-006', 'PB-Advanced-018', 'PB-Advanced-030', 'PB-Advanced-003', 'PB-Advanced-015', 'PB-Advanced-012',
]

MODEL_DIR = os.environ.get('MODEL_DIR', '/scratch/azureml/cr/j/05c921e0efaa4264b68084c90beb0eaf/cap/data-capability/wd/INPUT_msndni/shares/UDC/RnR/Users/yuhangbai/Models')
MODEL_NAME = os.environ.get('MODEL_NAME', 'qed-nano')

with open('configs/models/vllm/vllm-lm-provers-qed-nano.yaml') as f:
    model_config = yaml.safe_load(f)
model_config['max_tokens'] = 229000  # slightly less than server max_model_len to leave room for input tokens

with open('configs/prompts/proofbench_run.txt') as f:
    prompt = f.read()

ds = load_dataset('lm-provers/IMOProofBench')['train']
indices = [i for i in range(len(ds)) if ds[i]['question_id'] in SELECTED_IDS]
ds_sub = ds.select(indices)
print(f'=== {MODEL_NAME}: {len(ds_sub)} problems ===')
print(f'Difficulty: {dict(__import__("collections").Counter(ds_sub["level"]))}')
t0 = time.time()

output_file = f'outputs/{MODEL_NAME}_20q.jsonl'
results = run_bench(model_config, prompt, list(ds_sub['problem']), overwrite=True,
    other_params={'output_path': output_file, 'n': 1, 'model_config_name': MODEL_NAME},
    path='vllm/vllm-lm-provers-qed-nano')

elapsed = time.time() - t0
print(f'\nDone in {elapsed/60:.1f} min')

os.makedirs('outputs', exist_ok=True)
PERSIST_DIR = '/scratch/azureml/cr/j/05c921e0efaa4264b68084c90beb0eaf/cap/data-capability/wd/INPUT_msndni/shares/UDC/RnR/Users/yuhangbai/QEDEval'
os.makedirs(PERSIST_DIR, exist_ok=True)

ds2 = ds_sub.add_column('schema_0', [[{'desc': g, 'title': 'Proof Grade', 'points': 7}] for g in ds_sub['grading_guidelines']])
ds2 = ds2.add_column('model_solution', [r.get('response', '') or '' for r in results])
ds2 = ds2.add_column('history', [r.get('history', []) for r in results])
ds2 = ds2.add_column('cost_run', [r['cost'] for r in results])
ds2.to_json(output_file)

# Also save to persistent directory
persist_file = os.path.join(PERSIST_DIR, os.path.basename(output_file))
ds2.to_json(persist_file)
print(f'Also saved to {persist_file}')

total_tokens = sum(r['cost']['output_tokens'] for r in results)
completed = sum(1 for r in results if r.get('response', ''))
print(f'Completed proofs: {completed}/{len(results)}')
print(f'Total tokens: {total_tokens:,}')
for i, r in enumerate(results):
    resp = r.get('response', '') or ''
    qid = ds_sub[i]['question_id']
    lvl = ds_sub[i]['level']
    print(f'  {qid:20s} ({lvl:12s}): proof={len(resp):5d} chars, tokens={r["cost"]["output_tokens"]:6d}, time={r["cost"]["time"]:.0f}s')
print(f'\nSaved to {output_file}')
