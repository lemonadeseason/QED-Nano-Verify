"""Run evaluations using official imobench framework with LB, on our sampled subsets.

Usage:
    VLLM_API_KEY=EMPTY PYTHONPATH=src python run_officialscript.py --bench 20q --output sft-hf_20q_officialscript.jsonl
    VLLM_API_KEY=EMPTY PYTHONPATH=src python run_officialscript.py --bench 50q --output sft-hf_proofbench50_officialscript.jsonl
    VLLM_API_KEY=EMPTY PYTHONPATH=src python run_officialscript.py --bench 100q --output sft-hf_answerbench100_officialscript.jsonl
"""
import yaml, json, time, os, re, random, argparse
from datasets import load_dataset
from collections import defaultdict
from imobench.evaluation import run_bench

SEED = 42
YAML_PATH = 'configs/models/vllm/vllm-lb.yaml'

# ── 20q: hardcoded IDs from IMOProofBench ──
SELECTED_20Q_IDS = [
    'PB-Basic-002', 'PB-Basic-013', 'PB-Basic-017',
    'PB-Basic-001', 'PB-Advanced-019', 'PB-Basic-009', 'PB-Basic-025',
    'PB-Advanced-028', 'PB-Basic-019', 'PB-Advanced-025',
    'PB-Basic-007', 'PB-Advanced-014', 'PB-Basic-028', 'PB-Advanced-020',
    'PB-Advanced-006', 'PB-Advanced-018', 'PB-Advanced-030',
    'PB-Advanced-003', 'PB-Advanced-015', 'PB-Advanced-012',
]

def load_20q():
    ds = load_dataset('lm-provers/IMOProofBench')['train']
    indices = [i for i in range(len(ds)) if ds[i]['question_id'] in SELECTED_20Q_IDS]
    ds_sub = ds.select(indices)
    assert len(ds_sub) == 20
    return ds_sub, 'proofbench_run.txt', False

def load_50q():
    ds = load_dataset('lm-provers/ProofBench')['train']
    groups = defaultdict(list)
    for i in range(len(ds)):
        source = re.sub(r'-\d{4}.*', '', ds[i]['problem_id'])
        groups[source].append(i)
    SAMPLE_COUNTS = {'PUTNAM': 12, 'IMO': 8, 'USAMO': 8, 'EGMO': 8, 'APMO': 7, 'TST': 7}
    random.seed(SEED)
    selected = []
    for source, count in SAMPLE_COUNTS.items():
        pool = groups[source]
        picked = sorted(random.sample(pool, min(count, len(pool))))
        selected.extend(picked)
        print(f'  {source}: sampled {len(picked)}/{len(pool)}')
    selected.sort()
    ds_sub = ds.select(selected)
    return ds_sub, 'proofbench_run.txt', False

def load_100q():
    ds = load_dataset('Hwilner/imo-answerbench')['train']
    groups = defaultdict(list)
    for i in range(len(ds)):
        groups[ds[i]['Category']].append(i)
    random.seed(SEED)
    selected = []
    for cat in sorted(groups.keys()):
        pool = groups[cat]
        picked = sorted(random.sample(pool, min(25, len(pool))))
        selected.extend(picked)
        print(f'  {cat}: sampled {len(picked)}/{len(pool)}')
    selected.sort()
    ds_sub = ds.select(selected)
    return ds_sub, 'answerbench_run.txt', True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', choices=['20q', '50q', '100q'], required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    with open(YAML_PATH) as f:
        model_config = yaml.safe_load(f)

    if args.bench == '20q':
        ds_sub, prompt_file, is_answer = load_20q()
        problem_col = 'problem'
    elif args.bench == '50q':
        ds_sub, prompt_file, is_answer = load_50q()
        problem_col = 'problem'
    elif args.bench == '100q':
        ds_sub, prompt_file, is_answer = load_100q()
        problem_col = 'Problem'

    with open(f'configs/prompts/{prompt_file}') as f:
        prompt = f.read()

    outfile = f'outputs/{args.output}'
    print(f'\n=== {args.bench}: {len(ds_sub)} problems ===')
    print(f'Config: {model_config}')
    print(f'Output: {outfile}')

    questions = list(ds_sub[problem_col])
    t0 = time.time()
    results = run_bench(model_config, prompt, questions, overwrite=True,
        other_params={'output_path': outfile},
        path='vllm/vllm-lb')
    elapsed = time.time() - t0

    # Save results
    os.makedirs('outputs', exist_ok=True)
    with open(outfile, 'w') as fout:
        for i, r in enumerate(results):
            row = dict(ds_sub[i])
            resp = r.get('response', '') or ''
            row['model_solution'] = resp
            row['history'] = r.get('history', [])
            row['cost_run'] = r.get('cost', {})
            row['tokens'] = r['cost'].get('output_tokens', 0) if r.get('cost') else 0
            row['time'] = r['cost'].get('time', 0) if r.get('cost') else 0
            if is_answer:
                # Extract boxed answer
                boxed = re.findall(r'\\boxed\{([^}]*)\}', resp)
                row['extracted_answer'] = boxed[-1] if boxed else None
            fout.write(json.dumps(row, default=str) + '\n')

    completed = sum(1 for r in results if r.get('response'))
    total_tokens = sum(r['cost'].get('output_tokens', 0) for r in results if r.get('cost'))
    print(f'\n=== DONE in {elapsed/60:.1f} min ===')
    print(f'Saved to {outfile}')
    print(f'Completed: {completed}/{len(results)}, Total tokens: {total_tokens:,}')

if __name__ == '__main__':
    main()
