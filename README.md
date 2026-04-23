<div align="center">
    <h1><img height="150px" src="./images/logo.png" alt="QED Nano"><br>QED-Nano-Verify</h1>

  <a href="https://www.python.org/">
<img alt="Build" src="https://img.shields.io/badge/Python-3.12-1f425f.svg?color=blue">
  </a>
  <a href="https://opensource.org/licenses/Apache-2.0">
<img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-green.svg">
  </a>
  <a href="https://huggingface.co/lm-provers">
<img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-lm--provers-ffc107?color=ffc107&logoColor=white">
  </a>
</div>

# QED-Nano-Verify: Independent Reproduction of QED-Nano Results

This is a fork of [CMU-AIRe/QED-Nano](https://github.com/CMU-AIRe/QED-Nano) with additional scripts for **independently reproducing and verifying** the paper's evaluation results on IMO-ProofBench.

> **Paper**: [QED-Nano: Teaching a Tiny Model to Prove Hard Theorems](https://arxiv.org/abs/2604.04898) (arXiv:2604.04898)

## Goal

Verify two key claims from the paper:
1. **QED-Nano (RL+RC) outperforms its SFT-only ablation** on IMO-ProofBench (40.0% vs 39.5%)
2. **QED-Nano matches or exceeds the base Qwen3-4B-Thinking model** (40.0% vs 20.4%)

---

## Quick Start: End-to-End Evaluation

### Prerequisites

- 8x NVIDIA A100 80GB (or equivalent, ~640GB total GPU memory)
- Python 3.12, conda
- Access to an OpenAI-compatible judge model (we use Azure OpenAI GPT-5.2)

### Step 1: Environment Setup

```bash
conda create -n qed-eval python=3.12 -y
conda activate qed-eval
pip install vllm==0.11.2
cd eval && pip install -e .
```

### Step 2: Download Models

```bash
# All three models are ~7.6GB each
huggingface-cli download lm-provers/QED-Nano --local-dir Models/QED-Nano
huggingface-cli download Qwen/Qwen3-4B-Thinking-2507 --local-dir Models/Qwen3-4B-Thinking-2507
huggingface-cli download lm-provers/QED-Nano-SFT --local-dir Models/QED-Nano-SFT
```

### Step 3: Start vLLM Server

```bash
export VLLM_API_KEY=EMPTY

# For QED-Nano (swap model path for other models)
python -m vllm.entrypoints.openai.api_server \
  --model Models/QED-Nano \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --max-model-len 229376 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code
```

### Step 4: Generate Proofs

Two options:

**Option A — 20-problem representative subset (fast, ~70 min per model):**
```bash
cd eval
# Set MODEL_NAME to match the model you're serving
MODEL_NAME=qed-nano python run_20q.py
MODEL_NAME=qwen3-4b-thinking python run_20q.py
MODEL_NAME=qed-nano-sft python run_20q.py
```

**Option B — Full 60-problem IMO-ProofBench (paper's setup):**
```bash
cd eval
python scripts/run.py \
  --model-config vllm/vllm-lm-provers-qed-nano \
  --output-path outputs/qed-nano_full.jsonl
```

### Step 5: Judge with LLM

```bash
export AZURE_OPENAI_KEY="your-key-here"
cd eval
python judge_and_compare.py
```

This scores all three models and prints a comparison table.

---

## Evaluation Scripts

All custom scripts are under `eval/`:

| Script | Purpose |
|--------|---------|
| `eval/run_20q.py` | Run a 20-problem representative subset across difficulty levels |
| `eval/judge_and_compare.py` | Judge all 3 models with GPT-5.2, produce comparison table |
| `eval/judge_with_azure.py` | Judge a single model's output with Azure OpenAI |
| `eval/scripts/run.py` | Original full-benchmark generation (from upstream repo) |
| `eval/scripts/eval.py` | Original grading script using any judge model |
| `eval/scripts/grading.py` | Grading bench evaluation |
| `eval/scripts/stats.py` | Compute statistics from graded outputs |

### Key Config Files

| File | Purpose |
|------|---------|
| `eval/configs/models/vllm/vllm-lm-provers-qed-nano.yaml` | vLLM model config (API endpoint, max_tokens, etc.) |
| `eval/configs/models/azure-gpt5.yaml` | Azure OpenAI GPT-5.2 judge config |
| `eval/configs/prompts/proofbench_run.txt` | Proof generation prompt |
| `eval/configs/prompts/proofbench.txt` | IMO-style grading rubric prompt (0-7 scale) |
| `eval/configs/agents/reasoning_cache.yaml` | RSA (Reasoning Cache) agent config |

### Pipeline Overview

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. vLLM Server  │────▶│  2. Generate      │────▶│  3. Judge         │
│  (serve model)   │     │  (run_20q.py or   │     │  (judge_and_      │
│                  │     │   scripts/run.py) │     │   compare.py)    │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                                          │
                                                          ▼
                                                  ┌──────────────────┐
                                                  │  Comparison      │
                                                  │  Table + Stats   │
                                                  └──────────────────┘
```

**Generation** → Each problem is sent to vLLM with the proof prompt. The model generates a long-form mathematical proof (up to 229K tokens). Results are saved as JSONL.

**Judging** → Each proof is evaluated by GPT-5.2 using the IMO grading rubric (0-7 scale). The rubric includes problem-specific grading guidelines.

**Comparison** → Scores are aggregated by model and difficulty level (pre-IMO, IMO-easy, IMO-medium, IMO-hard) and compared to paper-reported values.

---

## Differences from Paper

| Aspect | Paper | Our Setup |
|--------|-------|-----------|
| Judge model | Gemini 2.5 Pro | GPT-5.2 |
| Samples per problem | avg@3 | 1 |
| Problem count | 60 (full) | 20 or 60 |
| vLLM version | not specified | 0.11.2 |
| Hardware | not specified | 8x A100 80GB |

---

## Original Repo Info

### Quick Links

- **Model**: [lm-provers/QED-Nano](https://huggingface.co/lm-provers/QED-Nano) on Hugging Face
- **Blog Post**: [QED-Nano: Teaching a Tiny Model to Prove Hard Theorems](https://huggingface.co/spaces/lm-provers/qed-nano-blogpost)

### Training Data

- **SFT Data**: [lm-provers/FineProofs-SFT](https://huggingface.co/datasets/lm-provers/FineProofs-SFT)
- **RL Data**: [lm-provers/FineProofs-RL](https://huggingface.co/datasets/lm-provers/FineProofs-RL)

### Repository Structure

- **`data/`** - Data generation scripts and SLURM configurations for creating SFT and RL training datasets
- **`training/`** - Training code for supervised fine-tuning (SFT) and reinforcement learning (RL) with reasoning cache
- **`eval/`** - Evaluation code for benchmarking models on IMOProofBench, IMOAnswerBench, and ProofBench

### Upstream Documentation

- [Data README](data/README.md)
- [Training README](training/README.md)
- [Evaluation README](eval/README.md)

## Citation

```bibtex
@misc{qednano2026,
  title        = {QED-Nano: Teaching a Tiny Model to Prove Hard Theorems},
  author       = {LM-Provers and Yuxiao Qu and Amrith Setlur and Jasper Dekoninck and Edward Beeching and Jia Li and Ian Wu and Lewis Tunstall and Aviral Kumar},
  year         = {2026},
  howpublished = {https://huggingface.co/spaces/lm-provers/qed-nano-blogpost},
  note         = {Blog post}
}
```


