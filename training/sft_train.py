"""
QED-Nano SFT Training Script

Reproduces the SFT stage from the QED-Nano paper (arXiv:2604.04898).
Uses TRL SFTTrainer with the exact hyperparameters from the model card.

Base model:  Qwen/Qwen3-4B-Thinking-2507
Dataset:     lm-provers/FineProofs-SFT (4,300 unique problems)
Framework:   TRL SFTTrainer + DeepSpeed ZeRO-3
Hardware:    8x A100 80GB (paper used 8x H100)

Usage:
  # Single GPU (debug)
  python sft_train.py --debug

  # 8-GPU with DeepSpeed ZeRO-3 (full training)
  accelerate launch --config_file accelerate_ds3.yaml sft_train.py

  # With wandb
  accelerate launch --config_file accelerate_ds3.yaml sft_train.py --wandb_project qed-nano-verify
"""

import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig


# Persistent storage paths (Cosmos on AML)
COSMOS_BASE = os.environ.get(
    "COSMOS_BASE",
    "/scratch/azureml/cr/j/05c921e0efaa4264b68084c90beb0eaf/cap/data-capability/wd/INPUT_msndni/shares/UDC/RnR/Users/yuhangbai"
)
DEFAULT_MODEL_PATH = os.path.join(COSMOS_BASE, "Models/Qwen3-4B-Thinking-2507")
# Checkpoints saved locally (fast SSD) during training, then copied to Cosmos at end
DEFAULT_OUTPUT_DIR = "/scratch/azureml/cr/j/05c921e0efaa4264b68084c90beb0eaf/exe/wd/sft_output"
COSMOS_OUTPUT_DIR = os.path.join(COSMOS_BASE, "Models/QED-Nano-SFT-Repro")

# Paper hyperparameters (from model card + paper Appendix C)
PAPER_CONFIG = {
    "max_steps": 620,                    # 5 epochs over 4,300 examples, batch=32 → ~672 steps, they used 620
    "per_device_train_batch_size": 1,    # per GPU (paper=2 on H100, we use 1 on A100 80GB)
    "gradient_accumulation_steps": 4,    # effective batch = 1 * 4 * 8 GPUs = 32 (same as paper)
    "learning_rate": 3e-5,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.03,               # model card says 0.03, paper says 10% — we follow model card
    "max_length": 32768,                 # paper=45056 on H100, reduced for A100 80GB OOM
    "bf16": True,
    "gradient_checkpointing": True,
}


def parse_args():
    parser = argparse.ArgumentParser(description="QED-Nano SFT Training")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to base model (local or HF hub)")
    parser.add_argument("--dataset", type=str, default="lm-provers/FineProofs-SFT",
                        help="HF dataset name or local path")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for checkpoints")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Wandb project name (enables wandb logging)")
    parser.add_argument("--wandb_run_name", type=str, default="qed-nano-sft-repro",
                        help="Wandb run name")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: 1 GPU, 10 steps, small batch")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max_steps (default: 620)")
    parser.add_argument("--save_steps", type=int, default=62,
                        help="Save checkpoint every N steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Wandb setup ---
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        report_to = "wandb"
    else:
        report_to = "none"

    # --- Load dataset ---
    print(f"Loading dataset: {args.dataset}")
    if os.path.exists(args.dataset):
        dataset = load_dataset("json", data_files=args.dataset, split="train")
    else:
        dataset = load_dataset(args.dataset, split="train")
    print(f"Dataset size: {len(dataset)} examples")

    # --- Load tokenizer ---
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Training config ---
    max_steps = args.max_steps or PAPER_CONFIG["max_steps"]
    per_device_batch = PAPER_CONFIG["per_device_train_batch_size"]
    grad_accum = PAPER_CONFIG["gradient_accumulation_steps"]

    if args.debug:
        max_steps = 10
        per_device_batch = 1
        grad_accum = 1
        max_length = 4096
        save_steps = 5
        print(">>> DEBUG MODE: 10 steps, batch=1, max_length=4096")
    else:
        max_length = PAPER_CONFIG["max_length"]
        save_steps = args.save_steps

    # DeepSpeed config path (relative to cwd)
    ds_config = os.path.join(os.path.dirname(__file__), "ds_config_zero3.json")

    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=grad_accum,
        learning_rate=PAPER_CONFIG["learning_rate"],
        lr_scheduler_type=PAPER_CONFIG["lr_scheduler_type"],
        warmup_ratio=PAPER_CONFIG["warmup_ratio"],
        max_length=max_length,
        bf16=PAPER_CONFIG["bf16"],
        gradient_checkpointing=PAPER_CONFIG["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        deepspeed=ds_config if not args.debug else None,
        logging_steps=1,
        save_steps=save_steps,
        save_total_limit=5,
        report_to=report_to,
        run_name=args.wandb_run_name,
        seed=42,
        dataloader_num_workers=4,
        remove_unused_columns=True,
        dataset_text_field=None,  # use 'messages' field with chat template
        packing=False,  # paper doesn't mention packing for SFT
    )

    # --- Load model ---
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype="bfloat16",
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # --- Train ---
    print(f"\n{'='*60}")
    print(f"  QED-Nano SFT Training")
    print(f"  Base model: {args.model_path}")
    print(f"  Dataset: {args.dataset} ({len(dataset)} examples)")
    print(f"  Max steps: {max_steps}")
    print(f"  Effective batch size: {per_device_batch * grad_accum} × num_gpus")
    print(f"  Learning rate: {PAPER_CONFIG['learning_rate']}")
    print(f"  Max seq length: {max_length}")
    print(f"  Output: {args.output_dir}")
    print(f"  Wandb: {args.wandb_project or 'disabled'}")
    print(f"{'='*60}\n")

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # --- Save final model ---
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    # --- Copy to persistent storage ---
    import shutil
    os.makedirs(COSMOS_OUTPUT_DIR, exist_ok=True)
    print(f"Copying to persistent storage: {COSMOS_OUTPUT_DIR}")
    for f in os.listdir(args.output_dir):
        src = os.path.join(args.output_dir, f)
        dst = os.path.join(COSMOS_OUTPUT_DIR, f)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            print(f"  Copied {f}")
    print(f"Done! Model persisted to {COSMOS_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
