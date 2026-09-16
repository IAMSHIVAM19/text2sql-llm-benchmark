"""QLoRA fine-tuning script for Qwen2.5-Coder-1.5B-Instruct on Spider Text-to-SQL."""

import argparse
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for Text-to-SQL")
    parser.add_argument("--train_file", type=str, default="data/processed/spider_instruct/train.jsonl")
    parser.add_argument("--val_file", type=str, default="data/processed/spider_instruct/val.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/checkpoints/qwen_spider_qlora")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--smoke_test", action="store_true", help="Run 5 steps only for sanity check")
    return parser.parse_args()

def main():
    args = parse_args()
    print("=== QLoRA Training Setup ===")
    print(f"Base Model:     {MODEL_ID}")
    print(f"Max Seq Length: {args.max_seq_length}")
    print(f"Batch Size:     {args.batch_size} (Grad Accum: {args.grad_accum}, Effective: {args.batch_size * args.grad_accum})")
    print(f"Learning Rate:  {args.learning_rate}")
    print(f"Smoke Test:     {args.smoke_test}")

    # 1. 4-bit Quantization Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # 2. Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    model = prepare_model_for_kbit_training(model)

    # 3. LoRA Configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    for param in model.parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)

    # 4. Load Datasets
    print(f"Loading data from {args.train_file} and {args.val_file}...")
    dataset = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file})

    max_steps = 5 if args.smoke_test else -1
    save_strategy = "no" if args.smoke_test else "steps"

    # 5. SFTConfig
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=1 if args.smoke_test else 25,
        save_strategy=save_strategy,
        save_steps=100,
        save_total_limit=2,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_length=args.max_seq_length,
        report_to="none",
    )

    # 6. SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if not args.smoke_test else None,
        peft_config=lora_config,
        processing_class=tokenizer,
        args=sft_config,
    )

    for name, param in trainer.model.named_parameters():
        if param.requires_grad and param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float32)

    print("Starting training...")
    trainer.train()

    print(f"Saving fine-tuned LoRA adapters to {args.output_dir}...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training process finished successfully!")

if __name__ == "__main__":
    main()
