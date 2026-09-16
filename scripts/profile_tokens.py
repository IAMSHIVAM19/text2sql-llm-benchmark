"""Profile token lengths for Spider instruction dataset using Qwen2.5 tokenizer."""

import json
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

def main():
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    data_file = Path("data/processed/spider_instruct/train.jsonl")
    if not data_file.exists():
        raise FileNotFoundError(f"Missing {data_file}")

    total_lengths = []
    question_lengths = []
    sql_lengths = []

    print("Profiling training samples...")
    with open(data_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            sample = json.loads(line)
            messages = sample["messages"]

            full_text = tokenizer.apply_chat_template(messages, tokenize=False)
            total_tokens = len(tokenizer.encode(full_text))
            total_lengths.append(total_tokens)

            question_tokens = len(tokenizer.encode(sample["question"]))
            sql_tokens = len(tokenizer.encode(sample["gold_sql"]))
            question_lengths.append(question_tokens)
            sql_lengths.append(sql_tokens)

            if (idx + 1) % 2000 == 0:
                print(f"  Processed {idx + 1} samples...")

    print("\n=== Token Length Profiling Results (ChatML Total) ===")
    print(f"Total Samples: {len(total_lengths)}")
    print(f"Min:    {np.min(total_lengths)}")
    print(f"Mean:   {np.mean(total_lengths):.1f}")
    print(f"Median: {np.median(total_lengths):.1f}")
    print(f"p90:    {np.percentile(total_lengths, 90):.1f}")
    print(f"p95:    {np.percentile(total_lengths, 95):.1f}")
    print(f"p99:    {np.percentile(total_lengths, 99):.1f}")
    print(f"Max:    {np.max(total_lengths)}")

    print("\n=== Coverage by max_seq_length Thresholds ===")
    for limit in [512, 1024, 1536, 2048]:
        pct = (np.array(total_lengths) <= limit).mean() * 100
        print(f"<= {limit:4d} tokens: {pct:6.2f}% of samples fit completely")

if __name__ == "__main__":
    main()
