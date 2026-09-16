"""Prepare Spider instruction-tuning dataset for QLoRA fine-tuning.

Converts Spider training data into ChatML formatted JSONL samples with
zero-leakage database-level splitting.
"""

from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict

from src.dataset.loader import SpiderLoader, SpiderItem
from src.dataset.schema import SchemaExtractor
from src.dataset.prompt_formatter import PromptFormatter, SYSTEM_PROMPT


def format_instruction_sample(item: SpiderItem, schema_ddl: str) -> dict:
    """Format a Spider item into standard chat messages for Qwen."""
    formatter = PromptFormatter(include_few_shot=False)
    user_content = formatter.build_user_content(schema_ddl, item.question)
    assistant_content = f"```sql\n{item.gold_sql.strip()}\n```"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    return {
        "item_id": item.item_id,
        "db_id": item.db_id,
        "question": item.question,
        "gold_sql": item.gold_sql,
        "difficulty": item.difficulty,
        "messages": messages,
    }


def main():
    spider_root = Path("data/raw/spider")
    output_dir = Path("data/processed/spider_instruct")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Spider training items...")
    loader = SpiderLoader(spider_root)
    all_train_items = loader.load_split("train", include_others=True)
    print(f"Loaded {len(all_train_items)} raw training items.")

    print("Performing zero-leakage database-level train/val split (10% val)...")
    train_items, val_items = SpiderLoader.split_by_database(all_train_items, val_ratio=0.1, seed=42)

    print("Extracting DDL schemas for all unique databases...")
    unique_db_ids = {item.db_id for item in all_train_items}
    schema_cache: Dict[str, str] = {}

    for db_id in unique_db_ids:
        db_path = spider_root / "database" / db_id / f"{db_id}.sqlite"
        if db_path.exists():
            tables = SchemaExtractor.extract_from_db(db_path)
            schema_cache[db_id] = SchemaExtractor.to_ddl(tables)
        else:
            print(f"Warning: DB not found at {db_path}")
            schema_cache[db_id] = ""

    print(f"Cached schemas for {len(schema_cache)} databases.")

    train_out = output_dir / "train.jsonl"
    with open(train_out, "w", encoding="utf-8") as f:
        for item in train_items:
            sample = format_instruction_sample(item, schema_cache.get(item.db_id, ""))
            f.write(json.dumps(sample) + "\n")

    val_out = output_dir / "val.jsonl"
    with open(val_out, "w", encoding="utf-8") as f:
        for item in val_items:
            sample = format_instruction_sample(item, schema_cache.get(item.db_id, ""))
            f.write(json.dumps(sample) + "\n")

    print(f"Saved {len(train_items)} training samples to: {train_out}")
    print(f"Saved {len(val_items)} validation samples to: {val_out}")
    print("Phase 2 data generation complete!")


if __name__ == "__main__":
    main()
