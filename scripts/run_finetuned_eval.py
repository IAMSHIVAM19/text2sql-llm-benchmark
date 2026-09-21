"""Run comparative evaluation of the fine-tuned QLoRA model on Spider dev set.

Features:
- Loads fine-tuned LoRA adapter on 4-bit base model
- Auto-saves progress every 20 queries
- Can resume from where it left off if interrupted
- Generates comprehensive difficulty & query-type metric tables
"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
from tqdm import tqdm

from src.dataset.loader import SpiderLoader, SpiderItem
from src.dataset.prompt_formatter import PromptFormatter
from src.evaluation.evaluator import SpiderEvaluator
from src.evaluation.executor import SQLiteExecutor
from src.evaluation.metrics import MetricsAggregator, EvaluationSampleResult
from src.models.inference import QwenInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Evaluate Fine-Tuned QLoRA model on Spider dev benchmark.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="Base model repository or path",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="models/checkpoints/qwen_spider_qlora",
        help="Path to fine-tuned LoRA checkpoint directory",
    )
    parser.add_argument(
        "--spider-dir",
        type=Path,
        default=Path("data/raw/spider"),
        help="Path to Spider directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/finetuned_metrics.json"),
        help="Destination path for evaluation results JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation samples",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 4: FINE-TUNED MODEL COMPARATIVE BENCHMARK")
    print("=" * 70)
    print(f"Base Model:       {args.model}")
    print(f"LoRA Adapter:     {args.adapter}")
    print(f"Spider Directory: {args.spider_dir}")
    print(f"Output File:      {args.output}")

    if not args.spider_dir.exists():
        print(f"[ERROR] Spider directory not found at: {args.spider_dir}")
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial_file = args.output.parent / f"{args.output.stem}_partial.json"

    # Initialize model with LoRA adapter attached
    print("\nLoading base model + fine-tuned LoRA adapter on T4 GPU...")
    engine = QwenInferenceEngine(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        load_in_4bit=True,
    )

    # Formatter & Evaluator (zero-shot to match training instruction distribution)
    formatter = PromptFormatter(include_few_shot=False)
    executor = SQLiteExecutor(timeout_seconds=3.0)
    evaluator = SpiderEvaluator(
        spider_root=args.spider_dir,
        executor=executor,
        prompt_formatter=formatter,
    )

    items = evaluator.loader.load_split("dev")
    if args.limit:
        items = items[:args.limit]
    print(f"Total queries to evaluate: {len(items)}")

    # Check for existing partial progress to resume
    completed_results = []
    completed_ids = set()
    if partial_file.exists():
        try:
            with open(partial_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                for entry in saved_data:
                    res = EvaluationSampleResult(**entry)
                    completed_results.append(res)
                    completed_ids.add(res.item_id)
            print(f"Resuming from partial checkpoint: {len(completed_results)} queries already evaluated!")
        except Exception as e:
            print(f"Could not load partial checkpoint ({e}), starting fresh.")
            completed_results = []
            completed_ids = set()

    # Evaluate remaining items
    remaining_items = [it for it in items if it.item_id not in completed_ids]
    print(f"Evaluating remaining {len(remaining_items)} queries...")

    for idx, item in enumerate(tqdm(remaining_items, desc="Evaluating Text-to-SQL")):
        sample_res = evaluator.evaluate_sample(item, engine.generate_sql)
        completed_results.append(sample_res)

        # Auto-save checkpoint every 20 queries
        if (idx + 1) % 20 == 0 or (idx + 1) == len(remaining_items):
            with open(partial_file, "w", encoding="utf-8") as f:
                json.dump([asdict(res) for res in completed_results], f, indent=2)

    # Compute final aggregate metrics
    metrics = MetricsAggregator.compute_metrics(completed_results)

    # Save final complete artifact
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(asdict(metrics) if hasattr(metrics, '__dataclass_fields__') else metrics.__dict__, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK RESULTS: BY DIFFICULTY TIER")
    print("=" * 70)
    print(metrics.summary_table())

    print("\n" + "=" * 70)
    print("FINAL BENCHMARK RESULTS: BY QUERY TYPE")
    print("=" * 70)
    print(metrics.query_type_table())

    print(f"\nValid SQL Execution Rate:   {metrics.valid_sql_rate * 100:.2f}%")
    print(f"Overall Execution Accuracy: {metrics.execution_accuracy * 100:.2f}%")
    print(f"Error Distribution:         {metrics.error_breakdown}")
    print(f"Full results saved to:      {args.output.resolve()}\n")


if __name__ == "__main__":
    main()
