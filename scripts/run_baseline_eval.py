"""Run prompting-only baseline evaluation on Spider dev set.

Usage:
    python scripts/run_baseline_eval.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --limit 100
"""

import argparse
from pathlib import Path
import sys

from src.dataset.loader import SpiderLoader
from src.dataset.prompt_formatter import PromptFormatter
from src.evaluation.evaluator import SpiderEvaluator
from src.evaluation.executor import SQLiteExecutor
from src.models.inference import QwenInferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Run Text-to-SQL prompting baseline benchmark.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
        help="HuggingFace model repository or local path",
    )
    parser.add_argument(
        "--spider-dir",
        type=Path,
        default=Path("data/raw/spider"),
        help="Path to unzipped Spider directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/results/baseline_metrics.json"),
        help="Destination path for evaluation results JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation samples for fast experimentation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device (cuda, mps, cpu)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit precision via bitsandbytes",
    )
    parser.add_argument(
        "--zero-shot",
        action="store_true",
        help="Disable few-shot demonstrations",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 1: TEXT-TO-SQL PROMPTING BASELINE BENCHMARK")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Spider Directory: {args.spider_dir}")
    print(f"Few-shot mode: {'Disabled (Zero-Shot)' if args.zero_shot else 'Enabled (2-shot)'}")
    print(f"Output File: {args.output}")

    if not args.spider_dir.exists():
        print(f"\n[ERROR] Spider directory not found at: {args.spider_dir}")
        print("Please run: python scripts/download_spider.py")
        sys.exit(1)

    # Initialize prompt formatter
    formatter = PromptFormatter(include_few_shot=not args.zero_shot)
    executor = SQLiteExecutor(timeout_seconds=3.0)

    # Initialize model
    engine = QwenInferenceEngine(
        model_name_or_path=args.model,
        device=args.device,
        load_in_4bit=args.load_in_4bit,
    )

    # Initialize evaluator
    evaluator = SpiderEvaluator(
        spider_root=args.spider_dir,
        executor=executor,
        prompt_formatter=formatter,
    )

    items = evaluator.loader.load_split("dev")
    print(f"Loaded {len(items)} queries from Spider dev set.")

    # Execute evaluation
    metrics = evaluator.evaluate_dataset(
        items=items,
        generate_fn=engine.generate_sql,
        limit=args.limit,
        output_json_path=args.output,
    )

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS: BY DIFFICULTY TIER")
    print("=" * 70)
    print(metrics.summary_table())

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS: BY QUERY TYPE")
    print("=" * 70)
    print(metrics.query_type_table())

    print(f"\nValid SQL Execution Rate: {metrics.valid_sql_rate * 100:.2f}%")
    print(f"Overall Execution Accuracy: {metrics.execution_accuracy * 100:.2f}%")
    print(f"Error Distribution: {metrics.error_breakdown}")
    print(f"Full results saved to: {args.output.resolve()}\n")


if __name__ == "__main__":
    main()
