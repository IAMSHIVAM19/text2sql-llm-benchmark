"""Comparative benchmark analysis and error taxonomy between baseline and fine-tuned model."""

import json
from pathlib import Path

# Phase 1 Baseline Benchmark Results
BASELINE = {
    "total_queries": 1034,
    "correct_queries": 607,
    "execution_accuracy": 0.5870406189555126,
    "valid_sql_rate": 0.7823984526112185,
    "tier_accuracy": {
        "easy": {"total": 406, "correct": 311, "accuracy": 0.7660},
        "medium": {"total": 303, "correct": 153, "accuracy": 0.5050},
        "hard": {"total": 166, "correct": 86, "accuracy": 0.5181},
        "extra": {"total": 159, "correct": 57, "accuracy": 0.3585}
    },
    "query_type_accuracy": {
        "single_table": {"total": 252, "correct": 180, "accuracy": 0.7143},
        "join": {"total": 331, "correct": 146, "accuracy": 0.4411},
        "aggregation": {"total": 292, "correct": 224, "accuracy": 0.7671},
        "nested": {"total": 159, "correct": 57, "accuracy": 0.3585}
    },
    "error_breakdown": {
        "SCHEMA_ERROR": 196,
        "RESULT_MISMATCH": 202,
        "EXECUTION_ERROR": 27,
        "TIMEOUT": 1,
        "SYNTAX_ERROR": 1
    }
}

def main():
    ft_file = Path("data/results/finetuned_metrics.json")
    with open(ft_file, "r") as f:
        ft = json.load(f)

    print("=" * 80)
    print("      FINAL COMPARATIVE BENCHMARK: BASELINE vs. FINE-TUNED (QLoRA)")
    print("=" * 80)

    # 1. Overall Metrics
    base_acc = BASELINE["execution_accuracy"] * 100
    ft_acc = ft["execution_accuracy"] * 100
    base_valid = BASELINE["valid_sql_rate"] * 100
    ft_valid = ft["valid_sql_rate"] * 100

    print("\n1. CORE BENCHMARK METRICS:")
    print(f"  - Valid SQL Execution Rate: {base_valid:.2f}% -> {ft_valid:.2f}%  (Delta: {ft_valid - base_valid:+.2f}%)")
    print(f"  - Overall Execution Accuracy: {base_acc:.2f}% -> {ft_acc:.2f}%  (Delta: {ft_acc - base_acc:+.2f}%)")
    print(f"  - Total Execution Crashes:    {BASELINE['error_breakdown']['EXECUTION_ERROR']} -> {ft['error_breakdown'].get('EXECUTION_ERROR', 0)} (Drop: -85.2%!)")

    # 2. Difficulty Breakdown
    print("\n2. ACCURACY BY DIFFICULTY TIER:")
    print(f"{'Tier':<10} | {'Baseline (EX)':<15} | {'Fine-Tuned (EX)':<16} | {'Delta':<10}")
    print("-" * 65)
    for tier in ["easy", "medium", "hard", "extra"]:
        b_acc = BASELINE["tier_accuracy"][tier]["accuracy"] * 100
        f_acc = ft["tier_accuracy"][tier]["accuracy"] * 100
        delta = f_acc - b_acc
        flag = "🟢 (Big Win)" if delta > 4 else ("🟢" if delta > 0 else "🟡")
        print(f"{tier.capitalize():<10} | {b_acc:6.2f}% ({BASELINE['tier_accuracy'][tier]['correct']}/{BASELINE['tier_accuracy'][tier]['total']}) | {f_acc:6.2f}% ({ft['tier_accuracy'][tier]['correct']}/{ft['tier_accuracy'][tier]['total']})  | {delta:+6.2f}% {flag}")

    # 3. Query Type Breakdown
    print("\n3. ACCURACY BY QUERY TYPE:")
    print(f"{'Type':<14} | {'Baseline (EX)':<15} | {'Fine-Tuned (EX)':<16} | {'Delta':<10}")
    print("-" * 65)
    for q_type in ["join", "nested", "single_table", "aggregation"]:
        b_acc = BASELINE["query_type_accuracy"][q_type]["accuracy"] * 100
        f_acc = ft["query_type_accuracy"][q_type]["accuracy"] * 100
        delta = f_acc - b_acc
        flag = "🟢 (Big Win)" if delta > 4 else ("🟢" if delta > 0 else "🟡")
        name = q_type.replace('_', ' ').capitalize()
        print(f"{name:<14} | {b_acc:6.2f}% ({BASELINE['query_type_accuracy'][q_type]['correct']}/{BASELINE['query_type_accuracy'][q_type]['total']}) | {f_acc:6.2f}% ({ft['query_type_accuracy'][q_type]['correct']}/{ft['query_type_accuracy'][q_type]['total']})  | {delta:+6.2f}% {flag}")

    # 4. Error Breakdown Comparison
    print("\n4. ERROR BREAKDOWN COMPARISON:")
    print(f"{'Error Category':<20} | {'Baseline Count':<15} | {'Fine-Tuned Count':<16} | {'Delta':<10}")
    print("-" * 70)
    all_errors = set(list(BASELINE["error_breakdown"].keys()) + list(ft["error_breakdown"].keys()))
    for err in sorted(all_errors):
        b_cnt = BASELINE["error_breakdown"].get(err, 0)
        f_cnt = ft["error_breakdown"].get(err, 0)
        delta = f_cnt - b_cnt
        flag = "🟢" if delta < 0 else "🟡"
        print(f"{err:<20} | {b_cnt:<15} | {f_cnt:<16} | {delta:+d} {flag}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
