"""Execution accuracy metrics and Spider difficulty classification.

Implements:
1. Set-equality (multiset Counter) and order-sensitive execution matching.
2. Query type tagging (Single-table, Join, Nested/Subquery, Aggregation).
3. Spider difficulty scoring (easy, medium, hard, extra).
4. Aggregated reporting tables.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple
from tabulate import tabulate

from src.evaluation.executor import ExecutionResult, ExecutionStatus


def is_order_sensitive(sql: str) -> bool:
    """Check whether a SQL query enforces row ordering via ORDER BY."""
    return bool(re.search(r"\border\s+by\b", sql, re.IGNORECASE))


def classify_query_type(sql: str) -> str:
    """Classify SQL query into semantic categories:
    - nested: contains subqueries (IN (SELECT...), etc.)
    - join: contains multiple tables via JOIN or Cartesian FROM
    - aggregation: contains aggregate functions or GROUP BY
    - single_table: simple single table query
    """
    sql_upper = sql.upper()

    # Check nested first: more than one SELECT keyword
    select_count = len(re.findall(r"\bSELECT\b", sql_upper))
    if select_count > 1:
        return "nested"

    # Check join: explicit JOIN or multiple comma-separated tables in FROM before WHERE/GROUP/ORDER
    if " JOIN " in sql_upper:
        return "join"
    from_match = re.search(r"\bFROM\s+([^;]+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|$)", sql_upper)
    if from_match and "," in from_match.group(1):
        return "join"

    # Check aggregation
    agg_keywords = ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX(", "GROUP BY", "HAVING"]
    if any(k in sql_upper for k in agg_keywords):
        return "aggregation"

    return "single_table"


def estimate_spider_difficulty(sql: str) -> str:
    """Estimate Spider difficulty tier (easy, medium, hard, extra) based on SQL complexity."""
    sql_upper = sql.upper()
    select_count = len(re.findall(r"\bSELECT\b", sql_upper))

    # Extra: nested queries or set operations
    has_set_op = any(op in sql_upper for op in [" UNION ", " EXCEPT ", " INTERSECT "])
    if select_count > 1 or has_set_op:
        return "extra"

    has_join = " JOIN " in sql_upper or (
        bool(re.search(r"\bFROM\s+[^,;]+,[^,;]+", sql_upper))
    )
    has_group = "GROUP BY" in sql_upper
    has_order = "ORDER BY" in sql_upper
    has_agg = any(fn in sql_upper for fn in ["COUNT(", "SUM(", "AVG(", "MIN(", "MAX("])
    where_count = len(re.findall(r"\b(AND|OR)\b", sql_upper))

    complexity_score = 0
    if has_join:
        complexity_score += 2
    if has_group:
        complexity_score += 1
    if has_order:
        complexity_score += 1
    if has_agg:
        complexity_score += 1
    complexity_score += where_count

    if complexity_score <= 1:
        return "easy"
    elif complexity_score <= 3:
        return "medium"
    else:
        return "hard"


def compare_results(
    gold_rows: List[Tuple[Any, ...]],
    pred_rows: List[Tuple[Any, ...]],
    gold_sql: str,
) -> bool:
    """Compare gold and predicted execution result sets.
    
    - If gold has ORDER BY: enforces strict positional list equality.
    - Otherwise: enforces multiset (bag) equality via Counter.
    """
    if is_order_sensitive(gold_sql):
        return gold_rows == pred_rows
    else:
        return Counter(gold_rows) == Counter(pred_rows)


@dataclass
class EvaluationSampleResult:
    item_id: int
    db_id: str
    question: str
    gold_sql: str
    predicted_sql: str
    difficulty: str
    query_type: str
    is_correct: bool
    gold_status: ExecutionStatus
    pred_status: ExecutionStatus
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class BenchmarkMetrics:
    total_queries: int = 0
    correct_queries: int = 0
    valid_sql_queries: int = 0
    execution_accuracy: float = 0.0
    valid_sql_rate: float = 0.0

    tier_accuracy: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    query_type_accuracy: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    error_breakdown: Dict[str, int] = field(default_factory=dict)

    def summary_table(self) -> str:
        """Render a clean summary table string."""
        headers = ["Difficulty Tier", "Total", "Correct", "Execution Accuracy (%)"]
        table_data = []

        tiers = ["easy", "medium", "hard", "extra"]
        for tier in tiers:
            stats = self.tier_accuracy.get(tier, {"total": 0, "correct": 0, "accuracy": 0.0})
            table_data.append([
                tier.capitalize(),
                stats["total"],
                stats["correct"],
                f"{stats['accuracy'] * 100:.2f}%",
            ])

        table_data.append([
            "OVERALL",
            self.total_queries,
            self.correct_queries,
            f"{self.execution_accuracy * 100:.2f}%",
        ])

        return tabulate(table_data, headers=headers, tablefmt="github")

    def query_type_table(self) -> str:
        """Render a query-type breakdown table string."""
        headers = ["Query Type", "Total", "Correct", "Execution Accuracy (%)"]
        table_data = []

        types = ["single_table", "join", "aggregation", "nested"]
        for qtype in types:
            stats = self.query_type_accuracy.get(qtype, {"total": 0, "correct": 0, "accuracy": 0.0})
            table_data.append([
                qtype.replace("_", " ").capitalize(),
                stats["total"],
                stats["correct"],
                f"{stats['accuracy'] * 100:.2f}%",
            ])

        return tabulate(table_data, headers=headers, tablefmt="github")


class MetricsAggregator:
    """Aggregates sample evaluation results into comprehensive benchmark metrics."""

    @staticmethod
    def compute_metrics(sample_results: List[EvaluationSampleResult]) -> BenchmarkMetrics:
        if not sample_results:
            return BenchmarkMetrics()

        total = len(sample_results)
        correct = sum(1 for s in sample_results if s.is_correct)
        valid_sql = sum(1 for s in sample_results if s.pred_status == ExecutionStatus.SUCCESS)

        # Breakdown by tier
        tier_counts = defaultdict(lambda: {"total": 0, "correct": 0})
        for s in sample_results:
            tier_counts[s.difficulty]["total"] += 1
            if s.is_correct:
                tier_counts[s.difficulty]["correct"] += 1

        tier_metrics = {}
        for tier, counts in tier_counts.items():
            tot = counts["total"]
            corr = counts["correct"]
            tier_metrics[tier] = {
                "total": tot,
                "correct": corr,
                "accuracy": (corr / tot) if tot > 0 else 0.0,
            }

        # Breakdown by query type
        type_counts = defaultdict(lambda: {"total": 0, "correct": 0})
        for s in sample_results:
            type_counts[s.query_type]["total"] += 1
            if s.is_correct:
                type_counts[s.query_type]["correct"] += 1

        type_metrics = {}
        for qtype, counts in type_counts.items():
            tot = counts["total"]
            corr = counts["correct"]
            type_metrics[qtype] = {
                "total": tot,
                "correct": corr,
                "accuracy": (corr / tot) if tot > 0 else 0.0,
            }

        # Error breakdown
        error_dist = defaultdict(int)
        for s in sample_results:
            if not s.is_correct:
                if s.pred_status != ExecutionStatus.SUCCESS:
                    error_dist[s.pred_status.value] += 1
                else:
                    error_dist["RESULT_MISMATCH"] += 1

        return BenchmarkMetrics(
            total_queries=total,
            correct_queries=correct,
            valid_sql_queries=valid_sql,
            execution_accuracy=(correct / total) if total > 0 else 0.0,
            valid_sql_rate=(valid_sql / total) if total > 0 else 0.0,
            tier_accuracy=tier_metrics,
            query_type_accuracy=type_metrics,
            error_breakdown=dict(error_dist),
        )
