"""Evaluation orchestrator for the Spider Text-to-SQL benchmark.

Coordinates schema introspection, prompt generation, model inference,
execution comparison, and detailed metrics logging.
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from tqdm import tqdm

from src.dataset.loader import SpiderItem, SpiderLoader
from src.dataset.prompt_formatter import PromptFormatter
from src.dataset.schema import SchemaExtractor
from src.evaluation.executor import ExecutionResult, ExecutionStatus, SQLiteExecutor
from src.evaluation.metrics import (
    BenchmarkMetrics,
    EvaluationSampleResult,
    MetricsAggregator,
    classify_query_type,
    compare_results,
    estimate_spider_difficulty,
)


class SpiderEvaluator:
    """End-to-end evaluation pipeline for Text-to-SQL models on Spider."""

    def __init__(
        self,
        spider_root: Path | str,
        executor: Optional[SQLiteExecutor] = None,
        prompt_formatter: Optional[PromptFormatter] = None,
    ):
        self.spider_root = Path(spider_root)
        self.loader = SpiderLoader(self.spider_root)
        self.executor = executor or SQLiteExecutor(timeout_seconds=3.0)
        self.prompt_formatter = prompt_formatter or PromptFormatter(include_few_shot=True)
        self.schema_cache: Dict[str, str] = {}

    def get_schema_ddl(self, db_id: str, db_path: Path) -> str:
        """Fetch cached DDL or introspect SQLite DB."""
        if db_id in self.schema_cache:
            return self.schema_cache[db_id]

        tables = SchemaExtractor.extract_from_db(db_path)
        ddl = SchemaExtractor.to_ddl(tables)
        self.schema_cache[db_id] = ddl
        return ddl

    def evaluate_sample(
        self,
        item: SpiderItem,
        generate_fn: Callable[[str], str],
    ) -> EvaluationSampleResult:
        """Evaluate a single Spider query item."""
        schema_ddl = self.get_schema_ddl(item.db_id, item.db_path)
        prompt_str = self.prompt_formatter.build_chatml_raw(schema_ddl, item.question)

        # 1. Model inference
        raw_response = generate_fn(prompt_str)
        predicted_sql = self.prompt_formatter.extract_sql(raw_response)

        # 2. Execute Gold SQL
        gold_result = self.executor.execute(item.db_path, item.gold_sql)

        # 3. Execute Predicted SQL
        pred_result = self.executor.execute(item.db_path, predicted_sql)

        # 4. Compare results
        is_correct = False
        error_msg = None

        if pred_result.status != ExecutionStatus.SUCCESS:
            error_msg = f"{pred_result.status.value}: {pred_result.error_message}"
        elif gold_result.status != ExecutionStatus.SUCCESS:
            error_msg = f"GOLD_SQL_ERROR: {gold_result.error_message}"
        else:
            is_correct = compare_results(gold_result.rows, pred_result.rows, item.gold_sql)
            if not is_correct:
                error_msg = "RESULT_SET_MISMATCH"

        difficulty = item.difficulty or estimate_spider_difficulty(item.gold_sql)
        query_type = classify_query_type(item.gold_sql)

        return EvaluationSampleResult(
            item_id=item.item_id,
            db_id=item.db_id,
            question=item.question,
            gold_sql=item.gold_sql,
            predicted_sql=predicted_sql,
            difficulty=difficulty,
            query_type=query_type,
            is_correct=is_correct,
            gold_status=gold_result.status,
            pred_status=pred_result.status,
            error_message=error_msg,
            execution_time_ms=pred_result.execution_time_ms,
        )

    def evaluate_dataset(
        self,
        items: List[SpiderItem],
        generate_fn: Callable[[str], str],
        limit: Optional[int] = None,
        output_json_path: Optional[Path | str] = None,
    ) -> BenchmarkMetrics:
        """Evaluate a dataset split and export rich report."""
        if limit:
            items = items[:limit]

        sample_results: List[EvaluationSampleResult] = []

        print(f"\nStarting benchmark evaluation across {len(items)} queries...")
        for item in tqdm(items, desc="Evaluating Text-to-SQL"):
            sample_res = self.evaluate_sample(item, generate_fn)
            sample_results.append(sample_res)

        metrics = MetricsAggregator.compute_metrics(sample_results)

        if output_json_path:
            out_path = Path(output_json_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            serializable_samples = [
                {
                    "item_id": s.item_id,
                    "db_id": s.db_id,
                    "question": s.question,
                    "gold_sql": s.gold_sql,
                    "predicted_sql": s.predicted_sql,
                    "difficulty": s.difficulty,
                    "query_type": s.query_type,
                    "is_correct": s.is_correct,
                    "gold_status": s.gold_status.value,
                    "pred_status": s.pred_status.value,
                    "error_message": s.error_message,
                    "execution_time_ms": s.execution_time_ms,
                }
                for s in sample_results
            ]

            full_payload = {
                "summary": {
                    "total_queries": metrics.total_queries,
                    "correct_queries": metrics.correct_queries,
                    "valid_sql_queries": metrics.valid_sql_queries,
                    "execution_accuracy": metrics.execution_accuracy,
                    "valid_sql_rate": metrics.valid_sql_rate,
                    "tier_accuracy": metrics.tier_accuracy,
                    "query_type_accuracy": metrics.query_type_accuracy,
                    "error_breakdown": metrics.error_breakdown,
                },
                "samples": serializable_samples,
            }

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(full_payload, f, indent=2)
            print(f"Saved comprehensive evaluation artifact to: {out_path}")

        return metrics
