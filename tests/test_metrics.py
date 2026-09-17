"""Tests for execution comparison, query type classification, and metrics aggregation."""

import pytest
from src.evaluation.executor import ExecutionStatus
from src.evaluation.metrics import (
    BenchmarkMetrics,
    EvaluationSampleResult,
    MetricsAggregator,
    classify_query_type,
    compare_results,
    estimate_spider_difficulty,
    is_order_sensitive,
)


def test_is_order_sensitive():
    assert is_order_sensitive("SELECT name FROM student ORDER BY age DESC") is True
    assert is_order_sensitive("SELECT name FROM student WHERE age > 20") is False


def test_compare_results_unordered_match():
    gold_sql = "SELECT id, name FROM users;"
    gold_rows = [(1, "Alice"), (2, "Bob")]
    pred_rows = [(2, "Bob"), (1, "Alice")]

    # Unordered query should match even if permutation differs
    assert compare_results(gold_rows, pred_rows, gold_sql) is True


def test_compare_results_ordered_strictness():
    gold_sql = "SELECT id, name FROM users ORDER BY id ASC;"
    gold_rows = [(1, "Alice"), (2, "Bob")]
    pred_rows = [(2, "Bob"), (1, "Alice")]

    # Ordered query must strictly match row sequence
    assert compare_results(gold_rows, pred_rows, gold_sql) is False


def test_compare_results_multiset_multiplicity():
    gold_sql = "SELECT dept_id FROM employees;"
    gold_rows = [(10,), (10,), (20,)]
    pred_rows = [(10,), (20,)]

    # Missing duplicate row should fail multiset equality
    assert compare_results(gold_rows, pred_rows, gold_sql) is False


def test_classify_query_type():
    single = "SELECT name, age FROM students WHERE age > 18;"
    assert classify_query_type(single) == "single_table"

    join_query = "SELECT s.name, c.title FROM students s JOIN enrollments e ON s.id = e.sid JOIN courses c ON e.cid = c.id;"
    assert classify_query_type(join_query) == "join"

    nested_query = "SELECT name FROM students WHERE id IN (SELECT student_id FROM honors);"
    assert classify_query_type(nested_query) == "nested"

    agg_query = "SELECT dept, count(*), avg(salary) FROM emp GROUP BY dept;"
    assert classify_query_type(agg_query) == "aggregation"


def test_estimate_spider_difficulty():
    easy = "SELECT name FROM stadium WHERE capacity > 5000;"
    assert estimate_spider_difficulty(easy) == "easy"

    medium = "SELECT count(*), max(capacity) FROM stadium GROUP BY location;"
    assert estimate_spider_difficulty(medium) in ["medium", "hard"]

    extra = "SELECT name FROM singer WHERE id NOT IN (SELECT singer_id FROM song) UNION SELECT name FROM artist;"
    assert estimate_spider_difficulty(extra) == "extra"


def test_metrics_aggregator():
    sample1 = EvaluationSampleResult(
        item_id=0,
        db_id="db1",
        question="q1",
        gold_sql="SELECT * FROM t1;",
        predicted_sql="SELECT * FROM t1;",
        difficulty="easy",
        query_type="single_table",
        is_correct=True,
        gold_status=ExecutionStatus.SUCCESS,
        pred_status=ExecutionStatus.SUCCESS,
    )
    sample2 = EvaluationSampleResult(
        item_id=1,
        db_id="db1",
        question="q2",
        gold_sql="SELECT * FROM t2;",
        predicted_sql="SELECT * FROM wrong;",
        difficulty="easy",
        query_type="single_table",
        is_correct=False,
        gold_status=ExecutionStatus.SUCCESS,
        pred_status=ExecutionStatus.SCHEMA_ERROR,
        error_message="no such table: wrong",
    )
    sample3 = EvaluationSampleResult(
        item_id=2,
        db_id="db2",
        question="q3",
        gold_sql="SELECT a FROM t3 JOIN t4;",
        predicted_sql="SELECT a FROM t3 JOIN t4;",
        difficulty="hard",
        query_type="join",
        is_correct=True,
        gold_status=ExecutionStatus.SUCCESS,
        pred_status=ExecutionStatus.SUCCESS,
    )

    metrics = MetricsAggregator.compute_metrics([sample1, sample2, sample3])

    assert metrics.total_queries == 3
    assert metrics.correct_queries == 2
    assert metrics.valid_sql_queries == 2
    assert pytest.approx(metrics.execution_accuracy, 0.01) == 2 / 3
    assert pytest.approx(metrics.valid_sql_rate, 0.01) == 2 / 3

    assert metrics.tier_accuracy["easy"]["total"] == 2
    assert metrics.tier_accuracy["easy"]["correct"] == 1
    assert pytest.approx(metrics.tier_accuracy["easy"]["accuracy"], 0.01) == 0.5

    assert metrics.tier_accuracy["hard"]["total"] == 1
    assert metrics.tier_accuracy["hard"]["correct"] == 1

    assert "SCHEMA_ERROR" in metrics.error_breakdown
    assert metrics.error_breakdown["SCHEMA_ERROR"] == 1

    summary_tbl = metrics.summary_table()
    assert "Easy" in summary_tbl
    assert "OVERALL" in summary_tbl
