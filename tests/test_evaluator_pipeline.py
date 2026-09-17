"""Integration test for end-to-end SpiderEvaluator pipeline."""

import sqlite3
from pathlib import Path
import pytest

from src.dataset.loader import SpiderItem
from src.dataset.prompt_formatter import PromptFormatter
from src.evaluation.evaluator import SpiderEvaluator
from src.evaluation.executor import SQLiteExecutor


@pytest.fixture
def mock_spider_env(tmp_path: Path):
    """Create a minimal mock Spider database directory."""
    db_root = tmp_path / "database" / "stadium"
    db_root.mkdir(parents=True)
    db_path = db_root / "stadium.sqlite"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE stadium (
            stadium_id INTEGER PRIMARY KEY,
            name TEXT,
            capacity INTEGER
        );
    """)
    cursor.executemany(
        "INSERT INTO stadium VALUES (?, ?, ?);",
        [
            (1, "Wembley", 90000),
            (2, "Camp Nou", 99000),
            (3, "Old Trafford", 74000),
        ],
    )
    conn.commit()
    conn.close()

    # Create dummy dev.json
    dev_json = tmp_path / "dev.json"
    import json
    with open(dev_json, "w") as f:
        json.dump(
            [
                {
                    "db_id": "stadium",
                    "question": "What is the name of the stadium with the largest capacity?",
                    "query": "SELECT name FROM stadium ORDER BY capacity DESC LIMIT 1;",
                },
                {
                    "db_id": "stadium",
                    "question": "How many stadiums have capacity greater than 80000?",
                    "query": "SELECT count(*) FROM stadium WHERE capacity > 80000;",
                },
            ],
            f,
        )

    return tmp_path


def test_evaluator_end_to_end(mock_spider_env: Path, tmp_path: Path):
    evaluator = SpiderEvaluator(
        spider_root=mock_spider_env,
        executor=SQLiteExecutor(timeout_seconds=2.0),
        prompt_formatter=PromptFormatter(include_few_shot=False),
    )

    items = evaluator.loader.load_split("dev")
    assert len(items) == 2

    # Mock model: answers first query correctly, second query with syntax error
    def mock_model(prompt: str) -> str:
        if "largest capacity" in prompt:
            return "```sql\nSELECT name FROM stadium ORDER BY capacity DESC LIMIT 1\n```"
        else:
            return "```sql\nSELECT count(*) FROM WHERE capacity > 80000\n```"

    output_json = tmp_path / "results.json"
    metrics = evaluator.evaluate_dataset(
        items=items,
        generate_fn=mock_model,
        output_json_path=output_json,
    )

    assert metrics.total_queries == 2
    assert metrics.correct_queries == 1
    assert metrics.valid_sql_queries == 1
    assert metrics.execution_accuracy == 0.5
    assert metrics.valid_sql_rate == 0.5
    assert output_json.exists()

    # Verify JSON content
    import json
    with open(output_json) as f:
        data = json.load(f)
    assert "summary" in data
    assert "samples" in data
    assert len(data["samples"]) == 2
    assert data["samples"][0]["is_correct"] is True
    assert data["samples"][1]["is_correct"] is False
