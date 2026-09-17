"""Tests for sandboxed SQLite executor, timeouts, and error handling."""

import sqlite3
import pytest
from pathlib import Path
from src.evaluation.executor import ExecutionStatus, SQLiteExecutor, normalize_row


@pytest.fixture
def sample_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "test_exec.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE products (id INT PRIMARY KEY, name TEXT, price REAL);")
    cursor.executemany(
        "INSERT INTO products VALUES (?, ?, ?);",
        [
            (1, "Laptop", 999.99123),
            (2, " Mouse ", 25.5),
            (3, "Keyboard", 75.0),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


def test_successful_execution(sample_db: Path):
    executor = SQLiteExecutor()
    res = executor.execute(sample_db, "SELECT id, name FROM products WHERE id = 1;")
    assert res.status == ExecutionStatus.SUCCESS
    assert res.row_count == 1
    assert res.rows == [(1, "Laptop")]


def test_normalization_floats_and_strings(sample_db: Path):
    executor = SQLiteExecutor()
    res = executor.execute(sample_db, "SELECT name, price FROM products WHERE id = 2;")
    assert res.status == ExecutionStatus.SUCCESS
    # String whitespace stripped (" Mouse " -> "Mouse"), float rounded (25.5)
    assert res.rows[0] == ("Mouse", 25.5)


def test_syntax_error_handling(sample_db: Path):
    executor = SQLiteExecutor()
    res = executor.execute(sample_db, "SELEC * FORM products;")
    assert res.status == ExecutionStatus.SYNTAX_ERROR
    assert res.rows == []
    assert res.error_message is not None


def test_schema_error_handling(sample_db: Path):
    executor = SQLiteExecutor()
    res = executor.execute(sample_db, "SELECT non_existent_col FROM products;")
    assert res.status == ExecutionStatus.SCHEMA_ERROR
    assert res.rows == []


def test_read_only_enforcement(sample_db: Path):
    executor = SQLiteExecutor()
    res = executor.execute(sample_db, "INSERT INTO products VALUES (4, 'Monitor', 200.0);")
    # Read-only connection raises operational error on write attempts
    assert res.status in [ExecutionStatus.EXECUTION_ERROR, ExecutionStatus.SCHEMA_ERROR]


def test_timeout_interruption(sample_db: Path):
    # Set an aggressive timeout of 0.05 seconds and run an intensive Cartesian join
    executor = SQLiteExecutor(timeout_seconds=0.05)
    heavy_query = """
        WITH RECURSIVE cnt(x) AS (
            SELECT 1
            UNION ALL
            SELECT x+1 FROM cnt WHERE x < 10000000
        )
        SELECT a.x FROM cnt a, cnt b, cnt c;
    """
    res = executor.execute(sample_db, heavy_query)
    assert res.status == ExecutionStatus.TIMEOUT
