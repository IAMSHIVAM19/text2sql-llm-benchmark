"""Sandboxed SQLite execution engine.

Executes SQL queries with:
1. Strict read-only database connections (file URI mode=ro).
2. Hardware/VM-instruction progress handler timeout watchdog against runaway Cartesian joins.
3. Standardized result set extraction and value normalization.
4. Granular error classification (SYNTAX_ERROR, SCHEMA_ERROR, TIMEOUT, EXECUTION_ERROR).
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import sqlite3
import time
from typing import Any, List, Optional, Tuple


class ExecutionStatus(str, Enum):
    SUCCESS = "SUCCESS"
    SYNTAX_ERROR = "SYNTAX_ERROR"
    SCHEMA_ERROR = "SCHEMA_ERROR"
    TIMEOUT = "TIMEOUT"
    EXECUTION_ERROR = "EXECUTION_ERROR"


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    rows: List[Tuple[Any, ...]]
    execution_time_ms: float
    error_message: Optional[str] = None
    row_count: int = 0


def normalize_val(val: Any) -> Any:
    """Normalize SQLite data values for fair, robust equality comparison."""
    if val is None:
        return None
    if isinstance(val, float):
        # Round floats to avoid IEEE 754 precision mismatches
        return round(val, 4)
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        # Strip leading/trailing whitespace
        s = val.strip()
        # Coerce numeric strings that match integers/floats if needed
        return s
    return val


def normalize_row(row: Tuple[Any, ...]) -> Tuple[Any, ...]:
    """Normalize a single result row tuple."""
    return tuple(normalize_val(x) for x in row)


class SQLiteExecutor:
    """Safely executes queries against SQLite databases with timeouts and isolation."""

    def __init__(self, timeout_seconds: float = 3.0, max_rows: int = 5000):
        self.timeout_seconds = timeout_seconds
        self.max_rows = max_rows

    def execute(self, db_path: Path | str, query: str) -> ExecutionResult:
        """Execute a SQL query against a SQLite database file."""
        db_path = Path(db_path)
        if not db_path.exists():
            return ExecutionResult(
                status=ExecutionStatus.EXECUTION_ERROR,
                rows=[],
                execution_time_ms=0.0,
                error_message=f"Database file does not exist: {db_path}",
            )

        start_time = time.perf_counter()
        uri = f"file:{db_path.resolve()}?mode=ro"

        conn = None
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=self.timeout_seconds)
            cursor = conn.cursor()

            # Set progress handler for VM instruction timeouts (checks every 1,000 opcode cycles)
            start_wall = time.time()
            timeout_limit = self.timeout_seconds

            def progress_watchdog():
                if time.time() - start_wall > timeout_limit:
                    return 1  # Non-zero interrupts execution
                return 0

            conn.set_progress_handler(progress_watchdog, 1000)

            cursor.execute(query)
            raw_rows = cursor.fetchmany(self.max_rows)
            normalized_rows = [normalize_row(r) for r in raw_rows]

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                rows=normalized_rows,
                execution_time_ms=elapsed_ms,
                row_count=len(normalized_rows),
            )

        except sqlite3.OperationalError as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            err_msg = str(e).lower()

            if "interrupt" in err_msg or "interrupted" in err_msg:
                status = ExecutionStatus.TIMEOUT
            elif "no such table" in err_msg or "no such column" in err_msg:
                status = ExecutionStatus.SCHEMA_ERROR
            elif "syntax error" in err_msg or "incomplete input" in err_msg or "near " in err_msg:
                status = ExecutionStatus.SYNTAX_ERROR
            else:
                status = ExecutionStatus.EXECUTION_ERROR

            return ExecutionResult(
                status=status,
                rows=[],
                execution_time_ms=elapsed_ms,
                error_message=str(e),
            )

        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return ExecutionResult(
                status=ExecutionStatus.EXECUTION_ERROR,
                rows=[],
                execution_time_ms=elapsed_ms,
                error_message=str(e),
            )

        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
