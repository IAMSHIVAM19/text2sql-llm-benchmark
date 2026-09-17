"""Tests for SQLite schema extraction and DDL serialization."""

import sqlite3
import pytest
from pathlib import Path
from src.dataset.schema import SchemaExtractor


@pytest.fixture
def sample_sqlite_db(tmp_path: Path) -> Path:
    """Create a temporary SQLite database with known tables and foreign keys."""
    db_path = tmp_path / "test_school.sqlite"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE departments (
            dept_id INTEGER PRIMARY KEY,
            dept_name TEXT NOT NULL
        );

        CREATE TABLE professors (
            prof_id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            dept_id INTEGER,
            salary REAL,
            FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
        );

        CREATE TABLE courses (
            course_id TEXT,
            sec_id INTEGER,
            title TEXT,
            prof_id INTEGER,
            PRIMARY KEY (course_id, sec_id),
            FOREIGN KEY (prof_id) REFERENCES professors(prof_id)
        );
    """)
    conn.commit()
    conn.close()
    return db_path


def test_schema_extraction_tables(sample_sqlite_db: Path):
    tables = SchemaExtractor.extract_from_db(sample_sqlite_db)
    assert set(tables.keys()) == {"departments", "professors", "courses"}


def test_schema_columns_and_types(sample_sqlite_db: Path):
    tables = SchemaExtractor.extract_from_db(sample_sqlite_db)
    prof_table = tables["professors"]

    col_names = [c.name for c in prof_table.columns]
    assert col_names == ["prof_id", "name", "dept_id", "salary"]

    salary_col = next(c for c in prof_table.columns if c.name == "salary")
    assert salary_col.data_type == "REAL"
    assert not salary_col.is_primary_key

    prof_id_col = next(c for c in prof_table.columns if c.name == "prof_id")
    assert prof_id_col.is_primary_key


def test_foreign_key_extraction(sample_sqlite_db: Path):
    tables = SchemaExtractor.extract_from_db(sample_sqlite_db)
    prof_table = tables["professors"]
    assert len(prof_table.foreign_keys) == 1

    fk = prof_table.foreign_keys[0]
    assert fk.from_column == "dept_id"
    assert fk.to_table == "departments"
    assert fk.to_column == "dept_id"


def test_ddl_serialization(sample_sqlite_db: Path):
    tables = SchemaExtractor.extract_from_db(sample_sqlite_db)
    ddl = SchemaExtractor.to_ddl(tables)

    assert "CREATE TABLE departments" in ddl
    assert "dept_id INTEGER PRIMARY KEY" in ddl
    assert "CREATE TABLE professors" in ddl
    assert "FOREIGN KEY (dept_id) REFERENCES departments(dept_id)" in ddl
    assert "CREATE TABLE courses" in ddl
    assert "PRIMARY KEY (course_id, sec_id)" in ddl
