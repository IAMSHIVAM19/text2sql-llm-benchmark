"""SQLite schema extractor and DDL serializer.

Provides programmatic extraction of tables, columns, primary keys, and foreign keys
from SQLite databases, producing standardized DDL representations for LLM prompts.
"""

from dataclasses import dataclass, field
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class ColumnInfo:
    cid: int
    name: str
    data_type: str
    not_null: bool
    default_value: Optional[str]
    is_primary_key: bool


@dataclass
class ForeignKeyInfo:
    from_column: str
    to_table: str
    to_column: str


@dataclass
class TableInfo:
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    foreign_keys: List[ForeignKeyInfo] = field(default_factory=list)

    @property
    def primary_keys(self) -> List[str]:
        return [c.name for c in self.columns if c.is_primary_key]


class SchemaExtractor:
    """Introspects a SQLite database file and extracts structured schema information."""

    # SQLite internal tables to exclude
    IGNORED_TABLES = {"sqlite_sequence", "sqlite_stat1", "sqlite_master"}

    @classmethod
    def extract_from_db(cls, db_path: Path | str) -> Dict[str, TableInfo]:
        """Extract schema directly from a SQLite database file."""
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found at: {db_path}")

        # Connect read-only
        uri = f"file:{db_path.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        cursor = conn.cursor()

        tables: Dict[str, TableInfo] = {}

        try:
            # 1. Fetch all user tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            raw_tables = [row[0] for row in cursor.fetchall() if row[0] not in cls.IGNORED_TABLES]

            for table_name in raw_tables:
                # 2. Fetch columns via PRAGMA table_info
                cursor.execute(f"PRAGMA table_info(`{table_name}`);")
                col_rows = cursor.fetchall()

                columns = [
                    ColumnInfo(
                        cid=row[0],
                        name=row[1],
                        data_type=row[2].upper() if row[2] else "TEXT",
                        not_null=bool(row[3]),
                        default_value=row[4],
                        is_primary_key=bool(row[5]),
                    )
                    for row in col_rows
                ]

                # 3. Fetch foreign keys via PRAGMA foreign_key_list
                cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
                fk_rows = cursor.fetchall()
                foreign_keys = [
                    ForeignKeyInfo(
                        from_column=row[3],
                        to_table=row[2],
                        to_column=row[4] if row[4] else row[3],
                    )
                    for row in fk_rows
                    if row[3] is not None
                ]

                tables[table_name] = TableInfo(
                    name=table_name,
                    columns=columns,
                    foreign_keys=foreign_keys,
                )

        finally:
            conn.close()

        return tables

    @classmethod
    def to_ddl(cls, tables: Dict[str, TableInfo]) -> str:
        """Serialize extracted TableInfo dict into standardized, clean DDL statements."""
        ddl_statements: List[str] = []

        for table_name, table in tables.items():
            lines: List[str] = []
            pk_cols = table.primary_keys

            # Single primary key inline or composite PK at the bottom
            has_single_pk = len(pk_cols) == 1

            for col in table.columns:
                col_type = col.data_type or "TEXT"
                col_line = f"    {col.name} {col_type}"
                if has_single_pk and col.is_primary_key:
                    col_line += " PRIMARY KEY"
                lines.append(col_line)

            # Composite primary keys
            if len(pk_cols) > 1:
                pk_str = ", ".join(pk_cols)
                lines.append(f"    PRIMARY KEY ({pk_str})")

            # Foreign keys
            for fk in table.foreign_keys:
                lines.append(f"    FOREIGN KEY ({fk.from_column}) REFERENCES {fk.to_table}({fk.to_column})")

            table_body = ",\n".join(lines)
            ddl_statements.append(f"CREATE TABLE {table_name} (\n{table_body}\n);")

        return "\n\n".join(ddl_statements)
