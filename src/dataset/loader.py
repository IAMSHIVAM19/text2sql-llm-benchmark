"""Spider dataset loader and validator.

Loads questions, gold SQL queries, and database paths from the Spider dataset format.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SpiderItem:
    item_id: int
    db_id: str
    question: str
    gold_sql: str
    db_path: Path
    difficulty: Optional[str] = None  # easy, medium, hard, extra


class SpiderLoader:
    """Loads and validates Spider dataset splits."""

    def __init__(self, spider_root: Path | str):
        self.spider_root = Path(spider_root)
        self.db_root = self.spider_root / "database"

        if not self.spider_root.exists():
            raise FileNotFoundError(f"Spider root not found: {self.spider_root}")

    def load_split(self, split: str = "dev") -> List[SpiderItem]:
        """Load items for a given split ('dev' or 'train')."""
        split_file = self.spider_root / f"{split}.json"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        items: List[SpiderItem] = []
        for idx, entry in enumerate(raw_data):
            db_id = entry["db_id"]
            db_file = self.db_root / db_id / f"{db_id}.sqlite"

            item = SpiderItem(
                item_id=idx,
                db_id=db_id,
                question=entry["question"],
                gold_sql=entry["query"],
                db_path=db_file,
                # Some versions/splits may include difficulty annotations
                difficulty=entry.get("difficulty"),
            )
            items.append(item)

        return items

    def list_available_databases(self) -> List[str]:
        """Return list of all database names present in the database directory."""
        if not self.db_root.exists():
            return []
        return [
            d.name for d in self.db_root.iterdir()
            if d.is_dir() and (d / f"{d.name}.sqlite").exists()
        ]
