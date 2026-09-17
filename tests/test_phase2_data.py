"""Tests for Phase 2 data loading, zero-leakage splitting, and ChatML formatting."""

import json
from pathlib import Path
import pytest
from src.dataset.loader import SpiderLoader


def test_database_split_zero_leakage():
    # Test on synthetic Spider items to verify strict disjointness
    from src.dataset.loader import SpiderItem
    items = [
        SpiderItem(0, "db_a", "q1", "sql1", Path("."), "easy"),
        SpiderItem(1, "db_a", "q2", "sql2", Path("."), "easy"),
        SpiderItem(2, "db_b", "q3", "sql3", Path("."), "medium"),
        SpiderItem(3, "db_c", "q4", "sql4", Path("."), "hard"),
        SpiderItem(4, "db_d", "q5", "sql5", Path("."), "extra"),
    ]
    train_part, val_part = SpiderLoader.split_by_database(items, val_ratio=0.25, seed=42)

    train_dbs = {it.db_id for it in train_part}
    val_dbs = {it.db_id for it in val_part}

    assert len(train_dbs.intersection(val_dbs)) == 0, "Leakage detected between train and val!"
