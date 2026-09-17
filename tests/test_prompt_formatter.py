"""Tests for ChatML prompt formatting and SQL extraction."""

import pytest
from src.dataset.prompt_formatter import PromptFormatter, SYSTEM_PROMPT


def test_build_user_content():
    formatter = PromptFormatter()
    schema = "CREATE TABLE users (id INT PRIMARY KEY, name TEXT);"
    question = "Find all user names."
    content = formatter.build_user_content(schema, question)

    assert "### SQLite Database Schema:" in content
    assert schema in content
    assert "### Question:" in content
    assert question in content


def test_build_chat_messages_structure():
    # Few-shot enabled
    formatter = PromptFormatter(include_few_shot=True)
    schema = "CREATE TABLE users (id INT PRIMARY KEY, name TEXT);"
    question = "Find all user names."
    messages = formatter.build_chat_messages(schema, question)

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    # 2 few-shot pairs = 4 messages + 1 system + 1 target = 6 messages
    assert len(messages) == 6
    assert messages[-1]["role"] == "user"
    assert question in messages[-1]["content"]


def test_build_chatml_raw_tags():
    formatter = PromptFormatter(include_few_shot=False)
    schema = "CREATE TABLE t (x INT);"
    question = "Select x."
    raw = formatter.build_chatml_raw(schema, question)

    assert "<|im_start|>system" in raw
    assert "<|im_end|>" in raw
    assert "<|im_start|>user" in raw
    assert raw.endswith("<|im_start|>assistant\n")


def test_extract_sql_markdown_fences():
    text = "Here is the query:\n```sql\nSELECT name FROM users WHERE id = 1;\n```\nExplanation..."
    extracted = PromptFormatter.extract_sql(text)
    assert extracted == "SELECT name FROM users WHERE id = 1"


def test_extract_sql_plain_text():
    text = "SELECT count(*) FROM orders WHERE total > 100;"
    extracted = PromptFormatter.extract_sql(text)
    assert extracted == "SELECT count(*) FROM orders WHERE total > 100"


def test_extract_sql_with_clause():
    text = "```sql\nWITH top_sales AS (SELECT * FROM sales) SELECT * FROM top_sales\n```"
    extracted = PromptFormatter.extract_sql(text)
    assert extracted == "WITH top_sales AS (SELECT * FROM sales) SELECT * FROM top_sales"
