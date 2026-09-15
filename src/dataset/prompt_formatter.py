"""Prompt engineering and ChatML formatting for Text-to-SQL generation.

Builds few-shot and zero-shot prompts adhering to Qwen2.5-Coder-1.5B-Instruct ChatML format,
and extracts clean SQL queries from generated outputs.
"""

import re
from typing import Dict, List, Optional


SYSTEM_PROMPT = (
    "You are an expert SQLite assistant. Given the database schema (DDL) and a natural language question, "
    "generate a single valid SQLite query that answers the question. "
    "Do not hallucinate column names or table names not present in the schema. "
    "Output ONLY the SQL query wrapped in a ```sql\n...``` code block."
)

# Canonical domain-agnostic few-shot demonstrations (domain disjoint from Spider dev databases)
FEW_SHOT_DEMONSTRATIONS = [
    {
        "schema": (
            "CREATE TABLE employee (\n"
            "    employee_id INTEGER PRIMARY KEY,\n"
            "    name TEXT,\n"
            "    department TEXT,\n"
            "    salary REAL,\n"
            "    hire_date TEXT\n"
            ");"
        ),
        "question": "What is the average salary of employees in the 'Engineering' department?",
        "sql": "SELECT avg(salary) FROM employee WHERE department = 'Engineering';",
    },
    {
        "schema": (
            "CREATE TABLE customer (\n"
            "    customer_id INTEGER PRIMARY KEY,\n"
            "    customer_name TEXT,\n"
            "    city TEXT\n"
            ");\n\n"
            "CREATE TABLE orders (\n"
            "    order_id INTEGER PRIMARY KEY,\n"
            "    customer_id INTEGER,\n"
            "    order_amount REAL,\n"
            "    order_date TEXT,\n"
            "    FOREIGN KEY (customer_id) REFERENCES customer(customer_id)\n"
            ");"
        ),
        "question": "List all customer names who have spent more than 500 in total across all their orders.",
        "sql": (
            "SELECT T1.customer_name FROM customer AS T1 "
            "JOIN orders AS T2 ON T1.customer_id = T2.customer_id "
            "GROUP BY T1.customer_name HAVING sum(T2.order_amount) > 500;"
        ),
    },
]


class PromptFormatter:
    """Formats prompts for ChatML models like Qwen2.5-Coder."""

    def __init__(self, include_few_shot: bool = True):
        self.include_few_shot = include_few_shot

    def build_user_content(self, schema_ddl: str, question: str) -> str:
        """Compose the user turn containing database schema and the query question."""
        return (
            f"### SQLite Database Schema:\n"
            f"```sql\n{schema_ddl.strip()}\n```\n\n"
            f"### Question:\n"
            f"{question.strip()}\n\n"
            f"### SQL Query:"
        )

    def build_chat_messages(self, schema_ddl: str, question: str) -> List[Dict[str, str]]:
        """Construct conversation message dictionaries for tokenizer.apply_chat_template."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        if self.include_few_shot:
            for demo in FEW_SHOT_DEMONSTRATIONS:
                messages.append({
                    "role": "user",
                    "content": self.build_user_content(demo["schema"], demo["question"]),
                })
                messages.append({
                    "role": "assistant",
                    "content": f"```sql\n{demo['sql']}\n```",
                })

        # Target query
        messages.append({
            "role": "user",
            "content": self.build_user_content(schema_ddl, question),
        })

        return messages

    def build_chatml_raw(self, schema_ddl: str, question: str) -> str:
        """Construct raw ChatML formatted prompt string directly."""
        messages = self.build_chat_messages(schema_ddl, question)
        prompt_parts: List[str] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

        # Open assistant turn
        prompt_parts.append("<|im_start|>assistant\n")
        return "\n".join(prompt_parts)

    @staticmethod
    def extract_sql(raw_output: str) -> str:
        """Extract executable SQL string from model generation text.
        
        Handles:
        - Markdown ```sql ... ``` fences
        - Plain text SQL
        - Post-processing (stripping comments, multiple semicolons, extra whitespace)
        """
        text = raw_output.strip()

        # Check for ```sql ... ``` blocks
        sql_match = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        if sql_match:
            candidate = sql_match.group(1).strip()
        else:
            # Fallback: look for SELECT / WITH statements
            select_match = re.search(r"\b(WITH[\s\S]+?SELECT[\s\S]+|SELECT[\s\S]+)", text, re.IGNORECASE)
            if select_match:
                candidate = select_match.group(1).strip()
            else:
                candidate = text

        # Clean trailing explanation or markdown fences if still attached
        candidate = candidate.split("```")[0].strip()
        # Clean semicolon
        if candidate.endswith(";"):
            candidate = candidate[:-1].strip()

        return candidate
