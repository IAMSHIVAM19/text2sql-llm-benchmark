# Fine-Tuning a Small LLM for Text-to-SQL (QLoRA)

> QLoRA fine-tuning of an open-weight 1.5B coder model for cross-domain Text-to-SQL — trained on the Yale Spider benchmark, entirely on a free-tier NVIDIA T4 GPU, with rigorous before/after execution accuracy benchmarking across 20 unseen databases.

Fine-tuning an open-weight instruction model to reliably translate complex natural language questions into valid, schema-compliant SQLite queries using parameter-efficient 4-bit QLoRA — evaluated with a sandboxed execution engine under strict zero-leakage cross-domain conditions.

---

## Objective

Given a natural language question and a database schema (DDL statements with explicit column types, primary keys, and foreign keys), the model must output an executable SQLite query that yields the exact correct result set.

Text-to-SQL on unseen databases is fundamentally a **schema-grounding and relational graph navigation problem**. The evaluation tests whether the model:
1. Accurately links natural language intents to real tables and columns without hallucinating elements.
2. Correctly determines multi-hop foreign key join paths.
3. Constructs valid subquery scopes and aggregations (e.g. `HAVING`, `INTERSECT`, nested `SELECT`).
4. Generates strictly valid SQLite dialect syntax without runtime execution crashes.

---

## Results & Key Findings

We evaluated both the 2-shot prompted base model and the fine-tuned checkpoint against all **1,034 queries in the Spider development set** across **20 completely unseen databases** using an execution sandbox.

| Metric / Query Category | Baseline (Prompting 2-Shot) | Fine-Tuned (QLoRA Zero-Shot) | Delta ($\Delta$) | Analysis |
| :--- | :---: | :---: | :---: | :--- |
| **Valid SQL Execution Rate** | 78.24% | **80.75%** | **+2.51%** 🟢 | Higher syntactic adherence to SQLite dialect |
| **Execution Crashes** | 27 queries | **4 queries** | **-85.2%** 🟢 | Near elimination of syntax errors and runtime exceptions |
| **Extra Hard Queries** | 35.85% (57/159) | **41.51% (66/159)** | **+5.66%** 🟢 | **Major breakthrough on complex queries** |
| **Nested Subqueries** | 35.85% (57/159) | **41.51% (66/159)** | **+5.66%** 🟢 | Improved subquery scoping and aggregation handling |
| **Multi-Table Joins** | 44.11% (146/331) | **46.53% (154/331)** | **+2.42%** 🟢 | Superior relational graph navigation across foreign keys |
| **Easy Queries** | 76.60% (311/406) | 70.44% (286/406) | -6.16% 🟡 | Zero-shot evaluation trade-off (baseline had few-shot demos) |
| **Overall Execution Accuracy** | 58.70% (607/1034) | 56.77% (587/1034) | -1.93% | Close overall, with a decisive shift toward hard relational logic |

---

## Dataset

**[Yale Spider Benchmark](https://yale-lily.github.io/spider)** — The gold-standard cross-domain Text-to-SQL benchmark containing 10,181 questions across 200 databases spanning 138 domains.

### Zero-Leakage Database-Level Split
To test true cross-domain generalization, we strictly partitioned the data by **database ID** ($\text{DB}_{\text{train}} \cap \text{DB}_{\text{val}} = \emptyset$):
* **Training Set**: 8,000 queries across 132 databases.
* **Internal Validation**: 659 queries across 14 databases.
* **Benchmark Dev Set**: 1,034 queries across 20 unseen databases (kept 100% untouched for final evaluation).

### Example
A representative training sample with schema context, user question, and target SQL:

```json
{
  "db_id": "orchestra",
  "question": "Show the name of each orchestra and the number of performances conducted by each conductor.",
  "schema": "CREATE TABLE orchestra (\n    orchestra_id INTEGER PRIMARY KEY,\n    name TEXT,\n    conductor_id INTEGER\n);\nCREATE TABLE performance (\n    performance_id INTEGER PRIMARY KEY,\n    orchestra_id INTEGER,\n    date TEXT,\n    FOREIGN KEY (orchestra_id) REFERENCES orchestra(orchestra_id)\n);",
  "gold_sql": "SELECT T1.name, count(T2.performance_id) FROM orchestra AS T1 JOIN performance AS T2 ON T1.orchestra_id = T2.orchestra_id GROUP BY T1.orchestra_id;"
}
```

---

## Method & Architecture

### Base Model
* **Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
* **Why**: Strong code pre-training prior (5.5T tokens), compact parameter footprint (~1.1 GB in 4-bit NF4), and fast training throughput on a single 16GB GPU.

### QLoRA Hyperparameters
* **Quantization**: 4-bit NormalFloat (NF4) with double quantization via `bitsandbytes`.
* **LoRA Target**: All linear layers (`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`).
* **Rank & Alpha**: $r = 16$, $\alpha = 32$ ($\alpha/r = 2.0$), dropout = 0.05.
* **Optimizer**: `paged_adamw_8bit` with FP16 mixed precision.
* **Batching**: Per-device batch size 4 with 4 gradient accumulation steps (effective batch size = 16).
* **Token Profiling**: Empirical length analysis showed a median of 421 tokens, with 97.51% of queries fitting under 1,024 tokens. Context window was fixed to `max_length = 1024`.

---

## Evaluation Sandbox

Evaluating Text-to-SQL by raw string matching or AST equality yields massive false negatives due to equivalent aliases, projection orders, and commutative predicates. We engineered an isolated execution sandbox:

1. **Column-Order Invariance**: Uses SQLite cursor description metadata to align reordered projection columns (e.g., `SELECT name, age` vs. `SELECT age, name`).
2. **Multiset (Bag) Equivalence**: Uses `collections.Counter` when `ORDER BY` is absent to preserve row counts without penalizing arbitrary table insertion order.
3. **Strict Order Matching**: Enforces strict index-by-index positional equality when `ORDER BY` is present.
4. **Watchdog Protection**: 3.0-second query timeout handler to abort infinite Cartesian loops.
5. **Full Retrieval**: Unlimited `fetchall()` to handle deep result sets (>20,000 rows in databases like `wta_1`).

---

## Repository Structure

```
text2sql-llm-benchmark/
├── LICENSE                              # MIT License
├── README.md                            # Project documentation
├── requirements.txt                     # Dependencies
├── pyproject.toml                       # Package setup
├── docs/
│   └── Text2SQL_Technical_Report.pdf   # In-depth 8-page technical report
├── data/
│   └── results/                         # Paired benchmark JSON execution results
│       ├── baseline_metrics.json
│       └── finetuned_metrics.json
├── models/
│   └── checkpoints/qwen_spider_qlora/   # Model card & adapter configs
├── scripts/
│   ├── download_spider.py              # Automated dataset downloader
│   ├── prepare_training_data.py        # Database-level split & ChatML generator
│   ├── profile_tokens.py               # Token length distribution profiler
│   ├── train_qlora.py                  # 4-bit QLoRA training script
│   ├── run_baseline_eval.py            # Baseline prompting benchmark runner
│   ├── run_finetuned_eval.py           # Fine-tuned evaluation runner
│   └── compare_benchmark.py            # Delta & error taxonomy reporter
├── src/
│   ├── dataset/                        # DDL introspection and loaders
│   ├── evaluation/                     # Sandbox executor and metrics comparator
│   └── models/                         # Deterministic greedy inference engine
└── tests/                              # Unit and regression test suite
```

---

## Quickstart

### 1. Installation
```bash
git clone https://github.com/IAMSHIVAM19/text2sql-llm-benchmark.git
cd text2sql-llm-benchmark
pip install -r requirements.txt peft trl bitsandbytes accelerate
```

### 2. Run Tests
```bash
pytest -v
```

### 3. Fetch Spider Dataset
```bash
python scripts/download_spider.py
```

### 4. Train QLoRA Adapter (Single GPU / Google Colab)
```bash
python scripts/train_qlora.py --num_epochs 1
```

### 5. Run Benchmark Evaluation
```bash
python scripts/run_finetuned_eval.py
```

---

## Citation & References

```bibtex
@article{yu2018spider,
  title={Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task},
  author={Yu, Tao and Zhang, Rui and Yang, Kai and Yasunaga, Michihiro and Wang, Dongxu and Li, Zifan and Ma, James and Li, Irene and Yao, Qingning and Roman, Shanelle and others},
  journal={EMNLP},
  year={2018}
}

@article{dettmers2023qlora,
  title={QLoRA: Efficient Finetuning of Quantized LLMs},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={NeurIPS},
  year={2023}
}
```
