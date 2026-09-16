---
base_model: Qwen/Qwen2.5-Coder-1.5B-Instruct
library_name: peft
model_name: qwen_spider_qlora
tags:
- base_model:adapter:Qwen/Qwen2.5-Coder-1.5B-Instruct
- lora
- qlora
- text-to-sql
- spider-benchmark
- sqlite
pipeline_tag: text-generation
license: mit
---

# Qwen2.5-Coder-1.5B Spider Text-to-SQL LoRA Adapter

This repository contains the fine-tuned LoRA adapter weights for **Qwen2.5-Coder-1.5B-Instruct**, trained specifically on the cross-domain **Yale Spider Text-to-SQL dataset** using 4-bit QLoRA.

## Model Summary
* **Base Model**: `Qwen/Qwen2.5-Coder-1.5B-Instruct`
* **Adapter Method**: QLoRA ($r=16, \alpha=32$, target: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`)
* **Training Data**: 8,000 instruction-formatted Text-to-SQL pairs across 132 disjoint SQLite databases
* **Evaluation Result**: +5.66% gain on Extra Hard queries, +5.66% on Nested queries, and an 85.2% reduction in SQL execution runtime crashes over the 2-shot base model.

## Quickstart Inference

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
adapter_path = "models/checkpoints/qwen_spider_qlora"

# 1. Load base model in 4-bit NF4
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

# 2. Attach fine-tuned LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# 3. Text-to-SQL Inference
schema_ddl = """
CREATE TABLE departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT
);
CREATE TABLE employees (
    emp_id INTEGER PRIMARY KEY,
    name TEXT,
    salary REAL,
    dept_id INTEGER,
    FOREIGN KEY (dept_id) REFERENCES departments(dept_id)
);
"""
question = "Find the name of each department that has an average salary above 80,000."

messages = [
    {"role": "system", "content": "You are an expert SQLite assistant. Output only the SQL query."},
    {"role": "user", "content": f"### Schema:\n{schema_ddl}\n\n### Question:\n{question}\n\n### SQL Query:"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print("Generated SQL:\n", response.strip())
```
