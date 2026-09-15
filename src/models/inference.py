"""Inference engine for Qwen2.5-Coder and base/fine-tuned models.

Provides deterministic greedy decoding, chat template handling, and VRAM-aware loading.
"""

from typing import List, Optional
import torch


class QwenInferenceEngine:
    """Wrapper for Hugging Face CausalLM generation with greedy decoding."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        device: Optional[str] = None,
        load_in_4bit: bool = False,
        adapter_path: Optional[str] = None,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device

        print(f"Loading tokenizer: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading model on {self.device} (4-bit={load_in_4bit})...")
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }

        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        elif self.device in ["cuda", "mps"]:
            model_kwargs["device_map"] = self.device

        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)

        if adapter_path:
            from peft import PeftModel
            print(f"Loading LoRA adapter from: {adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)

        self.model.eval()

    def generate_sql(self, prompt_chatml: str, max_new_tokens: int = 256) -> str:
        """Generate SQL query string with deterministic greedy decoding."""
        inputs = self.tokenizer(prompt_chatml, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Deterministic greedy decoding
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.encode("<|im_end|>")[0] if "<|im_end|>" in self.tokenizer.get_vocab() else self.tokenizer.eos_token_id,
            )

        # Slice off input prompt tokens to decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_len:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response.strip()
