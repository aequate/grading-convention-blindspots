"""Pinned Qwen3.5 model loading and generation used by the reported evaluation."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import importlib.metadata
import os
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass



def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def object_hash(value: object) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))

MODEL_ID = "Qwen/Qwen3.5-4B"
DEFAULT_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"


@dataclass(frozen=True)
class ExperimentConfig:
    model_id: str = MODEL_ID
    model_revision: str = DEFAULT_REVISION
    quantization: str = "nf4"  # "nf4" or "none"; never silently falls back.
    precision: str = "auto"  # native bfloat16 if supported, otherwise float16.
    seed: int = 42
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    num_beams: int = 1
    repetition_penalty: float = 1.0
    max_new_tokens: int = 512
    max_input_tokens: int = 4096
    enable_thinking: bool = False
    deterministic_algorithms: bool = True

    def validate(self) -> None:
        if self.model_id != MODEL_ID:
            raise ValueError(f"This project implements only {MODEL_ID}. No substitution is allowed.")
        if not self.model_revision.strip():
            raise ValueError("Specify a model revision or branch; it will be resolved to a full SHA.")
        if self.quantization not in {"nf4", "none"}:
            raise ValueError("quantization must be nf4 or none.")
        if self.precision not in {"auto", "float16", "bfloat16"}:
            raise ValueError("precision must be auto, float16, or bfloat16.")
        for field in ("do_sample", "enable_thinking", "deterministic_algorithms"):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be a boolean.")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if self.num_beams != 1:
            raise ValueError("Only single-beam generation is implemented to keep comparisons controlled.")
        if any(type(n) is not int or n <= 0 for n in (self.max_new_tokens, self.max_input_tokens)):
            raise ValueError("Token limits must be positive integers.")
        if not self.repetition_penalty > 0:
            raise ValueError("repetition_penalty must be positive.")
        if self.do_sample:
            if self.temperature is None or not 0 < self.temperature <= 10:
                raise ValueError("Sampling requires an explicit positive temperature (at most 10).")
            if self.top_p is None or not 0 < self.top_p <= 1:
                raise ValueError("Sampling requires top_p in (0, 1].")
            if type(self.top_k) is not int or self.top_k < 0:
                raise ValueError("Sampling requires top_k >= 0 (zero disables top-k filtering).")
        elif any(value is not None for value in (self.temperature, self.top_p, self.top_k)):
            raise ValueError("Greedy decoding requires temperature, top_p and top_k to be None.")
        object_hash(asdict(self))  # Also rejects non-finite floats before files are written.

    def controlled_settings(self) -> dict:
        return asdict(self)


class InferenceOOM(RuntimeError):
    """An out-of-memory failure leaves previously written responses intact."""


def environment_report() -> dict:
    versions = dict(
        sorted(
            {
                d.metadata["Name"]: d.version
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            }.items()
        )
    )
    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "dependency_versions": versions,
        "cuda_available": False,
        "gpus": [],
    }
    try:
        import torch

        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda_version"] = torch.version.cuda
        result["torch_cudnn_version"] = torch.backends.cudnn.version()
        if result["cuda_available"]:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                result["gpus"].append(
                    {
                        "index": index,
                        "name": props.name,
                        "total_memory_bytes": props.total_memory,
                        "compute_capability": [props.major, props.minor],
                    }
                )
            try:
                result["nvidia_driver_versions"] = (
                    subprocess.run(
                        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=10,
                    )
                    .stdout.strip()
                    .splitlines()
                )
            except (OSError, subprocess.SubprocessError) as exc:
                result["driver_report_error"] = str(exc)
    except (ImportError, OSError) as exc:
        result["torch_import_error"] = str(exc)
    return result


def seed_for_case(seed: int, case_id: str) -> int:
    """Order-independent random seed, including when sampling is explicitly enabled."""
    return int(object_hash({"seed": seed, "case_id": case_id})[:8], 16)


def set_reproducibility(seed: int, strict: bool) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(strict, warn_only=False)


def resolve_revision(model_id: str, revision: str) -> str:
    if re.fullmatch(r"[a-f0-9]{40}", revision):
        return revision
    from huggingface_hub import HfApi

    resolved = HfApi().model_info(model_id, revision=revision).sha
    if not resolved or not re.fullmatch(r"[a-f0-9]{40}", resolved):
        raise ValueError("Hugging Face did not provide an immutable model revision.")
    return resolved


def oom_instruction() -> str:
    return (
        "CUDA out of memory. Responses written before this error remain in the output directory. "
        "Free GPU memory and rerun with a fresh output directory. Restart the runtime after a "
        "loading failure. Changes to quantization or token limits define a separate experiment."
    )


class QwenRunner:
    """Full multimodal checkpoint, used with text-only content and official processor API."""

    def __init__(self, model, processor, config: ExperimentConfig, metadata: dict, generation):
        self.model, self.processor = model, processor
        self.config, self.metadata, self.generation = config, metadata, generation
        self.chat_template = processor.chat_template

    def with_config(self, config: ExperimentConfig):
        """Reuse loaded weights for explicitly separated decoding/seed configurations."""
        config.validate()
        allowed = {"seed", "do_sample", "temperature", "top_p", "top_k"}
        original = self.config.controlled_settings()
        target = config.controlled_settings()
        if any(original[key] != target[key] for key in original if key not in allowed):
            raise ValueError("Reusing weights permits only declared sampling settings and seed changes.")
        generation = copy.deepcopy(self.generation)
        generation.do_sample = config.do_sample
        generation.temperature = config.temperature if config.do_sample else 1.0
        generation.top_p = config.top_p if config.do_sample else 1.0
        generation.top_k = config.top_k if config.do_sample else 0
        metadata = copy.deepcopy(self.metadata)
        metadata["effective_generation_config"] = generation.to_dict()
        return QwenRunner(self.model, self.processor, config, metadata, generation)

    def prepare(self, prompt: str) -> dict:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        arguments = {"add_generation_prompt": True, "enable_thinking": self.config.enable_thinking}
        rendered = self.processor.apply_chat_template(messages, tokenize=False, **arguments)
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt", **arguments
        )
        ids = inputs["input_ids"][0].tolist()
        if len(ids) > self.config.max_input_tokens:
            raise ValueError(
                f"Rendered prompt has {len(ids)} tokens, over max_input_tokens="
                f"{self.config.max_input_tokens}. No truncation was applied. Use a new run to change limits."
            )
        context_limit = self.model.config.text_config.max_position_embeddings
        if len(ids) + self.config.max_new_tokens > context_limit:
            raise ValueError("Input plus generation budget exceeds the model context limit.")
        snapshot = {
            "messages": messages,
            "rendered_input": rendered,
            "input_token_ids": ids,
            "model_input_keys": sorted(inputs.keys()),
            "attention_mask": inputs["attention_mask"][0].tolist(),
        }
        return {"snapshot": snapshot, "tensors": inputs}

    def generate(self, prepared: dict, seed: int) -> dict:
        import torch

        set_reproducibility(seed, self.config.deterministic_algorithms)
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        baseline = torch.cuda.memory_allocated(0)
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                inputs = prepared["tensors"].to(self.model.device)
                generated = self.model.generate(**inputs, generation_config=self.generation)
            torch.cuda.synchronize(0)
            elapsed = time.perf_counter() - start
            prefix_length = inputs["input_ids"].shape[-1]
            if generated.sequences[0, :prefix_length].tolist() != inputs["input_ids"][0].tolist():
                raise ValueError("Generated sequence prefix differs from the saved model input.")
            tokens = generated.sequences[0, prefix_length:].tolist()
            raw = self.processor.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            eos = self.generation.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            ended = bool(tokens and tokens[-1] in eos_ids)
            result = {
                "raw_response": raw,
                "generated_token_ids": tokens,
                "generated_tokens": len(tokens),
                "input_tokens": prefix_length,
                "response_sha256": sha256_bytes(raw.encode("utf-8")),
                "latency_seconds": elapsed,
                "latency_scope": "transfer_and_generate_and_synchronize",
                "baseline_gpu_memory_bytes": baseline,
                "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "finish_reason": "eos"
                if ended
                else "token_limit"
                if len(tokens) >= self.config.max_new_tokens
                else "other_stop",
                "token_limit_reached": len(tokens) >= self.config.max_new_tokens,
                "case_seed": seed,
                "enable_thinking": self.config.enable_thinking,
            }
            del generated, inputs
            return result
        except torch.OutOfMemoryError as exc:
            gc.collect()
            torch.cuda.empty_cache()
            raise InferenceOOM(oom_instruction()) from exc


def load_model(config: ExperimentConfig) -> QwenRunner:
    """from_pretrained downloads actual safetensors weights; it does not call a model API."""
    config.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from transformers import (
        AutoConfig,
        AutoProcessor,
        BitsAndBytesConfig,
        GenerationConfig,
        Qwen3_5ForConditionalGeneration,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Enable an NVIDIA GPU in Colab: Runtime > Change runtime type > GPU.")
    native_bf16 = torch.cuda.is_bf16_supported(including_emulation=False)
    precision = config.precision
    if precision == "auto":
        precision = "bfloat16" if native_bf16 else "float16"
    if precision == "bfloat16" and not native_bf16:
        raise ValueError("This GPU does not natively support bfloat16; explicitly select float16 or auto.")
    dtype = getattr(torch, precision)
    revision = resolve_revision(config.model_id, config.model_revision)
    hf_config = AutoConfig.from_pretrained(config.model_id, revision=revision, trust_remote_code=False)
    if hf_config.model_type != "qwen3_5" or "Qwen3_5ForConditionalGeneration" not in hf_config.architectures:
        raise ValueError("Checkpoint architecture differs from the verified Qwen3.5 multimodal interface.")
    set_reproducibility(config.seed, config.deterministic_algorithms)
    processor = AutoProcessor.from_pretrained(config.model_id, revision=revision, trust_remote_code=False)
    if not isinstance(processor.chat_template, str) or "enable_thinking" not in processor.chat_template:
        raise ValueError("The pinned processor lacks the verified thinking-aware chat template.")
    software_message = [{"role": "user", "content": [{"type": "text", "text": "Software setup check."}]}]
    templates = [
        processor.apply_chat_template(
            software_message, tokenize=False, add_generation_prompt=True, enable_thinking=state
        )
        for state in (False, True)
    ]
    if templates[0] == templates[1]:
        raise ValueError("Thinking switch did not change the rendered template; loading stopped.")
    quantization = None
    if config.quantization == "nf4":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["lm_head", "model.visual"],
        )
    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            config.model_id,
            revision=revision,
            config=hf_config,
            dtype=dtype,
            device_map={"": 0},
            attn_implementation="sdpa",
            trust_remote_code=False,
            use_safetensors=True,
            quantization_config=quantization,
            use_kernels=False,
        ).eval()
    except torch.OutOfMemoryError as exc:
        gc.collect()
        torch.cuda.empty_cache()
        raise InferenceOOM(oom_instruction()) from exc
    except (ImportError, ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"Loading {config.model_id} at {revision} with {config.quantization} failed. "
            "No fallback was attempted. Check installed versions, GPU memory and BitsAndBytes "
            "compatibility; deliberately change quantization only in a new run. "
            f"Original error: {exc}"
        ) from exc
    modules_4bit = [
        name for name, module in model.named_modules() if module.__class__.__name__ == "Linear4bit"
    ]
    if config.quantization == "nf4" and (not getattr(model, "is_loaded_in_4bit", False) or not modules_4bit):
        raise ValueError("NF4 requested but no actual 4-bit layers were found; loading stopped.")
    eos = model.generation_config.eos_token_id
    if eos is None:
        raise ValueError("The pinned model has no end-of-sequence token configuration.")
    pad = processor.tokenizer.pad_token_id
    if pad is None:
        pad = eos[0] if isinstance(eos, list) else eos
    generation = GenerationConfig(
        max_new_tokens=config.max_new_tokens,
        do_sample=config.do_sample,
        num_beams=1,
        num_return_sequences=1,
        repetition_penalty=config.repetition_penalty,
        temperature=config.temperature if config.do_sample else 1.0,
        top_p=config.top_p if config.do_sample else 1.0,
        top_k=config.top_k if config.do_sample else 0,
        bos_token_id=model.generation_config.bos_token_id,
        eos_token_id=eos,
        pad_token_id=pad,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=False,
    )
    metadata = {
        "model_id": config.model_id,
        "resolved_revision": revision,
        "model_class": type(model).__name__,
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(processor.tokenizer).__name__,
        "compute_precision": precision,
        "attention_implementation": "sdpa",
        "device_map": {"": "cuda:0"},
        "hub_kernels_enabled": False,
        "quantization_config": quantization.to_dict() if quantization else None,
        "quantized_module_names": modules_4bit,
        "chat_template_sha256": sha256_bytes(processor.chat_template.encode("utf-8")),
        "effective_generation_config": generation.to_dict(),
        "decode_settings": {"skip_special_tokens": False, "clean_up_tokenization_spaces": False},
        "determinism": {
            "strict_algorithms": config.deterministic_algorithms,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "tf32": False,
        },
        "environment": environment_report(),
    }
    return QwenRunner(model, processor, config, metadata, generation)
