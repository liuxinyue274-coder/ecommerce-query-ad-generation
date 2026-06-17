"""Lightweight LLM generation adapter for ad copy inference.

This module intentionally keeps the interface provider-agnostic.
Current providers:
- mock: returns a deterministic pseudo-LLM copy for pipeline testing
- local: reserved for local model integration
- api_openai_compatible: minimal OpenAI-compatible chat-completions adapter
- sft_local: reserved local SFT / LoRA inference entry
"""

from __future__ import annotations

import json
import importlib
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional


_SFT_LOCAL_CACHE: Dict[str, object] = {
    "model_path": None,
    "adapter_path": None,
    "tokenizer": None,
    "model": None,
}


def load_llm_config(config_path: Path) -> Dict[str, object]:
    """Load an LLM config JSON file with Windows-friendly UTF-8 handling."""

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"LLM config file not found: {path}")

    with path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError("LLM config must be a JSON object.")
    return config


def _clean_text(value: object) -> str:
    """Convert values to stripped text."""

    if value is None:
        return ""
    return str(value).strip()


def _first_non_empty(data: Optional[Dict[str, object]], keys: Iterable[str]) -> str:
    """Read the first non-empty field from an item dict."""

    if not data:
        return ""
    for key in keys:
        value = _clean_text(data.get(key))
        if value:
            return value
    return ""


def _truncate_copy(text: str, max_chars: int = 50) -> str:
    """Keep copy short enough for the first LLM prototype."""

    clean = re.sub(r"\s+", " ", _clean_text(text))
    if len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 1].rstrip("，。；、 ") + "。"


def _shorten_title(title: str, max_chars: int = 14) -> str:
    """Trim long titles before composing mock copy."""

    clean = _clean_text(title)
    if len(clean) <= max_chars:
        return clean
    return clean[:max_chars].rstrip("，。；、 ") + "等商品"


def _extract_query_from_prompt(prompt: str) -> str:
    """Extract query text from the LLM prompt when using mock mode."""

    match = re.search(r"Query:\s*\n(.+?)(?:\n\n|\n用户画像：|\n最近行为：|\nEvidence Items:|\n商品证据：|$)", prompt, re.S)
    if match:
        return _clean_text(match.group(1))
    return ""


def _extract_first_evidence_title(prompt: str) -> str:
    """Extract the first evidence title from the prompt for mock generation."""

    block_match = re.search(
        r"(?:商品证据|Evidence Items):\s*\n(.+?)(?:\n\n输出格式|\n\nOutput:|$)",
        prompt,
        re.S,
    )
    if block_match:
        block = block_match.group(1)
        line_match = re.search(r"^\s*1\.\s*([^\n|]+)", block, re.M)
        if line_match:
            return _clean_text(line_match.group(1))
    return ""


def _mock_generate(prompt: str, config: Dict[str, object]) -> str:
    """Return a deterministic pseudo-LLM copy for offline pipeline checks."""

    del config
    query = _extract_query_from_prompt(prompt)
    title = _extract_first_evidence_title(prompt)

    if query and title:
        short_title = _shorten_title(title)
        return _truncate_copy(f"搜“{query}”可先关注{short_title}，更贴近当前需求。")
    if query:
        return _truncate_copy(f"搜“{query}”可先关注当前更匹配的商品方向，便于继续筛选。")
    return "推荐先关注当前更匹配的商品方向，便于继续筛选。"


def _extract_text_from_openai_compatible_payload(payload: Dict[str, object]) -> str:
    """Extract assistant text from an OpenAI-compatible chat response."""

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""

    message = first_choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return _clean_text(content)
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return _clean_text("".join(parts))

    text = first_choice.get("text")
    if isinstance(text, str):
        return _clean_text(text)
    return ""


def _call_openai_compatible(prompt: str, config: Dict[str, object], provider_name: str) -> Dict[str, object]:
    """Call an OpenAI-compatible chat-completions endpoint using stdlib only."""

    started_at = time.perf_counter()
    base_url = _clean_text(config.get("base_url"))
    model_name = _clean_text(config.get("model_name"))
    api_key_env = _clean_text(config.get("api_key_env")) or "LLM_API_KEY"
    api_key = _clean_text(os.environ.get(api_key_env))

    if not base_url:
        raise RuntimeError(f"{provider_name} requires `base_url` in config.")
    if not model_name:
        raise RuntimeError(f"{provider_name} requires `model_name` in config.")
    if not api_key:
        raise RuntimeError(f"{provider_name} requires environment variable `{api_key_env}`.")

    endpoint = base_url.rstrip("/") + "/chat/completions"
    request_body = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": config.get("temperature", 0.7),
        "max_tokens": int(config.get("max_new_tokens", 128) or 128),
    }
    timeout = int(config.get("timeout", 60) or 60)
    request_bytes = json.dumps(request_body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=request_bytes,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"{provider_name} http_error={exc.code} detail={detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{provider_name} network_error={exc}") from exc

    payload = json.loads(response_text)
    generated = _extract_text_from_openai_compatible_payload(payload)
    if not generated:
        raise RuntimeError(f"{provider_name} returned an empty completion.")
    return {
        "ok": True,
        "text": generated,
        "provider": provider_name,
        "model": model_name,
        "error": None,
        "latency_ms": int((time.perf_counter() - started_at) * 1000),
    }


def _generate_with_openai_compatible(prompt: str, config: Dict[str, object]) -> str:
    """Call an OpenAI-compatible chat-completions endpoint using stdlib only."""

    result = _call_openai_compatible(prompt, config, provider_name="api_openai_compatible")
    return _clean_text(result.get("text"))


def _build_deepseek_config(config: Dict[str, object]) -> Dict[str, object]:
    """Normalize DeepSeek config with environment-variable defaults."""

    normalized = dict(config)
    normalized["base_url"] = _clean_text(os.environ.get("DEEPSEEK_BASE_URL")) or _clean_text(normalized.get("base_url")) or "https://api.deepseek.com"
    normalized["model_name"] = _clean_text(os.environ.get("DEEPSEEK_MODEL")) or _clean_text(normalized.get("model_name")) or "deepseek-chat"
    normalized["api_key_env"] = "DEEPSEEK_API_KEY"
    normalized["temperature"] = float(_clean_text(os.environ.get("DEEPSEEK_TEMPERATURE")) or normalized.get("temperature") or 0.7)
    normalized["max_new_tokens"] = int(_clean_text(os.environ.get("DEEPSEEK_MAX_TOKENS")) or normalized.get("max_new_tokens") or 512)
    return normalized


def _build_sft_local_config(config: Dict[str, object]) -> Dict[str, object]:
    """Normalize local SFT config with environment-variable defaults."""

    normalized = dict(config)
    normalized["model_name"] = _clean_text(os.environ.get("SFT_MODEL_PATH")) or _clean_text(normalized.get("model_name"))
    normalized["adapter_path"] = _clean_text(os.environ.get("SFT_ADAPTER_PATH")) or _clean_text(normalized.get("adapter_path"))
    normalized["temperature"] = float(_clean_text(os.environ.get("SFT_TEMPERATURE")) or normalized.get("temperature") or 0.7)
    normalized["max_new_tokens"] = int(_clean_text(os.environ.get("SFT_MAX_TOKENS")) or normalized.get("max_new_tokens") or 512)
    normalized["timeout"] = int(normalized.get("timeout", 60) or 60)
    return normalized


def _check_import_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def _load_sft_local_artifacts(model_path: str, adapter_path: str) -> tuple[object, object]:
    """Lazily load tokenizer/model and optional LoRA adapter for local SFT inference."""

    if (
        _SFT_LOCAL_CACHE.get("tokenizer") is not None
        and _SFT_LOCAL_CACHE.get("model") is not None
        and _SFT_LOCAL_CACHE.get("model_path") == model_path
        and _SFT_LOCAL_CACHE.get("adapter_path") == adapter_path
    ):
        return _SFT_LOCAL_CACHE["tokenizer"], _SFT_LOCAL_CACHE["model"]

    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    AutoTokenizer = getattr(transformers, "AutoTokenizer")
    AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    if adapter_path:
        if _check_import_available("peft"):
            peft = importlib.import_module("peft")
            PeftModel = getattr(peft, "PeftModel")
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            raise RuntimeError("peft not available for loading SFT_ADAPTER_PATH")

    model.eval()
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        model = model.to("cuda")

    _SFT_LOCAL_CACHE["model_path"] = model_path
    _SFT_LOCAL_CACHE["adapter_path"] = adapter_path
    _SFT_LOCAL_CACHE["tokenizer"] = tokenizer
    _SFT_LOCAL_CACHE["model"] = model
    return tokenizer, model


def _call_sft_local(prompt: str, config: Dict[str, object]) -> Dict[str, object]:
    """Run local SFT / LoRA inference if the runtime is configured and available."""

    started_at = time.perf_counter()
    normalized = _build_sft_local_config(config)
    model_path = _clean_text(normalized.get("model_name"))
    adapter_path = _clean_text(normalized.get("adapter_path"))

    if not model_path:
        return {
            "ok": False,
            "text": "",
            "provider": "sft_local",
            "model": "",
            "error": "SFT_MODEL_PATH not configured",
            "latency_ms": 0,
            "status": "skipped_no_sft_model",
        }

    if not (_check_import_available("transformers") and _check_import_available("torch")):
        return {
            "ok": False,
            "text": "",
            "provider": "sft_local",
            "model": model_path,
            "error": "transformers/torch not available",
            "latency_ms": 0,
            "status": "skipped_missing_dependencies",
        }

    try:
        torch = importlib.import_module("torch")
        tokenizer, model = _load_sft_local_artifacts(model_path=model_path, adapter_path=adapter_path)
        encoded = tokenizer(prompt, return_tensors="pt")
        model_device = getattr(model, "device", None)
        if model_device is not None:
            encoded = {key: value.to(model_device) for key, value in encoded.items()}

        generation_kwargs = {
            "max_new_tokens": int(normalized.get("max_new_tokens") or 512),
            "do_sample": True,
            "temperature": float(normalized.get("temperature") or 0.7),
        }

        with torch.no_grad():
            output_ids = model.generate(**encoded, **generation_kwargs)

        prompt_token_count = int(encoded["input_ids"].shape[-1])
        generated_ids = output_ids[0][prompt_token_count:]
        generated_text = _clean_text(tokenizer.decode(generated_ids, skip_special_tokens=True))
        if not generated_text:
            return {
                "ok": False,
                "text": "",
                "provider": "sft_local",
                "model": model_path,
                "error": "sft_local returned an empty completion.",
                "latency_ms": int((time.perf_counter() - started_at) * 1000),
                "status": "runtime_error",
            }
        return {
            "ok": True,
            "text": generated_text,
            "provider": "sft_local",
            "model": model_path,
            "error": None,
            "latency_ms": int((time.perf_counter() - started_at) * 1000),
            "status": "ok",
        }
    except Exception as exc:
        return {
            "ok": False,
            "text": "",
            "provider": "sft_local",
            "model": model_path,
            "error": str(exc),
            "latency_ms": int((time.perf_counter() - started_at) * 1000),
            "status": "runtime_error",
        }


def _generate_with_deepseek_chat(prompt: str, config: Dict[str, object]) -> str:
    """Call DeepSeek chat through its OpenAI-compatible endpoint."""

    result = _call_openai_compatible(prompt, _build_deepseek_config(config), provider_name="deepseek_chat")
    return _clean_text(result.get("text"))


def generate_with_llm_result(prompt: str, config: Dict[str, object]) -> Dict[str, object]:
    """Generate ad copy with response metadata instead of raising provider details outward."""

    provider = _clean_text(config.get("provider")).lower() or "mock"
    started_at = time.perf_counter()
    if provider == "mock":
        return {
            "ok": True,
            "text": _mock_generate(prompt, config),
            "provider": "mock",
            "model": _clean_text(config.get("model_name")) or "mock",
            "error": None,
            "latency_ms": int((time.perf_counter() - started_at) * 1000),
            "status": "ok",
        }
    if provider == "deepseek_chat":
        try:
            result = _call_openai_compatible(prompt, _build_deepseek_config(config), provider_name="deepseek_chat")
            result["status"] = "ok"
            return result
        except Exception as exc:
            return {
                "ok": False,
                "text": "",
                "provider": "deepseek_chat",
                "model": _clean_text(os.environ.get("DEEPSEEK_MODEL")) or _clean_text(config.get("model_name")) or "deepseek-chat",
                "error": str(exc),
                "latency_ms": int((time.perf_counter() - started_at) * 1000),
                "status": "runtime_error",
            }
    if provider == "sft_local":
        return _call_sft_local(prompt, config)
    if provider == "api_openai_compatible":
        try:
            result = _call_openai_compatible(prompt, config, provider_name="api_openai_compatible")
            result["status"] = "ok"
            return result
        except Exception as exc:
            return {
                "ok": False,
                "text": "",
                "provider": "api_openai_compatible",
                "model": _clean_text(config.get("model_name")) or "unknown",
                "error": str(exc),
                "latency_ms": int((time.perf_counter() - started_at) * 1000),
                "status": "runtime_error",
            }
    if provider == "local":
        return {
            "ok": False,
            "text": "",
            "provider": "local",
            "model": _clean_text(config.get("model_name")) or "local",
            "error": "Local LLM provider is reserved but not wired in this prototype.",
            "latency_ms": int((time.perf_counter() - started_at) * 1000),
            "status": "not_implemented",
        }
    return {
        "ok": False,
        "text": "",
        "provider": provider,
        "model": _clean_text(config.get("model_name")) or "unknown",
        "error": f"Unsupported LLM provider: {provider}",
        "latency_ms": int((time.perf_counter() - started_at) * 1000),
        "status": "unsupported_provider",
    }


def generate_with_llm(prompt: str, config: Dict[str, object]) -> str:
    """Generate ad copy using the configured provider."""

    provider = _clean_text(config.get("provider")).lower() or "mock"
    if provider == "mock":
        return _mock_generate(prompt, config)
    if provider == "local":
        raise NotImplementedError("Local LLM provider is reserved but not wired in this prototype.")
    if provider == "deepseek_chat":
        return _generate_with_deepseek_chat(prompt, config)
    if provider == "sft_local":
        result = _call_sft_local(prompt, config)
        if not result.get("ok"):
            raise RuntimeError(str(result.get("error") or "sft_local failed"))
        return _clean_text(result.get("text"))
    if provider == "api_openai_compatible":
        return _generate_with_openai_compatible(prompt, config)
    raise ValueError(f"Unsupported LLM provider: {provider}")


def fallback_generate(prompt: str, evidence_items: Optional[List[Dict[str, object]]], query: str) -> str:
    """Generate a conservative fallback copy when LLM generation fails."""

    del prompt

    items = list(evidence_items or [])
    first_item = items[0] if items else {}
    title = _first_non_empty(first_item, ["item_title", "title"])
    seller = _first_non_empty(first_item, ["seller_name", "seller"])
    category_path = _first_non_empty(first_item, ["category_path"])
    category_short = category_path.split(">")[-1].strip() if category_path else ""
    clean_query = _clean_text(query)

    if title and seller:
        return _truncate_copy(f"围绕“{clean_query}”，可先看看这款{title}，由{seller}在售，信息更直观。")
    if title:
        return _truncate_copy(f"围绕“{clean_query}”，可先看看这款{title}，更贴近当前搜索需求。")
    if category_short:
        return _truncate_copy(f"围绕“{clean_query}”，建议先关注{category_short}方向的候选商品。")
    return _truncate_copy(f"围绕“{clean_query}”，建议先关注当前更匹配的候选商品方向。")
