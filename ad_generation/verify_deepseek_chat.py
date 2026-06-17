"""Manual verification script for comparing local_fake vs deepseek_chat.

Run from project root:
python ad_generation/verify_deepseek_chat.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from v5_dynamic_creative import v5_llm_dynamic_creative
from v5_runtime import build_v5_runtime_context, warm_retrieval_cache


DEFAULT_QUERIES = [
    "运动手环",
    "宿舍吹风机",
    "通勤双肩包",
    "敏感肌面霜",
    "手机壳",
]


def _project_paths() -> Dict[str, Path]:
    project_root = CURRENT_DIR.parent
    return {
        "project_root": project_root,
        "corpus_path": project_root / "items_lite" / "train.jsonl",
        "rank_path": project_root / "rank_lite" / "train.jsonl",
        "users_path": project_root / "users_lite" / "train.jsonl",
        "pairs_path": CURRENT_DIR / "data" / "train_pairs.jsonl",
        "local_config_path": CURRENT_DIR / "llm_config.mock.json",
        "deepseek_config_path": CURRENT_DIR / "llm_config.deepseek_chat.json",
        "jsonl_output_path": project_root / "outputs" / "deepseek_verification.jsonl",
        "markdown_output_path": project_root / "outputs" / "deepseek_verification.md",
    }


def _run_pipeline(
    query: str,
    provider: str,
    runtime_context: object,
    paths: Dict[str, Path],
) -> Dict[str, object]:
    config_path = paths["deepseek_config_path"] if provider == "deepseek_chat" else paths["local_config_path"]
    return v5_llm_dynamic_creative(
        query=query,
        corpus_path=paths["corpus_path"],
        rank_path=paths["rank_path"],
        users_path=paths["users_path"],
        pairs_path=paths["pairs_path"],
        llm_config_path=config_path,
        top_k=3,
        candidate_count=5,
        requested_tone="creative",
        runtime_context=runtime_context,
        provider_override=provider,
    )


def _extract_final_copy(result: Dict[str, object]) -> str:
    return str(result.get("final_ad_copy", {}).get("final_ad_copy") or "")


def _extract_candidates(result: Dict[str, object], limit: int = 5) -> List[str]:
    ranked_candidates = list(result.get("copy_ranker", {}).get("ranked_candidates", []))
    parsed_candidates = list(result.get("llm_output_parser", {}).get("parsed_candidates", []))
    raw_candidates = ranked_candidates or parsed_candidates

    outputs: List[str] = []
    seen = set()
    for item in raw_candidates:
        text = str(item.get("text") or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        outputs.append(text)
        if len(outputs) >= max(1, limit):
            break
    return outputs


def _extract_errors(result: Dict[str, object]) -> List[str]:
    provider_bundle = dict(result.get("llm_provider") or {})
    api_results = list(provider_bundle.get("api_results") or [])

    errors: List[str] = []
    for item in api_results:
        error = str(item.get("error") or "").strip()
        if error:
            errors.append(error)

    provider_status = str(provider_bundle.get("status") or "")
    if provider_status == "skipped_no_api_key":
        return []
    if not errors and provider_status not in {"", "provider_ok", "local_fake_json_ok"}:
        errors.append(f"llm_status={provider_status}")
    return errors


def _summarize_local_result(query: str, result: Dict[str, object]) -> Dict[str, object]:
    return {
        "query": query,
        "status": "ok",
        "requested_provider": "local_fake",
        "provider": str(result.get("provider") or ""),
        "model": str(result.get("model") or ""),
        "llm_status": str(result.get("llm_status") or ""),
        "final_ad_copy": _extract_final_copy(result),
        "candidates": _extract_candidates(result),
        "errors": [],
        "fallback_from": "",
    }


def _build_skipped_deepseek_summary(query: str, local_summary: Dict[str, object]) -> Dict[str, object]:
    return {
        "query": query,
        "status": "skipped_no_api_key",
        "requested_provider": "deepseek_chat",
        "provider": "deepseek_chat",
        "model": os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        "llm_status": "skipped_no_api_key",
        "final_ad_copy": "",
        "candidates": [],
        "errors": [],
        "fallback_from": "",
        "display_final_ad_copy": "",
        "display_candidates": [],
        "actual_final_ad_copy": "",
        "actual_candidates": [],
        "local_fake_fallback_preview": {
            "final_ad_copy": str(local_summary.get("final_ad_copy") or ""),
            "candidates": list(local_summary.get("candidates") or []),
        },
    }


def _summarize_deepseek_result(
    query: str,
    result: Dict[str, object],
    local_summary: Dict[str, object],
) -> Dict[str, object]:
    provider_bundle = dict(result.get("llm_provider") or {})
    llm_status = str(provider_bundle.get("status") or result.get("llm_status") or "")
    actual_final_copy = _extract_final_copy(result)
    actual_candidates = _extract_candidates(result)
    errors = _extract_errors(result)

    api_ok = llm_status == "provider_ok" and not errors
    status = "ok" if api_ok else "error"
    display_final_copy = actual_final_copy if api_ok else str(local_summary.get("final_ad_copy") or "")
    display_candidates = actual_candidates if api_ok else list(local_summary.get("candidates") or [])

    return {
        "query": query,
        "status": status,
        "requested_provider": "deepseek_chat",
        "provider": str(result.get("provider") or ""),
        "model": str(result.get("model") or ""),
        "llm_status": llm_status,
        "final_ad_copy": display_final_copy,
        "candidates": display_candidates,
        "errors": errors,
        "fallback_from": "" if api_ok else "local_fake",
        "display_final_ad_copy": display_final_copy,
        "display_candidates": display_candidates,
        "actual_final_ad_copy": actual_final_copy,
        "actual_candidates": actual_candidates,
    }


def _risk_observation(text: str) -> str:
    risk_terms = [
        "官方",
        "唯一",
        "100%",
        "顶级",
        "根治",
        "永久",
        "包治",
        "无副作用",
        "药效",
        "治疗",
    ]
    hits = [term for term in risk_terms if term in text]
    if hits:
        return "有潜在风险，命中词：" + "、".join(hits[:4])
    return "未见明显不合规或幻觉风险"


def _naturalness_observation(local_summary: Dict[str, object], deepseek_summary: Dict[str, object]) -> str:
    deepseek_status = str(deepseek_summary.get("status") or "")
    if deepseek_status == "skipped_no_api_key":
        return "未观察，未设置 DEEPSEEK_API_KEY"
    if deepseek_status == "error":
        return "未稳定观察，deepseek_chat 调用失败后展示为 local_fake 回退结果"

    local_text = str(local_summary.get("final_ad_copy") or "")
    deepseek_text = str(deepseek_summary.get("actual_final_ad_copy") or "")
    if not deepseek_text:
        return "未观察到有效输出"
    if deepseek_text == local_text:
        return "和 local_fake 基本一致，没有明显自然度优势"
    if len(deepseek_text) >= len(local_text):
        return "看起来更像自然推荐语气，但仍需人工主观判断"
    return "有差异，但是否更自然仍需人工主观判断"


def _richness_observation(local_summary: Dict[str, object], deepseek_summary: Dict[str, object]) -> str:
    deepseek_status = str(deepseek_summary.get("status") or "")
    if deepseek_status == "skipped_no_api_key":
        return "未观察，deepseek_chat 未执行"
    if deepseek_status == "error":
        return "未观察到稳定输出，报告展示的是 local_fake 回退结果"

    local_candidates = list(local_summary.get("candidates") or [])
    deepseek_candidates = list(deepseek_summary.get("actual_candidates") or [])
    local_len = len(str(local_summary.get("final_ad_copy") or ""))
    deepseek_len = len(str(deepseek_summary.get("actual_final_ad_copy") or ""))

    if len(deepseek_candidates) > len(local_candidates) or deepseek_len > local_len:
        return (
            f"相对更丰富，deepseek 候选 {len(deepseek_candidates)} 条，"
            f"local_fake 候选 {len(local_candidates)} 条"
        )
    if len(deepseek_candidates) == len(local_candidates) and deepseek_len == local_len:
        return "丰富度接近，更像是表达风格差异"
    return (
        f"没有明显更丰富，deepseek 候选 {len(deepseek_candidates)} 条，"
        f"local_fake 候选 {len(local_candidates)} 条"
    )


def _build_observations(local_summary: Dict[str, object], deepseek_summary: Dict[str, object]) -> List[str]:
    observation_text = str(
        deepseek_summary.get("actual_final_ad_copy")
        or deepseek_summary.get("final_ad_copy")
        or ""
    )
    return [
        "deepseek 是否更自然：" + _naturalness_observation(local_summary, deepseek_summary),
        "是否更丰富：" + _richness_observation(local_summary, deepseek_summary),
        "是否有不合规或幻觉风险：" + _risk_observation(observation_text),
    ]


def _render_candidates(candidates: List[str]) -> str:
    return " | ".join(candidates) if candidates else "(empty)"


def _append_markdown_block(
    lines: List[str],
    title: str,
    summary: Dict[str, object],
    include_status: bool,
) -> None:
    lines.append(title)
    if include_status:
        lines.append(f"- status: {summary.get('status')}")
    lines.append(f"- final_ad_copy: {summary.get('final_ad_copy')}")
    lines.append(f"- candidates: {_render_candidates(list(summary.get('candidates') or []))}")
    errors = list(summary.get("errors") or [])
    if errors:
        lines.append(f"- error: {errors[0]}")
    if summary.get("fallback_from"):
        lines.append(f"- fallback_from: {summary.get('fallback_from')}")


def main() -> None:
    paths = _project_paths()
    paths["jsonl_output_path"].parent.mkdir(parents=True, exist_ok=True)

    runtime_context = build_v5_runtime_context(
        corpus_path=paths["corpus_path"],
        rank_path=paths["rank_path"],
        users_path=paths["users_path"],
        pairs_path=paths["pairs_path"],
    )
    warm_retrieval_cache(
        DEFAULT_QUERIES,
        runtime_context=runtime_context,
        top_k=3,
        allow_expensive_fallback=False,
    )

    has_deepseek_key = bool(str(os.environ.get("DEEPSEEK_API_KEY") or "").strip())
    jsonl_records: List[Dict[str, object]] = []
    markdown_lines: List[str] = []

    for query in DEFAULT_QUERIES:
        local_result = _run_pipeline(
            query=query,
            provider="local_fake",
            runtime_context=runtime_context,
            paths=paths,
        )
        local_summary = _summarize_local_result(query=query, result=local_result)

        deepseek_result: Optional[Dict[str, object]] = None
        if has_deepseek_key:
            deepseek_result = _run_pipeline(
                query=query,
                provider="deepseek_chat",
                runtime_context=runtime_context,
                paths=paths,
            )
            deepseek_summary = _summarize_deepseek_result(
                query=query,
                result=deepseek_result,
                local_summary=local_summary,
            )
        else:
            deepseek_summary = _build_skipped_deepseek_summary(
                query=query,
                local_summary=local_summary,
            )

        observations = _build_observations(local_summary, deepseek_summary)
        jsonl_records.append(
            {
                "query": query,
                "local_fake": local_summary,
                "deepseek_chat": deepseek_summary,
                "local_fake_result": local_result,
                "deepseek_chat_result": deepseek_result,
                "observations": observations,
            }
        )

        markdown_lines.append(f"## Query: {query}")
        markdown_lines.append("")
        _append_markdown_block(markdown_lines, "### Local Fake", local_summary, include_status=False)
        markdown_lines.append("")
        _append_markdown_block(markdown_lines, "### DeepSeek Chat", deepseek_summary, include_status=True)
        markdown_lines.append("")
        markdown_lines.append("### 简要观察")
        for note in observations:
            markdown_lines.append(f"- {note}")
        markdown_lines.append("")

    with paths["jsonl_output_path"].open("w", encoding="utf-8") as handle:
        for record in jsonl_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    with paths["markdown_output_path"].open("w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown_lines).rstrip() + "\n")

    print(paths["jsonl_output_path"])
    print(paths["markdown_output_path"])


if __name__ == "__main__":
    main()
