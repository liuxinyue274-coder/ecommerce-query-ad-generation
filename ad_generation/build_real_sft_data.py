"""Build real SFT ad-copy data from the V5 generation pipeline.

Usage examples:

python ad_generation/build_real_sft_data.py --target_count 20 --max_source 80 --provider deepseek_chat --timeout 30 --max_retries 1
python ad_generation/build_real_sft_data.py --limit 20 --provider local_fake
python ad_generation/build_real_sft_data.py --dry_run_source --target_count 5 --max_source 20
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from build_train_pairs import clean_text, iter_jsonl
from prompt_builder import build_evidence_block, build_user_context
from v5_dynamic_creative import (
    evidence_selector_step,
    intent_enricher_step,
    prompt_builder_step,
    query_normalization_step,
    style_retriever_step,
    user_profile_builder_step,
    v5_llm_dynamic_creative,
)
from v5_runtime import build_v5_runtime_context, retrieve_evidence_bundle_cached


SYSTEM_PROMPT = (
    "你是电商搜索广告创意生成助手。请基于用户搜索意图、用户画像和商品证据，"
    "生成自然、短促、合规的广告文案。不得编造证据中没有的信息。"
)

TEMPLATE_PATTERNS = [
    "可以先看这款",
    "可先关注",
    "信息清晰",
    "值得优先关注",
    "适合先了解",
    "推荐先看",
    "当前候选主要集中在",
    "建议先从这类商品里继续筛选",
    "更贴近当前需求",
    "等商品",
]

SUSPICIOUS_TOKENS = [
    "官方",
    "正品",
    "唯一",
    "优惠",
    "立减",
    "限时",
    "抢购",
    "开抢",
    "销量",
    "热销",
    "爆款",
    "改善",
    "修复",
    "药效",
    "功效",
    "认证",
    "100%",
]

STOPWORDS = {
    "可以",
    "这款",
    "当前",
    "相关",
    "商品",
    "好用",
    "自然",
    "适合",
    "优选",
    "推荐",
    "一下",
    "一个",
    "日常",
    "通勤",
    "场景",
    "方向",
    "选择",
    "用户",
    "人群",
}

SUPPORTED_PROVIDERS = {"deepseek_chat", "local_fake"}
HIGH_CONFIDENCE_SOURCES = {
    "train_pairs_user_exact",
    "train_pairs_session_exact",
    "train_pairs_query_exact",
}
MEDIUM_CONFIDENCE_SOURCES = {
    "train_pairs_query_fuzzy",
    "rank_query_exact",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real SFT ad-copy data from the V5 pipeline.")
    parser.add_argument(
        "--target_count",
        type=int,
        default=100,
        help="Target number of accepted SFT samples to generate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Compatibility alias for --target_count.",
    )
    parser.add_argument(
        "--max_source",
        type=int,
        default=300,
        help="Maximum number of raw source rows to scan/attempt before stopping.",
    )
    parser.add_argument(
        "--provider",
        choices=sorted(SUPPORTED_PROVIDERS),
        default="deepseek_chat",
        help="Generation provider to request from the V5 pipeline.",
    )
    parser.add_argument("--min_score", type=float, default=80.0, help="Minimum rank score for keeping a sample.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/real_sft_ad_copy.jsonl"),
        help="Output JSONL path for accepted SFT samples.",
    )
    parser.add_argument(
        "--pairs_path",
        type=Path,
        default=Path("ad_generation/data/train_pairs.jsonl"),
        help="Source train_pairs JSONL path.",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Evidence item count passed into the V5 pipeline.")
    parser.add_argument(
        "--candidate_count",
        type=int,
        default=5,
        choices=[3, 4, 5],
        help="Number of candidate ad copies requested from the provider.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Per-request provider timeout in seconds.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum retry count after the first failed sample generation attempt.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="Sleep seconds between retries.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output JSONL and skip duplicate queries already written.",
    )
    parser.add_argument(
        "--dry_run_source",
        action="store_true",
        help="Only scan source rows and build V5 context up to prompt construction, without calling the provider.",
    )
    return parser.parse_args()


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
    }


def _effective_target_count(args: argparse.Namespace) -> int:
    return max(1, int(args.limit if args.limit is not None else args.target_count))


def _normalize_compare_text(text: object) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", clean_text(text)).lower()


def _tokenize(text: object) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,4}", clean_text(text))
    return [token.lower() for token in tokens if len(token) >= 2]


def _detect_suspicious_claims(copy_text: str) -> List[str]:
    text = clean_text(copy_text)
    findings = [token for token in SUSPICIOUS_TOKENS if token in text]
    if re.search(r"\d+(?:\.\d+)?元", text):
        findings.append("price_claim")
    return findings


def _is_template_like(copy_text: str) -> bool:
    text = clean_text(copy_text)
    if any(pattern in text for pattern in TEMPLATE_PATTERNS):
        return True
    if "搜“" in text and "推荐看看这款" in text:
        return True
    return False


def _format_intent(intent_bundle: Dict[str, object]) -> str:
    parts: List[str] = []
    intent_summary = clean_text(intent_bundle.get("intent_summary"))
    scenario = clean_text(intent_bundle.get("scenario"))
    audience = clean_text(intent_bundle.get("audience"))
    purchase_focus = clean_text(intent_bundle.get("purchase_focus"))
    attribute_hints = [clean_text(item) for item in (intent_bundle.get("attribute_hints") or []) if clean_text(item)]

    if intent_summary:
        parts.append(f"意图总结：{intent_summary}")
    if scenario:
        parts.append(f"场景：{scenario}")
    if audience:
        parts.append(f"人群：{audience}")
    if purchase_focus:
        parts.append(f"关注点：{purchase_focus}")
    if attribute_hints:
        parts.append("属性提示：" + "、".join(attribute_hints[:5]))
    return "\n".join(parts)


def _format_user_profile(user_profile: Dict[str, object], recent_behavior_titles: Sequence[object]) -> str:
    text = build_user_context(user_profile=user_profile, recent_behavior_titles=recent_behavior_titles)
    return clean_text(text) or "(empty)"


def _format_evidence(evidence_items: Sequence[Dict[str, object]]) -> str:
    text = build_evidence_block(evidence_items)
    return clean_text(text) or "(empty)"


def _format_style_examples(style_examples: Sequence[object]) -> str:
    cleaned = [clean_text(item) for item in style_examples if clean_text(item)]
    if not cleaned:
        return "(empty)"
    return "\n".join(f"{index}. {item}" for index, item in enumerate(cleaned[:5], start=1))


def _build_user_message(
    query: str,
    intent_bundle: Dict[str, object],
    user_profile_bundle: Dict[str, object],
    evidence_bundle: Dict[str, object],
    style_bundle: Dict[str, object],
) -> str:
    user_profile_raw = dict(user_profile_bundle.get("user_profile_raw") or {})
    recent_behavior_titles = list(user_profile_bundle.get("recent_behavior_titles") or [])
    selected_evidence = list(evidence_bundle.get("selected_evidence_items") or [])
    style_examples = list(style_bundle.get("style_examples") or [])
    if not style_examples and style_bundle.get("example_pattern"):
        style_examples = [style_bundle.get("example_pattern")]

    return "\n".join(
        [
            f"Query: {clean_text(query)}",
            f"Intent: {_format_intent(intent_bundle)}",
            f"User Profile: {_format_user_profile(user_profile_raw, recent_behavior_titles)}",
            f"Evidence: {_format_evidence(selected_evidence)}",
            f"Style Examples: {_format_style_examples(style_examples)}",
            "请生成1条自然广告文案。",
        ]
    )


def _evidence_keyword_set(query: str, evidence_items: Sequence[Dict[str, object]]) -> Set[str]:
    keywords: Set[str] = set()
    for token in _tokenize(query):
        if token not in STOPWORDS:
            keywords.add(token)

    for item in evidence_items:
        for field in [item.get("item_title"), item.get("brand_name"), item.get("seller_name"), item.get("category_path")]:
            for token in _tokenize(field):
                if token not in STOPWORDS:
                    keywords.add(token)
    return keywords


def _is_related_to_evidence(copy_text: str, query: str, evidence_items: Sequence[Dict[str, object]]) -> bool:
    copy_tokens = {token for token in _tokenize(copy_text) if token not in STOPWORDS}
    if not copy_tokens:
        return False
    return bool(copy_tokens & _evidence_keyword_set(query, evidence_items))


def _estimate_evidence_confidence(retrieval_source: str, selected_evidence: Sequence[Dict[str, object]]) -> str:
    if retrieval_source in HIGH_CONFIDENCE_SOURCES and len(selected_evidence) >= 2:
        return "high"
    if retrieval_source in HIGH_CONFIDENCE_SOURCES:
        return "medium"
    if retrieval_source in MEDIUM_CONFIDENCE_SOURCES:
        return "medium"
    if selected_evidence:
        return "low"
    return "very_low"


def _resolve_effective_provider(requested_provider: str) -> Tuple[str, Optional[str]]:
    if requested_provider == "deepseek_chat" and not clean_text(os.environ.get("DEEPSEEK_API_KEY")):
        return "local_fake", "DEEPSEEK_API_KEY missing, fallback to local_fake"
    return requested_provider, None


def _provider_config_path(paths: Dict[str, Path], provider: str) -> Path:
    return paths["deepseek_config_path"] if provider == "deepseek_chat" else paths["local_config_path"]


def _prepare_runtime_llm_config(base_config_path: Path, output_dir: Path, timeout_seconds: int) -> Path:
    with base_config_path.open("r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    config["timeout"] = int(timeout_seconds)
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = output_dir / f".real_sft_runtime_llm_config_{uuid.uuid4().hex}.json"
    runtime_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    return runtime_path


def _build_quality_tags(
    effective_provider: str,
    llm_status: str,
    final_source: str,
    evidence_confidence: str,
    style_examples: Sequence[object],
    rewritten: bool,
) -> List[str]:
    tags = [
        f"provider_{effective_provider}",
        f"llm_status_{llm_status}",
        f"final_source_{final_source}",
        f"evidence_{evidence_confidence}",
    ]
    tags.append("style_conditioned" if style_examples else "style_unconditioned")
    tags.append("rewritten" if rewritten else "direct_rank_top1")
    return tags


def _filter_reason(
    pipeline_output: Dict[str, object],
    effective_provider: str,
    min_score: float,
) -> Optional[str]:
    provider_bundle = dict(pipeline_output.get("llm_provider") or {})
    final_bundle = dict(pipeline_output.get("final_ad_copy") or {})
    ranker_bundle = dict(pipeline_output.get("copy_ranker") or {})
    rewriter_bundle = dict(pipeline_output.get("copy_rewriter") or {})
    evidence_bundle = dict(pipeline_output.get("evidence_selector") or {})
    query_bundle = dict(pipeline_output.get("query_normalization") or {})

    llm_status = clean_text(provider_bundle.get("status"))
    final_source = clean_text(final_bundle.get("final_source"))
    final_candidate = dict(final_bundle.get("final_candidate") or {})
    top_candidate = dict(ranker_bundle.get("top_candidate") or {})
    final_ad_copy = clean_text(final_bundle.get("final_ad_copy"))
    suspicious_claims = _detect_suspicious_claims(final_ad_copy)
    selected_evidence = list(evidence_bundle.get("selected_evidence_items") or [])
    normalized_query = clean_text(query_bundle.get("normalized_query") or query_bundle.get("raw_query"))
    rank_score = float(final_candidate.get("rank_score") or top_candidate.get("rank_score") or 0.0)
    compact_query = _normalize_compare_text(normalized_query)

    if not re.search(r"[a-zA-Z\u4e00-\u9fff]", compact_query):
        return "query_too_noisy"
    if effective_provider == "deepseek_chat" and llm_status != "provider_ok":
        return "provider_not_ok"
    if effective_provider == "local_fake" and llm_status != "local_fake_json_ok":
        return "provider_not_ok"
    if not top_candidate or not bool(top_candidate.get("is_valid")) or list(top_candidate.get("issues") or []):
        return "validator_invalid"
    if final_source in {"summary_fallback", "creative_template_fallback", "template_fallback"}:
        return "fallback_copy_source"
    if suspicious_claims:
        return "suspicious_claims"
    if rank_score < float(min_score):
        return "score_below_min"
    if not final_ad_copy or len(final_ad_copy) < 12 or len(final_ad_copy) > 40:
        return "copy_length_out_of_range"
    if _is_template_like(final_ad_copy):
        return "template_like"
    if not _is_related_to_evidence(final_ad_copy, normalized_query, selected_evidence):
        return "unrelated_to_evidence"
    if clean_text(final_candidate.get("source")) == "llm_rule_rewrite" and len(final_ad_copy) < 14:
        return "rewritten_copy_too_short"
    if rewriter_bundle.get("rewrite_reason") == "rule_rewrite_from_invalid_candidate":
        return "validator_invalid"
    return None


def _build_sft_record(
    pipeline_output: Dict[str, object],
    effective_provider: str,
) -> Dict[str, object]:
    query_bundle = dict(pipeline_output.get("query_normalization") or {})
    intent_bundle = dict(pipeline_output.get("intent_enricher") or {})
    evidence_bundle = dict(pipeline_output.get("evidence_selector") or {})
    user_profile_bundle = dict(pipeline_output.get("user_profile_builder") or {})
    style_bundle = dict(pipeline_output.get("style_retriever") or {})
    provider_bundle = dict(pipeline_output.get("llm_provider") or {})
    final_bundle = dict(pipeline_output.get("final_ad_copy") or {})
    final_candidate = dict(final_bundle.get("final_candidate") or {})

    query = clean_text(query_bundle.get("raw_query") or query_bundle.get("normalized_query"))
    selected_evidence = list(evidence_bundle.get("selected_evidence_items") or [])
    retrieval_source = clean_text(evidence_bundle.get("retrieval_source"))
    evidence_confidence = _estimate_evidence_confidence(retrieval_source, selected_evidence)
    style_examples = list(style_bundle.get("style_examples") or [])
    if not style_examples and style_bundle.get("example_pattern"):
        style_examples = [style_bundle.get("example_pattern")]
    llm_status = clean_text(provider_bundle.get("status"))
    final_source = clean_text(final_bundle.get("final_source"))
    rewritten = bool(dict(pipeline_output.get("copy_rewriter") or {}).get("rewritten"))
    score = float(final_candidate.get("rank_score") or 0.0)

    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": _build_user_message(
                    query=query,
                    intent_bundle=intent_bundle,
                    user_profile_bundle=user_profile_bundle,
                    evidence_bundle=evidence_bundle,
                    style_bundle=style_bundle,
                ),
            },
            {
                "role": "assistant",
                "content": clean_text(final_bundle.get("final_ad_copy")),
            },
        ],
        "metadata": {
            "query": query,
            "provider": effective_provider,
            "score": round(score, 2),
            "evidence_confidence": evidence_confidence,
            "quality_tags": _build_quality_tags(
                effective_provider=effective_provider,
                llm_status=llm_status,
                final_source=final_source,
                evidence_confidence=evidence_confidence,
                style_examples=style_examples,
                rewritten=rewritten,
            ),
        },
    }


def _render_preview_md(
    accepted_records: Sequence[Dict[str, object]],
    requested_provider: str,
    effective_provider: str,
    provider_note: Optional[str],
    scanned_count: int,
    filtered_counts: Counter[str],
    error_counts: Counter[str],
    skipped_counts: Counter[str],
    output_path: Path,
) -> str:
    lines: List[str] = [
        "# Real SFT Ad Copy Preview",
        "",
        f"- requested_provider: `{requested_provider}`",
        f"- effective_provider: `{effective_provider}`",
        f"- generated_samples: `{len(accepted_records)}`",
        f"- scanned_samples: `{scanned_count}`",
        f"- filtered_samples: `{sum(filtered_counts.values())}`",
        f"- error_samples: `{sum(error_counts.values())}`",
        f"- skipped_samples: `{sum(skipped_counts.values())}`",
        f"- output_path: `{output_path.as_posix()}`",
    ]
    if provider_note:
        lines.append(f"- provider_note: `{provider_note}`")

    lines.extend(["", "## Filter Counts", ""])
    if filtered_counts:
        for reason, count in filtered_counts.most_common():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Error Counts", ""])
    if error_counts:
        for reason, count in error_counts.most_common():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Skip Counts", ""])
    if skipped_counts:
        for reason, count in skipped_counts.most_common():
            lines.append(f"- {reason}: {count}")
    else:
        lines.append("- none")

    lines.extend(["", "## Preview Samples", ""])
    for index, record in enumerate(accepted_records[:5], start=1):
        lines.extend(
            [
                f"### Sample {index}",
                "",
                "```json",
                json.dumps(record, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _load_existing_records(path: Path) -> Tuple[List[Dict[str, object]], Set[str]]:
    if not path.exists():
        return [], set()

    records: List[Dict[str, object]] = []
    queries: Set[str] = set()
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        query = clean_text(dict(record.get("metadata") or {}).get("query"))
        if not query:
            continue
        records.append(record)
        queries.add(query)
    return records, queries


def _append_jsonl_record(path: Path, record: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _build_v5_context_without_provider(
    query: str,
    user_id: Optional[int],
    session_id: Optional[int],
    runtime_context: object,
    top_k: int,
    candidate_count: int,
) -> Dict[str, object]:
    retrieval_bundle = retrieve_evidence_bundle_cached(
        query=query,
        runtime_context=runtime_context,
        top_k=top_k,
        user_id=user_id,
        session_id=session_id,
    )
    query_bundle = query_normalization_step(str(retrieval_bundle.get("raw_query") or query))
    retrieval_result = dict(retrieval_bundle.get("result") or {})
    intent_bundle = intent_enricher_step(query_bundle=query_bundle, retrieval_result=retrieval_result)
    evidence_bundle = evidence_selector_step(
        intent_bundle=intent_bundle,
        retrieval_result=retrieval_result,
        top_k=top_k,
    )
    user_profile_bundle = user_profile_builder_step(
        user_id=user_id,
        retrieval_result=retrieval_result,
        users_map=retrieval_bundle.get("users_map"),
    )
    style_bundle = style_retriever_step(
        query_bundle=query_bundle,
        intent_bundle=intent_bundle,
        user_profile_bundle=user_profile_bundle,
        requested_tone="creative",
    )
    prompt_bundle = prompt_builder_step(
        query_bundle=query_bundle,
        intent_bundle=intent_bundle,
        evidence_bundle=evidence_bundle,
        user_profile_bundle=user_profile_bundle,
        style_bundle=style_bundle,
        candidate_count=candidate_count,
    )
    return {
        "query_normalization": query_bundle,
        "intent_enricher": intent_bundle,
        "evidence_selector": evidence_bundle,
        "user_profile_builder": user_profile_bundle,
        "style_retriever": style_bundle,
        "prompt_builder": prompt_bundle,
        "retrieval_bundle": retrieval_bundle,
    }


def _run_pipeline_once(
    query: str,
    user_id: Optional[int],
    session_id: Optional[int],
    paths: Dict[str, Path],
    runtime_context: object,
    llm_config_path: Path,
    effective_provider: str,
    top_k: int,
    candidate_count: int,
) -> Dict[str, object]:
    return v5_llm_dynamic_creative(
        query=query,
        corpus_path=paths["corpus_path"],
        rank_path=paths["rank_path"],
        users_path=paths["users_path"],
        pairs_path=paths["pairs_path"],
        llm_config_path=llm_config_path,
        top_k=top_k,
        candidate_count=candidate_count,
        requested_tone="creative",
        user_id=user_id,
        session_id=session_id,
        runtime_context=runtime_context,
        provider_override=effective_provider,
    )


def _run_with_retries(
    query: str,
    user_id: Optional[int],
    session_id: Optional[int],
    paths: Dict[str, Path],
    runtime_context: object,
    llm_config_path: Path,
    effective_provider: str,
    top_k: int,
    candidate_count: int,
    max_retries: int,
    sleep_seconds: float,
    dry_run_source: bool,
) -> Tuple[Optional[Dict[str, object]], Optional[str], int]:
    attempts = 0
    last_error: Optional[str] = None
    total_attempts = max(1, max_retries + 1)

    for attempt_index in range(total_attempts):
        attempts += 1
        try:
            if dry_run_source:
                return (
                    _build_v5_context_without_provider(
                        query=query,
                        user_id=user_id,
                        session_id=session_id,
                        runtime_context=runtime_context,
                        top_k=top_k,
                        candidate_count=candidate_count,
                    ),
                    None,
                    attempts,
                )

            pipeline_output = _run_pipeline_once(
                query=query,
                user_id=user_id,
                session_id=session_id,
                paths=paths,
                runtime_context=runtime_context,
                llm_config_path=llm_config_path,
                effective_provider=effective_provider,
                top_k=top_k,
                candidate_count=candidate_count,
            )
            llm_status = clean_text(pipeline_output.get("llm_status"))
            if effective_provider == "deepseek_chat" and llm_status == "provider_ok":
                return pipeline_output, None, attempts
            if effective_provider == "local_fake" and llm_status == "local_fake_json_ok":
                return pipeline_output, None, attempts
            last_error = f"llm_status={llm_status or 'unknown'}"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt_index < total_attempts - 1 and sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return None, last_error or "unknown_error", attempts


def _print_progress(
    scanned: int,
    accepted: int,
    filtered_counts: Counter[str],
    error_counts: Counter[str],
    provider: str,
    started_at: float,
    last_status: str,
) -> None:
    elapsed = time.perf_counter() - started_at
    filtered_total = sum(filtered_counts.values())
    error_total = sum(error_counts.values())
    print(
        f"[progress] scanned={scanned} accepted={accepted} filtered={filtered_total} "
        f"errors={error_total} provider={provider} elapsed={elapsed:.1f}s last_status={last_status}"
    )


def main() -> None:
    args = parse_args()
    target_count = _effective_target_count(args)
    max_source = max(1, int(args.max_source))
    top_k = max(1, int(args.top_k))
    candidate_count = int(args.candidate_count)

    paths = _project_paths()
    if not args.pairs_path.exists():
        raise FileNotFoundError(f"train_pairs file not found: {args.pairs_path}")

    output_path = args.output
    preview_path = output_path.with_name("real_sft_ad_copy_preview.md")
    requested_provider = args.provider
    effective_provider, provider_note = _resolve_effective_provider(requested_provider)

    existing_records: List[Dict[str, object]] = []
    existing_queries: Set[str] = set()
    if args.resume:
        existing_records, existing_queries = _load_existing_records(output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

    accepted_records: List[Dict[str, object]] = list(existing_records)
    accepted_queries: Set[str] = set(existing_queries)
    filtered_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()
    skipped_counts: Counter[str] = Counter()
    scanned_count = 0
    started_at = time.perf_counter()

    runtime_context = build_v5_runtime_context(
        corpus_path=paths["corpus_path"],
        rank_path=paths["rank_path"],
        users_path=paths["users_path"],
        pairs_path=args.pairs_path,
    )

    runtime_llm_config_path: Optional[Path] = None
    if not args.dry_run_source:
        base_config_path = _provider_config_path(paths, effective_provider)
        runtime_llm_config_path = _prepare_runtime_llm_config(
            base_config_path=base_config_path,
            output_dir=output_path.parent,
            timeout_seconds=int(args.timeout),
        )

    try:
        if len(accepted_records) >= target_count:
            print(f"Resume found {len(accepted_records)} existing samples, already >= target_count={target_count}.")
        else:
            for record in iter_jsonl(args.pairs_path):
                if len(accepted_records) >= target_count:
                    break
                if scanned_count >= max_source:
                    break

                scanned_count += 1
                query = clean_text(record.get("query"))
                if not query:
                    filtered_counts["empty_query"] += 1
                    _print_progress(
                        scanned=scanned_count,
                        accepted=len(accepted_records),
                        filtered_counts=filtered_counts,
                        error_counts=error_counts,
                        provider=effective_provider,
                        started_at=started_at,
                        last_status="empty_query",
                    )
                    continue

                if query in accepted_queries:
                    skipped_counts["resume_existing_query" if args.resume else "duplicate_query"] += 1
                    _print_progress(
                        scanned=scanned_count,
                        accepted=len(accepted_records),
                        filtered_counts=filtered_counts,
                        error_counts=error_counts,
                        provider=effective_provider,
                        started_at=started_at,
                        last_status="skip_duplicate_query",
                    )
                    continue

                user_id = int(record["user_id"]) if record.get("user_id") is not None else None
                session_id = int(record["session_id"]) if record.get("session_id") is not None else None

                pipeline_output, run_error, attempts = _run_with_retries(
                    query=query,
                    user_id=user_id,
                    session_id=session_id,
                    paths=paths,
                    runtime_context=runtime_context,
                    llm_config_path=runtime_llm_config_path if runtime_llm_config_path is not None else Path(),
                    effective_provider=effective_provider,
                    top_k=top_k,
                    candidate_count=candidate_count,
                    max_retries=max(0, int(args.max_retries)),
                    sleep_seconds=max(0.0, float(args.sleep)),
                    dry_run_source=bool(args.dry_run_source),
                )

                if run_error:
                    error_key = run_error if len(run_error) <= 80 else run_error[:80]
                    error_counts[error_key] += 1
                    _print_progress(
                        scanned=scanned_count,
                        accepted=len(accepted_records),
                        filtered_counts=filtered_counts,
                        error_counts=error_counts,
                        provider=effective_provider,
                        started_at=started_at,
                        last_status=f"error(attempts={attempts})",
                    )
                    continue

                if args.dry_run_source:
                    _print_progress(
                        scanned=scanned_count,
                        accepted=len(accepted_records),
                        filtered_counts=filtered_counts,
                        error_counts=error_counts,
                        provider=effective_provider,
                        started_at=started_at,
                        last_status="dry_run_source_ok",
                    )
                    continue

                assert pipeline_output is not None
                reason = _filter_reason(
                    pipeline_output=pipeline_output,
                    effective_provider=effective_provider,
                    min_score=float(args.min_score),
                )
                if reason:
                    filtered_counts[reason] += 1
                    _print_progress(
                        scanned=scanned_count,
                        accepted=len(accepted_records),
                        filtered_counts=filtered_counts,
                        error_counts=error_counts,
                        provider=effective_provider,
                        started_at=started_at,
                        last_status=reason,
                    )
                    continue

                sft_record = _build_sft_record(
                    pipeline_output=pipeline_output,
                    effective_provider=effective_provider,
                )
                accepted_records.append(sft_record)
                accepted_queries.add(query)
                _append_jsonl_record(output_path, sft_record)
                _print_progress(
                    scanned=scanned_count,
                    accepted=len(accepted_records),
                    filtered_counts=filtered_counts,
                    error_counts=error_counts,
                    provider=effective_provider,
                    started_at=started_at,
                    last_status="accepted",
                )

        preview_text = _render_preview_md(
            accepted_records=accepted_records,
            requested_provider=requested_provider,
            effective_provider=effective_provider,
            provider_note=provider_note,
            scanned_count=scanned_count,
            filtered_counts=filtered_counts,
            error_counts=error_counts,
            skipped_counts=skipped_counts,
            output_path=output_path,
        )
        preview_path.write_text(preview_text, encoding="utf-8")

        print("")
        print(f"Generated samples: {len(accepted_records)}")
        print(f"Scanned samples: {scanned_count}")
        print(f"Filtered samples: {sum(filtered_counts.values())}")
        print(f"Error samples: {sum(error_counts.values())}")
        print(f"Skipped samples: {sum(skipped_counts.values())}")
        if filtered_counts:
            print("Filtered reasons:")
            for reason, count in filtered_counts.most_common():
                print(f"- {reason}: {count}")
        if error_counts:
            print("Error reasons:")
            for reason, count in error_counts.most_common():
                print(f"- {reason}: {count}")
        if skipped_counts:
            print("Skipped reasons:")
            for reason, count in skipped_counts.most_common():
                print(f"- {reason}: {count}")
        print(f"Output JSONL: {output_path}")
        print(f"Output preview: {preview_path}")
        if args.dry_run_source:
            print("Dry run mode: provider was not called, output JSONL was not appended.")
    finally:
        if runtime_llm_config_path is not None and runtime_llm_config_path.exists():
            try:
                runtime_llm_config_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
