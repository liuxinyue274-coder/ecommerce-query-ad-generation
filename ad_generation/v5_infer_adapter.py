"""Adapter from V5 dynamic creative output to infer.py-compatible payload."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from build_train_pairs import clean_text
from v5_dynamic_creative import v5_llm_dynamic_creative


SUSPICIOUS_TOKENS = [
    "元",
    "¥",
    "折",
    "优惠",
    "立减",
    "销量",
    "热销",
    "爆款",
    "改善",
    "修复",
    "药效",
    "功效",
    "续航",
    "认证",
]


def detect_suspicious_claims(copy_text: str) -> List[str]:
    """Flag lightweight signs of unsupported pricing/sales/efficacy claims."""

    findings: List[str] = []
    text = clean_text(copy_text)
    for token in SUSPICIOUS_TOKENS:
        if token in text:
            findings.append(token)
    return findings


def build_v5_infer_payload(
    query: str,
    corpus_path: Path,
    rank_path: Path,
    users_path: Optional[Path],
    pairs_path: Optional[Path],
    llm_config_path: Optional[Path] = None,
    top_k: int = 3,
    copy_tone: str = "creative",
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
    generation_mode: str = "v5_llm_dynamic_creative",
) -> Dict[str, object]:
    """Run V5 and adapt its output to infer.py's output contract plus V5 fields."""

    pipeline_output = v5_llm_dynamic_creative(
        query=query,
        corpus_path=corpus_path,
        rank_path=rank_path,
        users_path=users_path,
        pairs_path=pairs_path,
        llm_config_path=llm_config_path,
        top_k=top_k,
        candidate_count=5,
        requested_tone=copy_tone,
        user_id=user_id,
        session_id=session_id,
    )

    provider_bundle = dict(pipeline_output.get("llm_provider") or {})
    prompt_bundle = dict(pipeline_output.get("prompt_builder") or {})
    query_bundle = dict(pipeline_output.get("query_normalization") or {})
    intent_bundle = dict(pipeline_output.get("intent_enricher") or {})
    evidence_bundle = dict(pipeline_output.get("evidence_selector") or {})
    user_profile_bundle = dict(pipeline_output.get("user_profile_builder") or {})
    style_bundle = dict(pipeline_output.get("style_retriever") or {})
    parser_bundle = dict(pipeline_output.get("llm_output_parser") or {})
    validator_bundle = dict(pipeline_output.get("copy_validator") or {})
    ranker_bundle = dict(pipeline_output.get("copy_ranker") or {})
    rewriter_bundle = dict(pipeline_output.get("copy_rewriter") or {})
    final_bundle = dict(pipeline_output.get("final_ad_copy") or {})
    retrieval_bundle = dict(pipeline_output.get("retrieval_bundle") or {})
    retrieval_result = dict(retrieval_bundle.get("result") or {})

    final_ad_copy = str(final_bundle.get("final_ad_copy") or "")
    final_source = str(final_bundle.get("final_source") or "")
    suspicious_claims = detect_suspicious_claims(final_ad_copy)

    validated_candidates = list(validator_bundle.get("validated_candidates") or [])
    final_candidate = dict(final_bundle.get("final_candidate") or {})
    bad_copy_detected = bool(final_candidate.get("issues"))
    fallback_reason = str(rewriter_bundle.get("rewrite_reason") or "")
    fallback_triggered = bool(provider_bundle.get("fallback_used")) or final_source != "llm_ranker_top1"
    if provider_bundle.get("fallback_used") and fallback_reason == "top_candidate_kept":
        fallback_reason = f"provider_{provider_bundle.get('status') or 'fallback'}"

    retrieved_style_examples = []
    if style_bundle.get("style_examples"):
        retrieved_style_examples = list(style_bundle.get("style_examples") or [])
    elif style_bundle.get("example_pattern"):
        retrieved_style_examples.append(style_bundle["example_pattern"])

    rewritten_candidates = []
    if rewriter_bundle.get("final_candidate"):
        rewritten_candidates.append(rewriter_bundle["final_candidate"])

    return {
        "raw_query": str(query_bundle.get("raw_query") or query),
        "normalized_query": str(query_bundle.get("normalized_query") or ""),
        "retrieval_source": str(evidence_bundle.get("retrieval_source") or retrieval_result.get("source") or "unknown"),
        "copy_style": generation_mode,
        "copy_tone": copy_tone,
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
        "bm25_status": str(retrieval_bundle.get("bm25_status") or ""),
        "llm_status": str(provider_bundle.get("status") or ""),
        "llm_provider": str(provider_bundle.get("provider") or ""),
        "llm_rewrite_attempted": bool(rewriter_bundle.get("rewritten")),
        "bad_copy_detected": bad_copy_detected,
        "bad_copy_reason": ", ".join(final_candidate.get("issues") or []) if bad_copy_detected else "",
        "suspicious_claims": suspicious_claims,
        "evidence_items": list(evidence_bundle.get("selected_evidence_items") or []),
        "user_context": str(user_profile_bundle.get("persona_summary") or ""),
        "user_profile": dict(user_profile_bundle.get("user_profile_raw") or {}),
        "recent_behavior_titles": list(user_profile_bundle.get("recent_behavior_titles") or []),
        "final_prompt": str(prompt_bundle.get("full_prompt") or ""),
        "final_ad_copy": final_ad_copy,
        "target_item": dict(evidence_bundle.get("anchor_item") or retrieval_result.get("target_item") or {}),
        "intent": intent_bundle,
        "selected_evidence": list(evidence_bundle.get("selected_evidence_items") or []),
        "retrieved_style_examples": retrieved_style_examples,
        "prompt_version": "v5_dynamic_creative",
        "raw_llm_output": list(provider_bundle.get("raw_generations") or []),
        "parsed_candidates": list(parser_bundle.get("parsed_candidates") or []),
        "validated_candidates": validated_candidates,
        "ranked_candidates": list(ranker_bundle.get("ranked_candidates") or []),
        "rewritten_candidates": rewritten_candidates,
        "best_copy": final_candidate,
        "pipeline_output": pipeline_output,
    }
