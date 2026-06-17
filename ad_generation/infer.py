"""Run ad generation inference from local evidence.

The inference order is:
1. train_pairs user exact
2. train_pairs session exact
3. train_pairs query exact
4. rank user/session exact
5. rank query exact
6. BM25 integration attempt from recall/BM25/bm25.py
7. local lexical overlap fallback

The final output supports template, summary, and LLM-backed copy styles.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from hashlib import md5
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from build_train_pairs import (
    build_query_record,
    canonicalize_item,
    clean_text,
    compute_sample_ranking_signal,
    first_non_empty,
    is_positive_record,
    iter_jsonl,
    load_corpus_map,
    load_user_map,
    safe_float,
    safe_int,
)
from llm_generator import fallback_generate, generate_with_llm, load_llm_config
from prompt_builder import (
    build_llm_prompt,
    build_prompt,
    build_user_context,
    render_creative_template_copy,
    render_summary_copy,
    render_template_copy,
)

LEGACY_COPY_STYLES = ["template", "summary", "llm"]
V5_GENERATION_MODES = ["llm_full_pipeline", "v5_llm_dynamic_creative"]
ALL_GENERATION_MODES = LEGACY_COPY_STYLES + V5_GENERATION_MODES


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Infer one second-version Chinese ad copy from query evidence.")
    parser.add_argument("--query", required=True, help="Input query text.")
    parser.add_argument("--user_id", type=int, default=None, help="Optional user_id for weak personalization.")
    parser.add_argument("--session_id", type=int, default=None, help="Optional session_id for weak personalization.")
    parser.add_argument("--corpus_path", type=Path, default=Path("data/corpus.jsonl"), help="Path to corpus/items JSONL.")
    parser.add_argument("--rank_path", type=Path, default=Path("data/rank.jsonl"), help="Path to rank JSONL.")
    parser.add_argument("--users_path", type=Path, default=Path("data/users.jsonl"), help="Optional path to users JSONL.")
    parser.add_argument(
        "--pairs_path",
        type=Path,
        default=Path("ad_generation/data/train_pairs.jsonl"),
        help="Path to prebuilt train_pairs JSONL.",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of evidence items to use.")
    parser.add_argument(
        "--copy_tone",
        choices=["safe", "creative", "concise"],
        default="creative",
        help="Preferred tone for LLM copy generation. Default is `creative`.",
    )
    parser.add_argument(
        "--copy_style",
        choices=ALL_GENERATION_MODES,
        default="summary",
        help=(
            "Copy generation style. Legacy modes: template/summary/llm. "
            "V5 modes: llm_full_pipeline/v5_llm_dynamic_creative."
        ),
    )
    parser.add_argument(
        "--generation_mode",
        choices=ALL_GENERATION_MODES,
        default=None,
        help="Optional alias for copy_style. When provided, it overrides --copy_style.",
    )
    parser.add_argument(
        "--llm_config",
        type=Path,
        default=Path("ad_generation/llm_config.example.json"),
        help="Path to LLM config JSON. Used when `--copy_style llm`.",
    )
    parser.add_argument(
        "--mode",
        choices=["template", "prompt_only"],
        default="template",
        help="template outputs copy, prompt_only prints the prompt and a template preview.",
    )
    return parser.parse_args()


def resolve_generation_mode(copy_style: str, generation_mode: Optional[str] = None) -> str:
    """Resolve the effective generation mode, letting `generation_mode` override `copy_style`."""

    selected_mode = clean_text(generation_mode) or clean_text(copy_style) or "summary"
    if selected_mode not in ALL_GENERATION_MODES:
        raise ValueError(f"Unsupported generation mode: {selected_mode}")
    return selected_mode


def normalize_query(raw_query: str) -> str:
    """Normalize raw query text while preserving the main Chinese/English/number body."""

    query = clean_text(raw_query)
    query = re.sub(r"\s+", " ", query)

    leading_chars = "\"'“”‘’#＃([（【{"
    trailing_chars = "\"'“”‘’#＃)]）】}"

    while query and query[0] in leading_chars:
        query = query[1:].lstrip()
    while query and query[-1] in trailing_chars:
        query = query[:-1].rstrip()

    query = re.sub(r"\s+", " ", query).strip()
    return query or clean_text(raw_query)


def _build_result_payload(
    query: str,
    source: str,
    evidence_items: List[Dict[str, object]],
    user_profile: Optional[Dict[str, object]] = None,
    recent_behavior_titles: Optional[List[str]] = None,
    target_item: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Build a uniform inference payload."""

    evidence_items = evidence_items or []
    return {
        "query": query,
        "user_profile": user_profile or {},
        "recent_behavior_titles": recent_behavior_titles or [],
        "evidence_items": evidence_items,
        "target_item": target_item or (evidence_items[0] if evidence_items else {}),
        "source": source,
    }


def load_pair_hit(
    pairs_path: Path,
    normalized_query: str,
    top_k: int,
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    """Load the best matching record from `train_pairs.jsonl` by priority."""

    if not pairs_path.exists():
        return None

    user_match = None
    session_match = None
    query_match = None

    for record in iter_jsonl(pairs_path):
        record_query = first_non_empty(record, ["query"])
        if normalize_query(record_query) != normalized_query:
            continue

        evidence_items = (record.get("evidence_items") or [])[:top_k]
        target_item = record.get("target_item") or (evidence_items[0] if evidence_items else {})
        payload = _build_result_payload(
            query=record_query,
            source="train_pairs_query_exact",
            evidence_items=evidence_items,
            user_profile=record.get("user_profile") or {},
            recent_behavior_titles=record.get("recent_behavior_titles") or [],
            target_item=target_item,
        )

        record_user_id = safe_int(record.get("user_id"))
        record_session_id = safe_int(record.get("session_id"))
        if user_id is not None and record_user_id == user_id and user_match is None:
            payload_user = dict(payload)
            payload_user["source"] = "train_pairs_user_exact"
            user_match = payload_user
        if session_id is not None and record_session_id == session_id and session_match is None:
            payload_session = dict(payload)
            payload_session["source"] = "train_pairs_session_exact"
            session_match = payload_session
        if query_match is None:
            query_match = payload

    if user_match is not None:
        return user_match
    if session_match is not None:
        return session_match
    return query_match


def _empty_group(query: str) -> Dict[str, object]:
    """Create an empty aggregation group compatible with build_query_record."""

    return {
        "query": query,
        "items": {},
        "best_record": None,
        "best_record_key": None,
    }


def _add_rank_record_to_group(group: Dict[str, object], record: Dict[str, object]) -> None:
    """Aggregate one positive rank record into a query group."""

    item_id = safe_int(record.get("target_item_id", record.get("item_id")))
    if item_id <= 0:
        return

    item_entry = group["items"].setdefault(
        item_id,
        {
            "item_id": item_id,
            "purchase_cnt": 0,
            "click_cnt": 0,
            "ranking_signal": 0.0,
        },
    )

    purchase = safe_int(record.get("is_purchased"))
    click = safe_int(record.get("is_clicked"))
    sample_signal = compute_sample_ranking_signal(record)

    item_entry["purchase_cnt"] += purchase
    item_entry["click_cnt"] += click
    item_entry["ranking_signal"] += sample_signal

    candidate_best_key = (
        sample_signal,
        purchase,
        click,
        -item_id,
    )
    if group["best_record"] is None or candidate_best_key > group["best_record_key"]:
        group["best_record"] = record
        group["best_record_key"] = candidate_best_key


def load_rank_hit(
    normalized_query: str,
    rank_path: Path,
    corpus_map: Dict[int, Dict[str, object]],
    users_map: Optional[Dict[int, Dict[str, object]]],
    top_k: int,
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    """Build the best matching record directly from positive rank logs."""

    if not rank_path.exists():
        return None

    query_group = _empty_group(normalized_query)
    user_group = _empty_group(normalized_query)
    session_group = _empty_group(normalized_query)

    for record in iter_jsonl(rank_path):
        if not is_positive_record(record):
            continue

        record_query = first_non_empty(record, ["query"])
        if normalize_query(record_query) != normalized_query:
            continue

        _add_rank_record_to_group(query_group, record)
        if user_id is not None and safe_int(record.get("user_id")) == user_id:
            _add_rank_record_to_group(user_group, record)
        if session_id is not None and safe_int(record.get("session_id")) == session_id:
            _add_rank_record_to_group(session_group, record)

    if user_id is not None and user_group["items"] and query_group["best_record"] is not None:
        record = build_query_record(
            query=query_group["best_record"].get("query", normalized_query),
            group=user_group,
            corpus_map=corpus_map,
            users_map=users_map,
            top_k=top_k,
        )
        record["source"] = "rank_user_exact"
        return record

    if session_id is not None and session_group["items"] and query_group["best_record"] is not None:
        record = build_query_record(
            query=query_group["best_record"].get("query", normalized_query),
            group=session_group,
            corpus_map=corpus_map,
            users_map=users_map,
            top_k=top_k,
        )
        record["source"] = "rank_session_exact"
        return record

    if query_group["items"] and query_group["best_record"] is not None:
        record = build_query_record(
            query=query_group["best_record"].get("query", normalized_query),
            group=query_group,
            corpus_map=corpus_map,
            users_map=users_map,
            top_k=top_k,
        )
        record["source"] = "rank_query_exact"
        return record

    return None


def try_bm25_hit(
    query: str,
    corpus_path: Path,
    corpus_map: Dict[int, Dict[str, object]],
    top_k: int,
) -> Tuple[Optional[Dict[str, object]], str]:
    """Try integrating with the legacy BM25 evaluator."""

    bm25_dir = PROJECT_ROOT / "recall" / "BM25"
    if str(bm25_dir) not in sys.path:
        sys.path.insert(0, str(bm25_dir))

    try:
        import bm25 as legacy_bm25  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"legacy_bm25_import_failed: {exc}"

    try:
        evaluator = legacy_bm25.Evaluator()
        index_name = f"adgen_{md5(str(corpus_path).encode('utf-8')).hexdigest()[:8]}"
        evaluator.build_index(str(corpus_path), index_name)
        evaluator.queries = {"adhoc_query": query}
        results = evaluator.run_retrieval(top_k=top_k)
        hits = results.get("adhoc_query", [])
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"legacy_bm25_runtime_failed: {exc}"

    evidence_items: List[Dict[str, object]] = []
    for hit in hits:
        item_id = safe_int(hit.get("doc_id"))
        if item_id <= 0:
            continue
        item_meta = corpus_map.get(item_id, {"item_id": item_id})
        evidence_items.append(
            {
                **canonicalize_item(item_meta),
                "ranking_signal": round(safe_float(hit.get("score")), 6),
                "purchase_cnt": 0,
                "click_cnt": 0,
                "relevance_signal": 0,
            }
        )

    if not evidence_items:
        return None, "legacy_bm25_empty_result"

    return _build_result_payload(
        query=query,
        source="legacy_bm25",
        evidence_items=evidence_items[:top_k],
    ), "legacy_bm25_ok"


def tokenize_for_overlap(text: str) -> List[str]:
    """Tokenize mixed Chinese / alnum text for lexical-overlap fallback."""

    clean = clean_text(text).lower()
    if not clean:
        return []

    pieces = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", clean)
    return [piece for piece in pieces if piece]


def lexical_overlap_hit(
    query: str,
    corpus_map: Dict[int, Dict[str, object]],
    top_k: int,
) -> Optional[Dict[str, object]]:
    """Fallback retrieval using simple lexical overlap over item text fields."""

    query_tokens = tokenize_for_overlap(query)
    if not query_tokens:
        return None

    query_token_set = set(query_tokens)
    scored_items: List[Tuple[float, int, Dict[str, object]]] = []

    for item_id, item in corpus_map.items():
        haystack = " ".join(
            [
                clean_text(item.get("item_title")),
                clean_text(item.get("brand_name")),
                clean_text(item.get("seller_name")),
                clean_text(item.get("category_path")),
            ]
        )
        doc_tokens = tokenize_for_overlap(haystack)
        if not doc_tokens:
            continue

        doc_token_set = set(doc_tokens)
        overlap = len(query_token_set & doc_token_set)
        if overlap <= 0:
            continue

        score = overlap + 0.001 * len(doc_tokens)
        scored_items.append((score, item_id, item))

    if not scored_items:
        return None

    scored_items.sort(key=lambda entry: (-entry[0], entry[1]))
    evidence_items: List[Dict[str, object]] = []
    for score, item_id, item in scored_items[:top_k]:
        evidence_items.append(
            {
                **canonicalize_item(item),
                "ranking_signal": round(score, 6),
                "purchase_cnt": 0,
                "click_cnt": 0,
                "relevance_signal": 0,
            }
        )

    return _build_result_payload(
        query=query,
        source="lexical_overlap",
        evidence_items=evidence_items,
    )


def pretty_print_evidence(evidence_items: Iterable[Dict[str, object]]) -> str:
    """Serialize evidence items for terminal display."""

    return json.dumps(list(evidence_items), ensure_ascii=False, indent=2)


def _build_summary_result(
    query: str,
    evidence_items: List[Dict[str, object]],
    user_profile: Dict[str, object],
) -> Dict[str, object]:
    """Run summary generation and return copy/fallback metadata."""

    summary_result = render_summary_copy(
        query=query,
        evidence_items=evidence_items,
        user_profile=user_profile,
    )
    return {
        "copy": summary_result["copy"],
        "fallback_triggered": bool(summary_result.get("fallback")),
        "fallback_reason": str(summary_result.get("reason") or ""),
    }


def is_bad_ad_copy(copy_text: str) -> Tuple[bool, str]:
    """Detect overly template-like LLM copy that reads like search-result explanation."""

    text = clean_text(copy_text)
    bad_patterns = [
        "当前候选主要集中",
        "建议先从这类商品里继续筛选",
        "更贴近当前需求",
        "信息清楚，适合先了解",
        "可先关注",
        "等商品",
    ]
    for pattern in bad_patterns:
        if pattern in text:
            return True, pattern
    return False, ""


def format_output_bundle(output: Dict[str, object]) -> str:
    """Format an inference payload for terminal display."""

    lines = [
        "=== Raw Query ===",
        str(output.get("raw_query", "")),
        "",
        "=== Normalized Query ===",
        str(output.get("normalized_query", "")),
        "",
        "=== Retrieval Source ===",
        str(output.get("retrieval_source", "unknown")),
    ]
    bm25_reason = str(output.get("bm25_status") or "")
    if bm25_reason and output.get("retrieval_source") != "legacy_bm25":
        lines.append(f"BM25 status: {bm25_reason}")
    lines.extend(
        [
            "",
            "=== Copy Style ===",
            str(output.get("copy_style", "template")),
            f"copy_tone={str(output.get('copy_tone') or '(none)')}",
            f"fallback_triggered={bool(output.get('fallback_triggered'))}",
            f"fallback_reason={str(output.get('fallback_reason') or '(none)')}",
            f"llm_status={str(output.get('llm_status') or '(none)')}",
            f"llm_provider={str(output.get('llm_provider') or '(none)')}",
            f"llm_rewrite_attempted={bool(output.get('llm_rewrite_attempted'))}",
            f"bad_copy_detected={bool(output.get('bad_copy_detected'))}",
            f"bad_copy_reason={str(output.get('bad_copy_reason') or '(none)')}",
            f"suspicious_claims={str(output.get('suspicious_claims') or [])}",
            "",
            "=== Top-K Evidence Items ===",
            pretty_print_evidence(output.get("evidence_items") or []),
            "",
            "=== User Context ===",
            str(output.get("user_context") or "(empty)"),
            "",
            "=== Final Prompt ===",
            str(output.get("final_prompt") or ""),
            "",
            "=== Final Ad Copy ===",
            str(output.get("final_ad_copy") or ""),
        ]
    )
    if output.get("intent") is not None:
        lines.extend(
            [
                "",
                "=== Intent ===",
                json.dumps(output.get("intent") or {}, ensure_ascii=False, indent=2),
                "",
                "=== Selected Evidence ===",
                pretty_print_evidence(output.get("selected_evidence") or []),
                "",
                "=== Retrieved Style Examples ===",
                json.dumps(output.get("retrieved_style_examples") or [], ensure_ascii=False, indent=2),
                "",
                "=== Prompt Version ===",
                str(output.get("prompt_version") or ""),
                "",
                "=== Raw LLM Output ===",
                json.dumps(output.get("raw_llm_output") or [], ensure_ascii=False, indent=2),
                "",
                "=== Parsed Candidates ===",
                json.dumps(output.get("parsed_candidates") or [], ensure_ascii=False, indent=2),
                "",
                "=== Validated Candidates ===",
                json.dumps(output.get("validated_candidates") or [], ensure_ascii=False, indent=2),
                "",
                "=== Ranked Candidates ===",
                json.dumps(output.get("ranked_candidates") or [], ensure_ascii=False, indent=2),
                "",
                "=== Rewritten Candidates ===",
                json.dumps(output.get("rewritten_candidates") or [], ensure_ascii=False, indent=2),
                "",
                "=== Best Copy ===",
                json.dumps(output.get("best_copy") or {}, ensure_ascii=False, indent=2),
            ]
        )
    return "\n".join(lines)


def build_generation_output(
    raw_query: str,
    result: Dict[str, object],
    copy_style: str = "summary",
    copy_tone: str = "creative",
    user_id: Optional[int] = None,
    users_map: Optional[Dict[int, Dict[str, object]]] = None,
    bm25_status: str = "",
    llm_config_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Convert a retrieval result into a reusable generation payload."""

    normalized_query = normalize_query(raw_query)
    if not normalized_query:
        raise ValueError("Query must not be empty.")

    final_user_profile = dict(result.get("user_profile") or {})
    if not final_user_profile and user_id is not None and users_map is not None:
        final_user_profile = dict(users_map.get(user_id, {}))

    final_recent_behavior_titles = list(result.get("recent_behavior_titles") or [])
    user_context = build_user_context(
        user_profile=final_user_profile,
        recent_behavior_titles=final_recent_behavior_titles,
    )

    prompt = build_prompt(
        query=normalized_query,
        evidence_items=result.get("evidence_items") or [],
        user_profile=final_user_profile,
        recent_behavior_titles=final_recent_behavior_titles,
    )
    llm_prompt = build_llm_prompt(
        query=normalized_query,
        evidence_items=result.get("evidence_items") or [],
        user_profile=final_user_profile,
        recent_behavior_titles=final_recent_behavior_titles,
        copy_tone=copy_tone,
    )
    fallback_triggered = False
    fallback_reason = ""
    effective_copy_style = copy_style
    llm_status = ""
    llm_provider = ""
    llm_rewrite_attempted = False
    bad_copy_detected = False
    bad_copy_reason = ""
    if copy_style == "summary":
        summary_result = _build_summary_result(
            query=normalized_query,
            evidence_items=list(result.get("evidence_items") or []),
            user_profile=final_user_profile,
        )
        final_copy = summary_result["copy"]
        fallback_triggered = summary_result["fallback_triggered"]
        fallback_reason = summary_result["fallback_reason"]
    elif copy_style == "llm":
        evidence_items = list(result.get("evidence_items") or [])
        summary_result = _build_summary_result(
            query=normalized_query,
            evidence_items=evidence_items,
            user_profile=final_user_profile,
        )
        template_copy = render_template_copy(
            query=normalized_query,
            item=result.get("target_item") or {},
            user_profile=final_user_profile,
        )
        creative_template_copy = render_creative_template_copy(
            query=normalized_query,
            evidence_items=evidence_items,
            item=result.get("target_item") or {},
            user_profile=final_user_profile,
        )
        llm_provider = "mock"
        try:
            if llm_config_path is None:
                raise FileNotFoundError("No llm_config path provided.")
            llm_config = load_llm_config(llm_config_path)
            llm_provider = clean_text(llm_config.get("provider")) or "mock"
            llm_candidate = generate_with_llm(llm_prompt, llm_config)
            if copy_tone == "creative":
                bad_copy_detected, bad_copy_reason = is_bad_ad_copy(llm_candidate)
                if bad_copy_detected:
                    llm_rewrite_attempted = True
                    rewrite_prompt = build_llm_prompt(
                        query=normalized_query,
                        evidence_items=evidence_items,
                        user_profile=final_user_profile,
                        recent_behavior_titles=final_recent_behavior_titles,
                        copy_tone=copy_tone,
                        rewrite_request=True,
                    )
                    rewritten_candidate = generate_with_llm(rewrite_prompt, llm_config)
                    rewritten_bad, rewritten_reason = is_bad_ad_copy(rewritten_candidate)
                    if rewritten_bad:
                        bad_copy_reason = rewritten_reason or bad_copy_reason
                        raise RuntimeError(f"bad_ad_copy_detected:{bad_copy_reason}")
                    final_copy = rewritten_candidate
                    llm_status = "llm_ok_rewritten"
                else:
                    final_copy = llm_candidate
                    llm_status = "llm_ok"
            else:
                final_copy = llm_candidate
                llm_status = "llm_ok"
        except Exception as exc:
            llm_status = f"llm_failed: {exc}"
            fallback_triggered = True
            if copy_tone == "creative":
                if bad_copy_detected and str(exc).startswith("bad_ad_copy_detected:"):
                    fallback_reason = "llm_bad_copy_then_creative_template"
                else:
                    fallback_reason = "llm_failed_then_creative_template"
                final_copy = creative_template_copy or template_copy or fallback_generate(llm_prompt, evidence_items, normalized_query)
            elif bad_copy_detected and str(exc).startswith("bad_ad_copy_detected:"):
                if summary_result["fallback_triggered"]:
                    fallback_reason = f"llm_bad_copy_then_summary_{summary_result['fallback_reason']}"
                    final_copy = template_copy if template_copy else fallback_generate(llm_prompt, evidence_items, normalized_query)
                else:
                    fallback_reason = "llm_bad_copy_then_summary_ok"
                    final_copy = summary_result["copy"]
            elif summary_result["fallback_triggered"]:
                fallback_reason = f"llm_failed_then_summary_{summary_result['fallback_reason']}"
                final_copy = template_copy if template_copy else fallback_generate(llm_prompt, evidence_items, normalized_query)
            else:
                fallback_reason = "llm_failed_then_summary_ok"
                final_copy = summary_result["copy"]
    else:
        final_copy = render_template_copy(
            query=normalized_query,
            item=result.get("target_item") or {},
            user_profile=final_user_profile,
        )
    return {
        "raw_query": raw_query,
        "normalized_query": normalized_query,
        "retrieval_source": result.get("source", "unknown"),
        "copy_style": effective_copy_style,
        "copy_tone": copy_tone if copy_style == "llm" else "",
        "fallback_triggered": fallback_triggered,
        "fallback_reason": fallback_reason,
        "bm25_status": bm25_status,
        "llm_status": llm_status,
        "llm_provider": llm_provider,
        "llm_rewrite_attempted": llm_rewrite_attempted,
        "bad_copy_detected": bad_copy_detected,
        "bad_copy_reason": bad_copy_reason,
        "suspicious_claims": [],
        "evidence_items": list(result.get("evidence_items") or []),
        "user_context": user_context,
        "user_profile": final_user_profile,
        "recent_behavior_titles": final_recent_behavior_titles,
        "final_prompt": llm_prompt if copy_style == "llm" else prompt,
        "final_ad_copy": final_copy,
        "target_item": result.get("target_item") or {},
    }


def retrieve_evidence_bundle(
    query: str,
    corpus_path: Path,
    rank_path: Path,
    users_path: Optional[Path],
    pairs_path: Optional[Path],
    top_k: int = 3,
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
) -> Dict[str, object]:
    """Resolve evidence only, without running any copy-generation strategy."""

    raw_query = clean_text(query)
    normalized_query = normalize_query(raw_query)
    if not normalized_query:
        raise ValueError("Query must not be empty.")

    users_map: Optional[Dict[int, Dict[str, object]]] = None
    result = None
    bm25_reason = ""

    if pairs_path is not None:
        result = load_pair_hit(
            pairs_path,
            normalized_query=normalized_query,
            top_k=max(1, top_k),
            user_id=user_id,
            session_id=session_id,
        )

    corpus_map: Optional[Dict[int, Dict[str, object]]] = None
    if result is None:
        if not corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
        if not rank_path.exists():
            raise FileNotFoundError(f"Rank file not found: {rank_path}")

        corpus_map = load_corpus_map(corpus_path)
        users_map = load_user_map(users_path if users_path and users_path.exists() else None)
        result = load_rank_hit(
            normalized_query=normalized_query,
            rank_path=rank_path,
            corpus_map=corpus_map,
            users_map=users_map,
            top_k=max(1, top_k),
            user_id=user_id,
            session_id=session_id,
        )

    if result is None:
        if corpus_map is None:
            corpus_map = load_corpus_map(corpus_path)
        result, bm25_reason = try_bm25_hit(
            query=normalized_query,
            corpus_path=corpus_path,
            corpus_map=corpus_map,
            top_k=max(1, top_k),
        )

    if result is None:
        if corpus_map is None:
            corpus_map = load_corpus_map(corpus_path)
        result = lexical_overlap_hit(query=normalized_query, corpus_map=corpus_map, top_k=max(1, top_k))

    if result is None:
        raise RuntimeError("No evidence items found from train_pairs, rank logs, BM25, or lexical overlap fallback.")

    if users_map is None and user_id is not None and users_path is not None and users_path.exists():
        users_map = load_user_map(users_path)

    return {
        "raw_query": raw_query,
        "normalized_query": normalized_query,
        "result": result,
        "users_map": users_map,
        "bm25_status": bm25_reason,
    }


def infer_once(
    query: str,
    corpus_path: Path,
    rank_path: Path,
    users_path: Optional[Path],
    pairs_path: Optional[Path],
    top_k: int = 3,
    copy_style: str = "summary",
    generation_mode: Optional[str] = None,
    copy_tone: str = "creative",
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
    llm_config_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Run one inference and return a structured payload for CLI or evaluation."""

    effective_mode = resolve_generation_mode(copy_style=copy_style, generation_mode=generation_mode)

    if effective_mode in V5_GENERATION_MODES:
        from v5_infer_adapter import build_v5_infer_payload

        return build_v5_infer_payload(
            query=query,
            corpus_path=corpus_path,
            rank_path=rank_path,
            users_path=users_path,
            pairs_path=pairs_path,
            llm_config_path=llm_config_path,
            top_k=top_k,
            copy_tone=copy_tone,
            user_id=user_id,
            session_id=session_id,
            generation_mode=effective_mode,
        )

    retrieval_bundle = retrieve_evidence_bundle(
        query=query,
        corpus_path=corpus_path,
        rank_path=rank_path,
        users_path=users_path,
        pairs_path=pairs_path,
        top_k=top_k,
        user_id=user_id,
        session_id=session_id,
    )

    return build_generation_output(
        raw_query=str(retrieval_bundle["raw_query"]),
        result=dict(retrieval_bundle["result"]),
        copy_style=effective_mode,
        copy_tone=copy_tone,
        user_id=user_id,
        users_map=retrieval_bundle.get("users_map"),
        bm25_status=str(retrieval_bundle.get("bm25_status") or ""),
        llm_config_path=llm_config_path,
    )


def main() -> None:
    """Resolve evidence for the query and print the second-version ad output."""

    args = parse_args()
    output = infer_once(
        query=args.query,
        corpus_path=args.corpus_path,
        rank_path=args.rank_path,
        users_path=args.users_path,
        pairs_path=args.pairs_path,
        top_k=args.top_k,
        copy_style=args.copy_style,
        generation_mode=args.generation_mode,
        copy_tone=args.copy_tone,
        user_id=args.user_id,
        session_id=args.session_id,
        llm_config_path=args.llm_config,
    )
    print(format_output_bundle(output))


if __name__ == "__main__":
    main()
