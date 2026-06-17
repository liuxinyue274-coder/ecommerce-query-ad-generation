"""Runtime cache for V5 dynamic creative pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from build_train_pairs import (
    build_query_record,
    collect_query_groups,
    first_non_empty,
    iter_jsonl,
    load_corpus_map,
    load_user_map,
    safe_int,
)
from infer import lexical_overlap_hit, normalize_query, retrieve_evidence_bundle
import re


def _build_result_payload(
    query: str,
    source: str,
    evidence_items: List[Dict[str, object]],
    user_profile: Optional[Dict[str, object]] = None,
    recent_behavior_titles: Optional[List[str]] = None,
    target_item: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    evidence_items = evidence_items or []
    return {
        "query": query,
        "user_profile": user_profile or {},
        "recent_behavior_titles": recent_behavior_titles or [],
        "evidence_items": evidence_items,
        "target_item": target_item or (evidence_items[0] if evidence_items else {}),
        "source": source,
    }


@dataclass
class V5RuntimeContext:
    corpus_path: Path
    rank_path: Path
    users_path: Optional[Path]
    pairs_path: Optional[Path]
    corpus_map: Dict[int, Dict[str, object]] = field(default_factory=dict)
    users_map: Dict[int, Dict[str, object]] = field(default_factory=dict)
    pair_query_index: Dict[str, List[Dict[str, object]]] = field(default_factory=dict)
    pair_token_index: Dict[str, Set[str]] = field(default_factory=dict)
    normalized_rank_groups: Dict[str, Dict[str, object]] = field(default_factory=dict)
    retrieval_cache: Dict[Tuple[str, int, Optional[int], Optional[int]], Dict[str, object]] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    corpus_loaded: bool = False
    rank_loaded: bool = False
    pair_record_count: int = 0

    def cache_key(self, normalized_query: str, top_k: int, user_id: Optional[int], session_id: Optional[int]) -> Tuple[str, int, Optional[int], Optional[int]]:
        return (normalized_query, max(1, top_k), user_id, session_id)


def _tokenize_query(text: str) -> List[str]:
    return [token for token in re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{1,4}", normalize_query(text)) if token]


def _merge_rank_groups(raw_groups: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    normalized_groups: Dict[str, Dict[str, object]] = {}
    for raw_query, group in raw_groups.items():
        normalized_query = normalize_query(raw_query)
        merged = normalized_groups.setdefault(
            normalized_query,
            {
                "query": raw_query,
                "items": {},
                "best_record": None,
                "best_record_key": None,
            },
        )
        for item_id, item_entry in (group.get("items") or {}).items():
            current = merged["items"].setdefault(
                item_id,
                {
                    "item_id": item_entry.get("item_id"),
                    "purchase_cnt": 0,
                    "click_cnt": 0,
                    "ranking_signal": 0.0,
                },
            )
            current["purchase_cnt"] += safe_int(item_entry.get("purchase_cnt"))
            current["click_cnt"] += safe_int(item_entry.get("click_cnt"))
            current["ranking_signal"] += float(item_entry.get("ranking_signal") or 0.0)
        candidate_key = group.get("best_record_key")
        if merged["best_record"] is None or (candidate_key is not None and candidate_key > merged["best_record_key"]):
            merged["query"] = raw_query
            merged["best_record"] = group.get("best_record")
            merged["best_record_key"] = candidate_key
    return normalized_groups


def build_v5_runtime_context(
    corpus_path: Path,
    rank_path: Path,
    users_path: Optional[Path],
    pairs_path: Optional[Path],
    preload_rank_groups: bool = False,
    preload_corpus_map: bool = False,
) -> V5RuntimeContext:
    context = V5RuntimeContext(
        corpus_path=corpus_path,
        rank_path=rank_path,
        users_path=users_path,
        pairs_path=pairs_path,
    )
    context.users_map = load_user_map(users_path if users_path and users_path.exists() else None)

    if pairs_path and pairs_path.exists():
        for record in iter_jsonl(pairs_path):
            context.pair_record_count += 1
            record_query = first_non_empty(record, ["query"])
            normalized_query = normalize_query(record_query)
            context.pair_query_index.setdefault(normalized_query, []).append(record)
            for token in _tokenize_query(normalized_query):
                context.pair_token_index.setdefault(token, set()).add(normalized_query)
    if preload_rank_groups:
        ensure_rank_groups(context)
    if preload_corpus_map:
        ensure_corpus_map(context)
    return context


def ensure_corpus_map(runtime_context: V5RuntimeContext) -> Dict[int, Dict[str, object]]:
    if not runtime_context.corpus_loaded:
        runtime_context.corpus_map = load_corpus_map(runtime_context.corpus_path)
        runtime_context.corpus_loaded = True
    return runtime_context.corpus_map


def ensure_rank_groups(runtime_context: V5RuntimeContext) -> Dict[str, Dict[str, object]]:
    if not runtime_context.rank_loaded:
        raw_rank_groups = collect_query_groups(
            rank_path=runtime_context.rank_path,
            corpus_map={},
            users_map=runtime_context.users_map,
        )
        runtime_context.normalized_rank_groups = _merge_rank_groups(raw_rank_groups)
        runtime_context.rank_loaded = True
    return runtime_context.normalized_rank_groups


def _load_pair_hit_from_context(
    runtime_context: V5RuntimeContext,
    normalized_query: str,
    top_k: int,
    user_id: Optional[int],
    session_id: Optional[int],
) -> Optional[Dict[str, object]]:
    records = runtime_context.pair_query_index.get(normalized_query) or []
    if not records:
        return None

    user_match = None
    session_match = None
    query_match = None
    for record in records:
        evidence_items = list(record.get("evidence_items") or [])[:top_k]
        target_item = record.get("target_item") or (evidence_items[0] if evidence_items else {})
        payload = _build_result_payload(
            query=first_non_empty(record, ["query"]),
            source="train_pairs_query_exact",
            evidence_items=evidence_items,
            user_profile=record.get("user_profile") or {},
            recent_behavior_titles=record.get("recent_behavior_titles") or [],
            target_item=target_item,
        )
        record_user_id = safe_int(record.get("user_id"))
        record_session_id = safe_int(record.get("session_id"))
        if user_id is not None and record_user_id == user_id and user_match is None:
            user_match = {**payload, "source": "train_pairs_user_exact"}
        if session_id is not None and record_session_id == session_id and session_match is None:
            session_match = {**payload, "source": "train_pairs_session_exact"}
        if query_match is None:
            query_match = payload

    return user_match or session_match or query_match


def _load_fuzzy_pair_hit_from_context(
    runtime_context: V5RuntimeContext,
    normalized_query: str,
    top_k: int,
) -> Optional[Dict[str, object]]:
    query_tokens = _tokenize_query(normalized_query)
    if not query_tokens:
        return None

    candidate_queries: Set[str] = set()
    for token in query_tokens:
        candidate_queries.update(runtime_context.pair_token_index.get(token, set()))
    if not candidate_queries:
        return None

    scored_candidates: List[Tuple[int, int, str]] = []
    query_token_set = set(query_tokens)
    for candidate_query in candidate_queries:
        candidate_tokens = set(_tokenize_query(candidate_query))
        overlap = len(query_token_set & candidate_tokens)
        if overlap <= 0:
            continue
        length_gap = abs(len(candidate_query) - len(normalized_query))
        scored_candidates.append((overlap, -length_gap, candidate_query))

    if not scored_candidates:
        return None

    scored_candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    best_query = scored_candidates[0][2]
    records = runtime_context.pair_query_index.get(best_query) or []
    if not records:
        return None
    record = records[0]
    evidence_items = list(record.get("evidence_items") or [])[:top_k]
    target_item = record.get("target_item") or (evidence_items[0] if evidence_items else {})
    return _build_result_payload(
        query=first_non_empty(record, ["query"]),
        source="train_pairs_query_fuzzy",
        evidence_items=evidence_items,
        user_profile=record.get("user_profile") or {},
        recent_behavior_titles=record.get("recent_behavior_titles") or [],
        target_item=target_item,
    )


def retrieve_evidence_bundle_cached(
    query: str,
    runtime_context: V5RuntimeContext,
    top_k: int = 3,
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
) -> Dict[str, object]:
    def _attach_runtime_meta(bundle: Dict[str, object], cache_status: str) -> Dict[str, object]:
        enriched = dict(bundle)
        enriched["runtime_cache"] = {
            "cache_status": cache_status,
            "cache_key": list(cache_key),
            "cache_hits": runtime_context.cache_hits,
            "cache_misses": runtime_context.cache_misses,
            "retrieval_cache_size": len(runtime_context.retrieval_cache),
            "pair_query_count": len(runtime_context.pair_query_index),
            "pair_record_count": runtime_context.pair_record_count,
            "pair_token_count": len(runtime_context.pair_token_index),
            "users_count": len(runtime_context.users_map),
            "rank_loaded": runtime_context.rank_loaded,
            "corpus_loaded": runtime_context.corpus_loaded,
        }
        return enriched

    raw_query = str(query).strip()
    normalized_query = normalize_query(raw_query)
    cache_key = runtime_context.cache_key(normalized_query, top_k, user_id, session_id)
    if cache_key in runtime_context.retrieval_cache:
        runtime_context.cache_hits += 1
        return _attach_runtime_meta(runtime_context.retrieval_cache[cache_key], cache_status="hit")

    runtime_context.cache_misses += 1

    result = _load_pair_hit_from_context(
        runtime_context=runtime_context,
        normalized_query=normalized_query,
        top_k=top_k,
        user_id=user_id,
        session_id=session_id,
    )
    bm25_status = ""
    if result is None:
        result = _load_fuzzy_pair_hit_from_context(
            runtime_context=runtime_context,
            normalized_query=normalized_query,
            top_k=top_k,
        )
    if result is None:
        rank_group = ensure_rank_groups(runtime_context).get(normalized_query)
        if rank_group and rank_group.get("items") and rank_group.get("best_record") is not None:
            result = build_query_record(
                query=str(rank_group.get("query") or normalized_query),
                group=rank_group,
                corpus_map=ensure_corpus_map(runtime_context),
                users_map=runtime_context.users_map,
                top_k=max(1, top_k),
            )
            result["source"] = "rank_query_exact"

    if result is None:
        result = lexical_overlap_hit(query=normalized_query, corpus_map=ensure_corpus_map(runtime_context), top_k=max(1, top_k))

    if result is None:
        bundle = retrieve_evidence_bundle(
            query=query,
            corpus_path=runtime_context.corpus_path,
            rank_path=runtime_context.rank_path,
            users_path=runtime_context.users_path,
            pairs_path=runtime_context.pairs_path,
            top_k=top_k,
            user_id=user_id,
            session_id=session_id,
        )
    else:
        bundle = {
            "raw_query": raw_query,
            "normalized_query": normalized_query,
            "result": result,
            "users_map": runtime_context.users_map if user_id is not None else None,
            "bm25_status": bm25_status,
        }
    runtime_context.retrieval_cache[cache_key] = dict(bundle)
    return _attach_runtime_meta(bundle, cache_status="miss")


def warm_retrieval_cache(
    queries: List[str],
    runtime_context: V5RuntimeContext,
    top_k: int = 3,
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
    allow_expensive_fallback: bool = False,
) -> Dict[str, int]:
    warmed = 0
    skipped = 0
    for query in queries:
        normalized_query = normalize_query(query)
        cache_key = runtime_context.cache_key(normalized_query, top_k, user_id, session_id)
        if cache_key in runtime_context.retrieval_cache:
            continue

        result = _load_pair_hit_from_context(
            runtime_context=runtime_context,
            normalized_query=normalized_query,
            top_k=top_k,
            user_id=user_id,
            session_id=session_id,
        )
        if result is None:
            result = _load_fuzzy_pair_hit_from_context(
                runtime_context=runtime_context,
                normalized_query=normalized_query,
                top_k=top_k,
            )

        if result is None and allow_expensive_fallback:
            retrieve_evidence_bundle_cached(
                query=query,
                runtime_context=runtime_context,
                top_k=top_k,
                user_id=user_id,
                session_id=session_id,
            )
            warmed += 1
            continue

        if result is None:
            skipped += 1
            continue

        runtime_context.retrieval_cache[cache_key] = {
            "raw_query": str(query).strip(),
            "normalized_query": normalized_query,
            "result": result,
            "users_map": runtime_context.users_map if user_id is not None else None,
            "bm25_status": "",
        }
        warmed += 1
    return {
        "warmed": warmed,
        "skipped": skipped,
        "retrieval_cache_size": len(runtime_context.retrieval_cache),
    }
