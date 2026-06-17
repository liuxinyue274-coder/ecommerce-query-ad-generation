"""Demo runner for the V5 LLM dynamic creative pipeline.

Run from project root:
python ad_generation/demo_v5_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from v5_dynamic_creative import v5_llm_dynamic_creative
from v5_runtime import build_v5_runtime_context, warm_retrieval_cache


DEMO_QUERIES = [
    "运动手环",
    "宿舍吹风机",
    "通勤双肩包",
    "敏感肌面霜",
    "手机壳",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the V5 ad-generation demo with runtime cache.")
    parser.add_argument("--debug", action="store_true", help="Print full prompt and retrieval details.")
    parser.add_argument("--limit", type=int, default=5, help="Number of demo queries to run.")
    parser.add_argument("--provider", choices=["local_fake", "deepseek_chat", "sft_local"], default="local_fake", help="LLM provider for the V5 pipeline.")
    parser.add_argument("--preload_rank", action="store_true", help="Eagerly preload rank_lite into runtime context.")
    parser.add_argument("--preload_corpus", action="store_true", help="Eagerly preload items_lite into runtime context.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("outputs/demo_v5_outputs.jsonl"),
        help="Path to save full V5 outputs as JSONL.",
    )
    return parser.parse_args()


def _decode_raw_llm_output(raw_generations: List[Dict[str, object]]) -> List[Dict[str, object]]:
    decoded: List[Dict[str, object]] = []
    for item in raw_generations:
        raw_text = str(item.get("raw_text") or "")
        try:
            payload = json.loads(raw_text) if raw_text else {}
        except json.JSONDecodeError:
            payload = {"raw_text": raw_text}
        decoded.append(
            {
                "candidate_id": item.get("candidate_id"),
                "copies": payload.get("copies", []),
            }
        )
    return decoded


def _simplify_parsed_candidates(parsed_candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        {
            "candidate_id": item.get("candidate_id"),
            "strategy": item.get("strategy"),
            "copy": item.get("text"),
            "used_evidence": item.get("used_evidence", []),
        }
        for item in parsed_candidates
    ]


def _simplify_ranked_candidates(ranked_candidates: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        {
            "candidate_id": item.get("candidate_id"),
            "strategy": item.get("strategy"),
            "copy": item.get("text"),
            "rank_score": item.get("rank_score"),
        }
        for item in ranked_candidates[:3]
    ]


def _print_core_result(query: str, result: Dict[str, object], debug: bool) -> None:
    print(f"########## demo query: {query} ##########")
    runtime_cache = dict(result.get("runtime_cache") or {})
    retrieval_bundle = dict(result.get("retrieval_bundle") or {})
    retrieval_result = dict(retrieval_bundle.get("result") or {})
    print("=== llm_provider ===")
    print(
        json.dumps(
            {
                "provider": result.get("provider"),
                "model": result.get("model"),
                "llm_status": result.get("llm_status"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("")
    print("=== retrieval ===")
    print(
        json.dumps(
            {
                "source": retrieval_result.get("source"),
                "cache_status": runtime_cache.get("cache_status"),
                "rank_loaded": runtime_cache.get("rank_loaded"),
                "corpus_loaded": runtime_cache.get("corpus_loaded"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("")
    print("=== raw_llm_output ===")
    print(
        json.dumps(
            _decode_raw_llm_output(list(result.get("llm_provider", {}).get("raw_generations", []))),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("")
    print("=== parsed_candidates ===")
    print(
        json.dumps(
            _simplify_parsed_candidates(list(result.get("llm_output_parser", {}).get("parsed_candidates", []))),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("")
    print("=== ranked_candidates ===")
    print(
        json.dumps(
            _simplify_ranked_candidates(list(result.get("copy_ranker", {}).get("ranked_candidates", []))),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("")
    print("=== final_ad_copy ===")
    print(json.dumps(result.get("final_ad_copy", {}).get("final_ad_copy", ""), ensure_ascii=False, indent=2))
    if debug:
        print("")
        print("=== debug_retrieval_bundle ===")
        print(json.dumps(retrieval_bundle, ensure_ascii=False, indent=2))
        print("")
        print("=== debug_prompt_builder ===")
        print(json.dumps(result.get("prompt_builder", {}), ensure_ascii=False, indent=2))
    print("")


def main() -> None:
    args = parse_args()
    project_root = CURRENT_DIR.parent
    corpus_path = project_root / "items_lite" / "train.jsonl"
    rank_path = project_root / "rank_lite" / "train.jsonl"
    users_path = project_root / "users_lite" / "train.jsonl"
    pairs_path = CURRENT_DIR / "data" / "train_pairs.jsonl"
    llm_config_path = CURRENT_DIR / ("llm_config.deepseek_chat.json" if args.provider == "deepseek_chat" else "llm_config.mock.json")

    print("loading runtime cache...")
    runtime_context = build_v5_runtime_context(
        corpus_path=corpus_path,
        rank_path=rank_path,
        users_path=users_path,
        pairs_path=pairs_path,
        preload_rank_groups=args.preload_rank,
        preload_corpus_map=args.preload_corpus,
    )
    selected_queries = DEMO_QUERIES[: max(1, args.limit)]
    warmup_summary = warm_retrieval_cache(
        queries=selected_queries,
        runtime_context=runtime_context,
        top_k=3,
        allow_expensive_fallback=False,
    )
    print(
        f"runtime cache ready: pair_queries={len(runtime_context.pair_query_index)} "
        f"pair_records={runtime_context.pair_record_count} "
        f"pair_tokens={len(runtime_context.pair_token_index)} "
        f"users={len(runtime_context.users_map)} "
        f"warmed={warmup_summary['warmed']} skipped={warmup_summary['skipped']}"
    )
    print("")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = []

    for query in selected_queries:
        result = v5_llm_dynamic_creative(
            query=query,
            corpus_path=corpus_path,
            rank_path=rank_path,
            users_path=users_path,
            pairs_path=pairs_path,
            llm_config_path=llm_config_path,
            top_k=3,
            candidate_count=5,
            requested_tone="creative",
            runtime_context=runtime_context,
            provider_override=args.provider,
        )
        results.append(result)
        _print_core_result(query=query, result=result, debug=args.debug)

    with args.output_path.open("w", encoding="utf-8") as handle:
        for query, result in zip(selected_queries, results):
            handle.write(json.dumps({"query": query, "result": result}, ensure_ascii=False) + "\n")

    print("=== final_ad_copy_summary ===")
    for query, result in zip(selected_queries, results):
        final_copy = str(result.get("final_ad_copy", {}).get("final_ad_copy") or "")
        print(f"{query}: {final_copy}")
    print("")
    print("=== runtime_cache_stats ===")
    print(
        json.dumps(
            {
                "cache_hits": runtime_context.cache_hits,
                "cache_misses": runtime_context.cache_misses,
                "retrieval_cache_size": len(runtime_context.retrieval_cache),
                "pair_query_count": len(runtime_context.pair_query_index),
                "pair_record_count": runtime_context.pair_record_count,
                "pair_token_count": len(runtime_context.pair_token_index),
                "users_count": len(runtime_context.users_map),
                "rank_loaded": runtime_context.rank_loaded,
                "corpus_loaded": runtime_context.corpus_loaded,
                "output_path": str(args.output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
