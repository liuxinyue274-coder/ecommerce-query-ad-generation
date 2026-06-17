"""Run supplemental baselines on the same 120-query batch used by the deepseek eval set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional


CURRENT_DIR = Path(__file__).resolve().parent
AD_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AD_DIR.parent

if str(AD_DIR) not in sys.path:
    sys.path.insert(0, str(AD_DIR))

from infer import build_generation_output, normalize_query
from v5_dynamic_creative import v5_llm_dynamic_creative
from v5_runtime import build_v5_runtime_context, warm_retrieval_cache


TOP_K = 3
CANDIDATE_COUNT = 5
BASELINES = [
    "baseline_template_top1",
    "baseline_template_personalized",
    "baseline_summary_topk",
    "baseline_llm_topk_local_fake",
    "baseline_v5_local_fake",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build baseline results for the same query batch used by deepseek eval.")
    parser.add_argument(
        "--source_eval_path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "demo_v5_eval_min120_deepseek.jsonl",
        help="Path to the already-built deepseek eval JSONL. Its sample order defines the batch.",
    )
    parser.add_argument(
        "--pairs_path",
        type=Path,
        default=AD_DIR / "data" / "train_pairs.jsonl",
        help="Path to train_pairs JSONL.",
    )
    parser.add_argument(
        "--corpus_path",
        type=Path,
        default=PROJECT_ROOT / "items_lite" / "train.jsonl",
        help="Path to items_lite/train.jsonl.",
    )
    parser.add_argument(
        "--rank_path",
        type=Path,
        default=PROJECT_ROOT / "rank_lite" / "train.jsonl",
        help="Path to rank_lite/train.jsonl.",
    )
    parser.add_argument(
        "--users_path",
        type=Path,
        default=PROJECT_ROOT / "users_lite" / "train.jsonl",
        help="Path to users_lite/train.jsonl.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "demo_v5_eval_min120_baselines.jsonl",
        help="Path to write the supplemental baseline JSONL.",
    )
    parser.add_argument(
        "--stats_path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "demo_v5_eval_min120_baselines_stats.json",
        help="Path to write the supplemental baseline stats JSON.",
    )
    parser.add_argument(
        "--notes_path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "demo_v5_eval_min120_baselines_notes.md",
        help="Path to write the supplemental baseline notes Markdown.",
    )
    parser.add_argument(
        "--mock_llm_config_path",
        type=Path,
        default=AD_DIR / "llm_config.mock.json",
        help="Config path used for the legacy llm baseline with mock/local provider.",
    )
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Top-k evidence items for summary/V5 baselines.")
    parser.add_argument("--candidate_count", type=int, default=CANDIDATE_COUNT, help="Candidate count for V5 local_fake.")
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_source_batch(path: Path) -> List[Dict[str, object]]:
    records = list(iter_jsonl(path))
    if not records:
        raise RuntimeError(f"Source eval file is empty: {path}")
    return records


def build_train_pair_index(path: Path) -> Dict[str, Dict[str, object]]:
    index: Dict[str, Dict[str, object]] = {}
    for record in iter_jsonl(path):
        normalized = normalize_query(str(record.get("query") or ""))
        if normalized and normalized not in index:
            index[normalized] = record
    return index


def build_result_payload(case: Dict[str, object], top_k: int, source: str) -> Dict[str, object]:
    evidence_items = list(case.get("evidence_items") or [])[:top_k]
    target_item = case.get("target_item") or (evidence_items[0] if evidence_items else {})
    return {
        "query": case.get("query", ""),
        "user_profile": case.get("user_profile") or {},
        "recent_behavior_titles": case.get("recent_behavior_titles") or [],
        "evidence_items": evidence_items,
        "target_item": target_item,
        "source": source,
    }


def personalized_source(case: Dict[str, object]) -> str:
    if case.get("user_id") is not None:
        return "train_pairs_user_exact"
    if case.get("session_id") is not None:
        return "train_pairs_session_exact"
    return "train_pairs_query_exact"


def summarize_legacy_output(output: Dict[str, object]) -> Dict[str, object]:
    return {
        "copy_style": output.get("copy_style"),
        "retrieval_source": output.get("retrieval_source"),
        "fallback_triggered": bool(output.get("fallback_triggered")),
        "fallback_reason": output.get("fallback_reason"),
        "llm_provider": output.get("llm_provider"),
        "llm_status": output.get("llm_status"),
        "llm_rewrite_attempted": bool(output.get("llm_rewrite_attempted")),
        "bad_copy_detected": bool(output.get("bad_copy_detected")),
        "bad_copy_reason": output.get("bad_copy_reason"),
        "target_item": output.get("target_item") or {},
        "evidence_items": output.get("evidence_items") or [],
        "user_profile": output.get("user_profile") or {},
        "recent_behavior_titles": output.get("recent_behavior_titles") or [],
        "final_ad_copy": output.get("final_ad_copy") or "",
    }


def summarize_v5_output(result: Dict[str, object]) -> Dict[str, object]:
    return {
        "provider": result.get("provider"),
        "model": result.get("model"),
        "llm_status": result.get("llm_status"),
        "fallback_triggered": bool(result.get("llm_provider", {}).get("fallback_used")),
        "fallback_reason": result.get("llm_provider", {}).get("fallback_reason"),
        "cache_status": result.get("runtime_cache", {}).get("cache_status"),
        "intent": result.get("intent_enricher") or {},
        "selected_evidence": result.get("evidence_selector", {}).get("selected_evidence_items", []) or [],
        "ranked_candidates": result.get("copy_ranker", {}).get("ranked_candidates", []) or [],
        "best_copy": (result.get("copy_ranker", {}).get("top_candidate", {}) or {}).get("text", ""),
        "final_ad_copy": (result.get("final_ad_copy", {}) or {}).get("final_ad_copy", ""),
    }


def counter_to_dict(counter: Counter) -> Dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter.keys(), key=lambda item: str(item))}


def build_stats(records: List[Dict[str, object]], source_eval_path: Path) -> Dict[str, object]:
    stats: Dict[str, object] = {
        "total_samples": len(records),
        "source_eval_path": str(source_eval_path),
        "baselines_run": list(BASELINES),
        "sft_model_path_present": bool(os.environ.get("SFT_MODEL_PATH")),
        "baseline_summary": {},
    }

    for baseline_name in BASELINES:
        fallback_counter = 0
        fallback_reason_counter: Counter = Counter()
        llm_provider_counter: Counter = Counter()
        llm_status_counter: Counter = Counter()
        provider_counter: Counter = Counter()
        model_counter: Counter = Counter()
        nonempty_final_counter = 0

        for record in records:
            payload = record.get(baseline_name) or {}
            final_copy = str(payload.get("final_ad_copy") or "")
            if final_copy:
                nonempty_final_counter += 1
            if payload.get("fallback_triggered"):
                fallback_counter += 1
            reason = str(payload.get("fallback_reason") or "")
            if reason:
                fallback_reason_counter[reason] += 1
            llm_provider = str(payload.get("llm_provider") or "")
            if llm_provider:
                llm_provider_counter[llm_provider] += 1
            llm_status = str(payload.get("llm_status") or "")
            if llm_status:
                llm_status_counter[llm_status] += 1
            provider = str(payload.get("provider") or "")
            if provider:
                provider_counter[provider] += 1
            model = str(payload.get("model") or "")
            if model:
                model_counter[model] += 1

        stats["baseline_summary"][baseline_name] = {
            "count": len(records),
            "nonempty_final_ad_copy_count": nonempty_final_counter,
            "fallback_count": fallback_counter,
            "fallback_ratio": round(fallback_counter / len(records), 4) if records else 0.0,
            "fallback_reason_distribution": counter_to_dict(fallback_reason_counter),
            "llm_provider_distribution": counter_to_dict(llm_provider_counter),
            "llm_status_distribution": counter_to_dict(llm_status_counter),
            "provider_distribution": counter_to_dict(provider_counter),
            "model_distribution": counter_to_dict(model_counter),
        }

    stats["baseline_summary"]["baseline_v5_sft_local"] = {
        "count": 0,
        "skipped": True,
        "reason": "SFT_MODEL_PATH missing; local SFT baseline not executed in this session.",
    }
    return stats


def build_notes(
    source_eval_path: Path,
    output_path: Path,
    stats_path: Path,
    notes_path: Path,
    mock_llm_config_path: Path,
    stats: Dict[str, object],
) -> str:
    summary = stats["baseline_summary"]
    lines = [
        f"# {output_path.stem} 说明",
        "",
        "## 1. 文件对应关系",
        "",
        f"- deepseek 参考批次：`{source_eval_path}`",
        f"- baseline 主文件：`{output_path}`",
        f"- baseline 统计文件：`{stats_path}`",
        f"- baseline 说明文件：`{notes_path}`",
        "",
        "## 2. 运行目标",
        "",
        "- 使用 `demo_v5_eval_min120_deepseek.jsonl` 中已经固定好的同一批 120 条 query。",
        "- 在不改变 query 批次和样本顺序的前提下，回放其他 baseline，补充对照结果。",
        "- 本文件对应的 baseline 包括：",
        "  - `baseline_template_top1`",
        "  - `baseline_template_personalized`",
        "  - `baseline_summary_topk`",
        "  - `baseline_llm_topk_local_fake`",
        "  - `baseline_v5_local_fake`",
        "",
        "## 3. baseline 口径",
        "",
        "- `baseline_template_top1`：旧版 `template` 模式，`top_k=1`。",
        "- `baseline_template_personalized`：旧版 `template` 模式，优先使用 train_pairs 中同 query 的 `user_id/session_id`。",
        "- `baseline_summary_topk`：旧版 `summary` 模式，`top_k=3`。",
        f"- `baseline_llm_topk_local_fake`：旧版 `llm` 模式，配置文件为 `{mock_llm_config_path}`，实际 provider 是 mock/local。",
        "- `baseline_v5_local_fake`：V5 全链路，provider 固定为 `local_fake`。",
        "- `baseline_v5_sft_local`：当前未执行，因为 `SFT_MODEL_PATH` 不可见。",
        "",
        "## 4. 本次运行摘要",
        "",
        f"- 样本数：`{stats['total_samples']}`",
        f"- `SFT_MODEL_PATH` 当前进程可见：`{stats['sft_model_path_present']}`",
        "",
        "| baseline | fallback_ratio | 备注 |",
        "| --- | ---: | --- |",
    ]

    for baseline_name in BASELINES:
        item = summary[baseline_name]
        note = ""
        if item["llm_provider_distribution"]:
            note = f"llm_provider={item['llm_provider_distribution']}"
        elif item["provider_distribution"]:
            note = f"provider={item['provider_distribution']}"
        lines.append(f"| {baseline_name} | {item['fallback_ratio']} | {note} |")

    lines.extend(
        [
            "",
            "## 5. 复现命令",
            "",
            "```powershell",
            "$env:PYTHONDONTWRITEBYTECODE='1'",
            (
                f"python -B -X utf8 {AD_DIR / 'eval_v5' / 'build_eval_baselines_from_deepseek.py'} "
                f"--source_eval_path {source_eval_path} "
                f"--output_path {output_path} "
                f"--stats_path {stats_path} "
                f"--notes_path {notes_path}"
            ),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_jsonl(records: Iterable[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    ensure_exists(args.source_eval_path, "source_eval_path")
    ensure_exists(args.pairs_path, "pairs_path")
    ensure_exists(args.corpus_path, "corpus_path")
    ensure_exists(args.rank_path, "rank_path")
    ensure_exists(args.users_path, "users_path")
    ensure_exists(args.mock_llm_config_path, "mock_llm_config_path")

    source_records = load_source_batch(args.source_eval_path)
    pair_index = build_train_pair_index(args.pairs_path)

    runtime_context = build_v5_runtime_context(
        corpus_path=args.corpus_path,
        rank_path=args.rank_path,
        users_path=args.users_path,
        pairs_path=args.pairs_path,
    )
    warmup = warm_retrieval_cache(
        queries=[str(item.get("query") or "") for item in source_records],
        runtime_context=runtime_context,
        top_k=args.top_k,
        allow_expensive_fallback=False,
    )
    print(json.dumps({"warm_cache": warmup, "total_samples": len(source_records)}, ensure_ascii=False, indent=2))

    records: List[Dict[str, object]] = []
    total = len(source_records)
    for index, source in enumerate(source_records, start=1):
        query = str(source.get("query") or "")
        normalized_query = normalize_query(query)
        case = pair_index.get(normalized_query)
        if case is None:
            raise RuntimeError(f"Cannot find train_pairs case for normalized_query={normalized_query!r}")

        user_id = case.get("user_id")
        personalized = personalized_source(case)

        template_top1 = summarize_legacy_output(
            build_generation_output(
                raw_query=query,
                result=build_result_payload(case=case, top_k=1, source="train_pairs_query_exact"),
                copy_style="template",
            )
        )
        template_personalized = summarize_legacy_output(
            build_generation_output(
                raw_query=query,
                result=build_result_payload(case=case, top_k=1, source=personalized),
                copy_style="template",
                user_id=int(user_id) if user_id is not None else None,
            )
        )
        summary_topk = summarize_legacy_output(
            build_generation_output(
                raw_query=query,
                result=build_result_payload(case=case, top_k=args.top_k, source=personalized),
                copy_style="summary",
                user_id=int(user_id) if user_id is not None else None,
            )
        )
        llm_topk_local_fake = summarize_legacy_output(
            build_generation_output(
                raw_query=query,
                result=build_result_payload(case=case, top_k=args.top_k, source=personalized),
                copy_style="llm",
                user_id=int(user_id) if user_id is not None else None,
                llm_config_path=args.mock_llm_config_path,
            )
        )
        v5_local_fake = summarize_v5_output(
            v5_llm_dynamic_creative(
                query=query,
                corpus_path=args.corpus_path,
                rank_path=args.rank_path,
                users_path=args.users_path,
                pairs_path=args.pairs_path,
                llm_config_path=args.mock_llm_config_path,
                top_k=args.top_k,
                candidate_count=args.candidate_count,
                requested_tone="creative",
                runtime_context=runtime_context,
                provider_override="local_fake",
            )
        )

        record = {
            "sample_id": source.get("sample_id"),
            "query": query,
            "normalized_query": source.get("normalized_query") or normalized_query,
            "query_bucket": source.get("query_bucket"),
            "query_len": source.get("query_len"),
            "query_category": source.get("query_category"),
            "evidence_score": source.get("evidence_score"),
            "evidence_strength": source.get("evidence_strength"),
            "reference_deepseek": {
                "provider": source.get("provider"),
                "model": source.get("model"),
                "llm_status": source.get("llm_status"),
                "final_ad_copy": source.get("final_ad_copy"),
            },
            "source_case": {
                "user_id": case.get("user_id"),
                "session_id": case.get("session_id"),
                "retrieval_source": personalized,
            },
            "baseline_template_top1": template_top1,
            "baseline_template_personalized": template_personalized,
            "baseline_summary_topk": summary_topk,
            "baseline_llm_topk_local_fake": llm_topk_local_fake,
            "baseline_v5_local_fake": v5_local_fake,
        }
        records.append(record)

        if index == 1 or index % 10 == 0 or index == total:
            print(f"[{index}/{total}] {query} -> template={template_top1['final_ad_copy']} | v5_local_fake={v5_local_fake['final_ad_copy']}")

    stats = build_stats(records=records, source_eval_path=args.source_eval_path)
    write_jsonl(records, args.output_path)
    args.stats_path.parent.mkdir(parents=True, exist_ok=True)
    with args.stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    args.notes_path.parent.mkdir(parents=True, exist_ok=True)
    with args.notes_path.open("w", encoding="utf-8") as handle:
        handle.write(
            build_notes(
                source_eval_path=args.source_eval_path,
                output_path=args.output_path,
                stats_path=args.stats_path,
                notes_path=args.notes_path,
                mock_llm_config_path=args.mock_llm_config_path,
                stats=stats,
            )
        )

    print("")
    print("baseline build complete")
    print(f"output_path: {args.output_path}")
    print(f"stats_path: {args.stats_path}")
    print(f"notes_path: {args.notes_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
