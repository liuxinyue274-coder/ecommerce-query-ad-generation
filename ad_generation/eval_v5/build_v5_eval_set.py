"""Build a stratified V5 evaluation set without modifying existing project code.

Usage example:

python -X utf8 ad_generation/eval_v5/build_v5_eval_set.py ^
  --provider local_fake ^
  --output_path outputs/demo_v5_eval_min120.jsonl ^
  --stats_path outputs/demo_v5_eval_min120_stats.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
AD_DIR = CURRENT_DIR.parent
PROJECT_ROOT = AD_DIR.parent

if str(AD_DIR) not in sys.path:
    sys.path.insert(0, str(AD_DIR))

from infer import normalize_query
from v5_dynamic_creative import v5_llm_dynamic_creative
from v5_runtime import build_v5_runtime_context, warm_retrieval_cache


TOP_K = 3
CANDIDATE_COUNT = 5
PLACEHOLDER_BRANDS = {"", "无品牌", "其他", "other", "unknown", "未知", "nan", "none", "null"}
QUERY_BUCKET_ORDER = ["short", "medium", "long"]
EVIDENCE_STRENGTH_ORDER = ["high", "low"]
QUERY_CATEGORY_ORDER = ["digital", "apparel", "food", "health", "misc"]
DEFAULT_QUOTAS: Dict[Tuple[str, str], int] = {
    ("short", "high"): 20,
    ("short", "low"): 20,
    ("medium", "high"): 30,
    ("medium", "low"): 30,
    ("long", "high"): 10,
    ("long", "low"): 10,
}
QUERY_CATEGORY_KEYWORDS = {
    "digital": [
        "手机",
        "数码",
        "电脑",
        "办公设备",
        "办公用品",
        "家用电器",
        "电器",
        "耳机",
        "相机",
        "键盘",
        "显示器",
        "充电",
        "手环",
        "手表",
        "平板",
        "路由器",
        "硬盘",
    ],
    "apparel": [
        "女装",
        "男装",
        "童装",
        "内衣",
        "女鞋",
        "男鞋",
        "箱包",
        "饰品",
        "配饰",
        "套装",
        "牛仔裤",
        "裤",
        "裙",
        "上衣",
        "外套",
        "毛衣",
        "衬衫",
        "t恤",
    ],
    "food": [
        "食品",
        "零食",
        "生鲜",
        "饮料",
        "酒水",
        "粮油",
        "坚果",
        "特产",
        "饼干",
        "奶茶",
        "牛奶",
        "咖啡",
        "茶",
        "面包",
        "大米",
        "速食",
    ],
    "health": [
        "营养健康",
        "医疗保健",
        "保健",
        "氨糖",
        "钙片",
        "维生素",
        "美容护肤",
        "美妆",
        "个护清洁",
        "身体护理",
        "口腔护理",
        "面霜",
        "精华",
        "防晒",
        "洗发",
        "护发",
        "洁面",
    ],
}


@dataclass
class PlannedSample:
    query: str
    normalized_query: str
    query_bucket: str
    query_len: int
    evidence_score: float
    evidence_strength: str
    query_category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the V5 stratified evaluation set.")
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
        "--provider",
        choices=["local_fake", "deepseek_chat", "sft_local"],
        default="local_fake",
        help="Provider used for the V5 generation pass.",
    )
    parser.add_argument(
        "--llm_config_path",
        type=Path,
        default=None,
        help="Optional explicit llm_config path.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "demo_v5_eval_min120.jsonl",
        help="Path to write the evaluation JSONL.",
    )
    parser.add_argument(
        "--stats_path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "demo_v5_eval_min120_stats.json",
        help="Path to write the evaluation stats JSON.",
    )
    parser.add_argument(
        "--notes_path",
        type=Path,
        default=None,
        help="Optional path to write the sampling notes Markdown. Defaults to <output_stem>_sampling_notes.md.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Top-k evidence items used in V5.")
    parser.add_argument("--candidate_count", type=int, default=CANDIDATE_COUNT, help="Candidate count used in V5.")
    return parser.parse_args()


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def resolve_llm_config(provider: str, llm_config_path: Path | None) -> Path:
    if llm_config_path is not None:
        return llm_config_path
    if provider == "deepseek_chat":
        return AD_DIR / "llm_config.deepseek_chat.json"
    return AD_DIR / "llm_config.mock.json"


def query_bucket(normalized_query: str) -> str:
    query_len = len(normalized_query)
    if query_len <= 6:
        return "short"
    if query_len <= 15:
        return "medium"
    return "long"


def evidence_field_hits(item: Dict[str, object]) -> int:
    brand_name = str(item.get("brand_name") or "").strip().lower()
    return sum(
        [
            1 if str(item.get("item_title") or item.get("title") or "").strip() else 0,
            1 if str(item.get("category_path") or "").strip() else 0,
            1 if brand_name and brand_name not in PLACEHOLDER_BRANDS else 0,
            1 if str(item.get("seller_name") or "").strip() else 0,
            1 if item.get("ranking_signal") not in (None, "") else 0,
            1 if item.get("relevance_signal") not in (None, "") else 0,
        ]
    )


def compute_evidence_score(selected_evidence: List[Dict[str, object]], top_k: int) -> float:
    total_hits = sum(evidence_field_hits(item) for item in selected_evidence[:top_k])
    denominator = max(1, top_k) * 6
    return total_hits / denominator


def evidence_strength(score: float) -> str:
    return "high" if score >= 0.75 else "low"


def infer_query_category(query: str, selected_evidence: List[Dict[str, object]]) -> str:
    parts = [query]
    for item in selected_evidence[:TOP_K]:
        parts.append(str(item.get("category_path") or ""))
        parts.append(str(item.get("item_title") or item.get("title") or ""))
    haystack = " ".join(parts).lower()
    for category in ["health", "food", "apparel", "digital"]:
        if any(keyword.lower() in haystack for keyword in QUERY_CATEGORY_KEYWORDS[category]):
            return category
    return "misc"


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def build_candidate_pool(pairs_path: Path, top_k: int) -> Dict[Tuple[str, str], List[PlannedSample]]:
    pool: Dict[Tuple[str, str], List[PlannedSample]] = defaultdict(list)
    seen_normalized_queries: set[str] = set()

    for record in iter_jsonl(pairs_path):
        raw_query = str(record.get("query") or "").strip()
        normalized = normalize_query(raw_query)
        if not normalized or normalized in seen_normalized_queries:
            continue

        evidence_items = list(record.get("evidence_items") or [])[:top_k]
        if not evidence_items:
            continue

        score = compute_evidence_score(evidence_items, top_k=top_k)
        bucket = query_bucket(normalized)
        strength = evidence_strength(score)
        category = infer_query_category(raw_query, evidence_items)
        pool[(bucket, strength)].append(
            PlannedSample(
                query=raw_query,
                normalized_query=normalized,
                query_bucket=bucket,
                query_len=len(normalized),
                evidence_score=score,
                evidence_strength=strength,
                query_category=category,
            )
        )
        seen_normalized_queries.add(normalized)
    return pool


def select_stratified_samples(
    pool: Dict[Tuple[str, str], List[PlannedSample]],
    quotas: Dict[Tuple[str, str], int],
    seed: int,
) -> List[PlannedSample]:
    rng = random.Random(seed)
    selected: List[PlannedSample] = []

    for bucket in QUERY_BUCKET_ORDER:
        for strength in EVIDENCE_STRENGTH_ORDER:
            key = (bucket, strength)
            candidates = list(pool.get(key) or [])
            if len(candidates) < quotas[key]:
                raise RuntimeError(
                    f"Not enough candidates for bucket={bucket}, evidence_strength={strength}: "
                    f"required={quotas[key]}, available={len(candidates)}"
                )
            rng.shuffle(candidates)
            chosen = sorted(candidates[: quotas[key]], key=lambda item: (item.query_category, item.normalized_query))
            selected.extend(chosen)
    return selected


def normalize_intent(intent_bundle: Dict[str, object]) -> Dict[str, object]:
    return {
        "intent_summary": str(intent_bundle.get("intent_summary") or ""),
        "scenario": str(intent_bundle.get("scenario") or ""),
        "audience": str(intent_bundle.get("audience") or ""),
        "purchase_focus": str(intent_bundle.get("purchase_focus") or ""),
        "attribute_hints": list(intent_bundle.get("attribute_hints") or []),
        "search_focus": list(intent_bundle.get("search_focus") or []),
    }


def extract_rewritten_candidates(result: Dict[str, object]) -> List[Dict[str, object]]:
    rewriter_bundle = dict(result.get("copy_rewriter") or {})
    final_candidate = dict(rewriter_bundle.get("final_candidate") or {})
    if not rewriter_bundle.get("rewritten"):
        return []
    if not final_candidate:
        return []
    return [
        {
            "candidate_id": final_candidate.get("candidate_id"),
            "strategy": final_candidate.get("strategy"),
            "text": final_candidate.get("text"),
            "rewrite_reason": rewriter_bundle.get("rewrite_reason"),
            "source": final_candidate.get("source"),
        }
    ]


def build_sample_record(
    sample_id: str,
    planned_sample: PlannedSample,
    result: Dict[str, object],
    top_k: int,
) -> Dict[str, object]:
    selected_evidence = list(result.get("evidence_selector", {}).get("selected_evidence_items", []) or [])
    evidence_score = compute_evidence_score(selected_evidence, top_k=top_k)
    evidence_strength_value = evidence_strength(evidence_score)
    category = infer_query_category(planned_sample.normalized_query, selected_evidence)
    anchor_item = dict(result.get("evidence_selector", {}).get("anchor_item") or {})
    top_candidate = dict(result.get("copy_ranker", {}).get("top_candidate") or {})
    final_copy_bundle = dict(result.get("final_ad_copy") or {})

    return {
        "sample_id": sample_id,
        "query": planned_sample.query,
        "normalized_query": str(result.get("query_normalization", {}).get("normalized_query") or planned_sample.normalized_query),
        "query_bucket": str(result.get("query_normalization", {}).get("normalized_query") and query_bucket(str(result.get("query_normalization", {}).get("normalized_query"))) or planned_sample.query_bucket),
        "query_len": int(result.get("query_normalization", {}).get("query_length") or planned_sample.query_len),
        "query_category": category,
        "intent": normalize_intent(dict(result.get("intent_enricher") or {})),
        "selected_evidence": selected_evidence,
        "anchor_item_id": anchor_item.get("item_id"),
        "evidence_score": round(evidence_score, 4),
        "evidence_strength": evidence_strength_value,
        "provider": str(result.get("provider") or ""),
        "model": str(result.get("model") or ""),
        "llm_status": str(result.get("llm_status") or ""),
        "raw_llm_output": list(result.get("llm_provider", {}).get("raw_generations", []) or []),
        "parsed_candidates": list(result.get("llm_output_parser", {}).get("parsed_candidates", []) or []),
        "validated_candidates": list(result.get("copy_validator", {}).get("validated_candidates", []) or []),
        "ranked_candidates": list(result.get("copy_ranker", {}).get("ranked_candidates", []) or []),
        "rewritten_candidates": extract_rewritten_candidates(result),
        "best_copy": str(top_candidate.get("text") or ""),
        "final_ad_copy": str(final_copy_bundle.get("final_ad_copy") or ""),
        "fallback_triggered": bool(result.get("llm_provider", {}).get("fallback_used")),
        "fallback_reason": str(result.get("llm_provider", {}).get("fallback_reason") or ""),
        "cache_status": str(result.get("runtime_cache", {}).get("cache_status") or ""),
    }


def write_jsonl(records: Iterable[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def resolve_notes_path(output_path: Path, notes_path: Path | None) -> Path:
    if notes_path is not None:
        return notes_path
    return output_path.with_name(f"{output_path.stem}_sampling_notes.md")


def counter_to_ordered_dict(counter: Counter, ordered_keys: List[str]) -> Dict[str, int]:
    ordered = {key: int(counter.get(key, 0)) for key in ordered_keys if key in counter or key in ordered_keys}
    extras = sorted(key for key in counter.keys() if key not in ordered)
    for key in extras:
        ordered[key] = int(counter[key])
    return ordered


def build_stats(
    records: List[Dict[str, object]],
    quotas: Dict[Tuple[str, str], int],
    provider: str,
    llm_config_path: Path,
) -> Dict[str, object]:
    bucket_counter = Counter(str(record.get("query_bucket") or "") for record in records)
    strength_counter = Counter(str(record.get("evidence_strength") or "") for record in records)
    category_counter = Counter(str(record.get("query_category") or "") for record in records)
    provider_counter = Counter(str(record.get("provider") or "") for record in records)
    model_counter = Counter(str(record.get("model") or "") for record in records)
    llm_status_counter = Counter(str(record.get("llm_status") or "") for record in records)
    cache_status_counter = Counter(str(record.get("cache_status") or "") for record in records)
    quota_counter = Counter(f"{bucket}-{strength}" for bucket, strength in quotas.keys())
    actual_quota_counter = Counter(f"{record.get('query_bucket')}-{record.get('evidence_strength')}" for record in records)
    fallback_count = sum(1 for record in records if record.get("fallback_triggered"))

    return {
        "total_samples": len(records),
        "requested_provider": provider,
        "llm_config_path": str(llm_config_path),
        "deepseek_api_key_present": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "sft_model_path_present": bool(os.environ.get("SFT_MODEL_PATH")),
        "requested_quotas": {f"{bucket}-{strength}": count for (bucket, strength), count in quotas.items()},
        "actual_quota_distribution": {key: int(actual_quota_counter.get(key, 0)) for key in sorted(quota_counter.keys())},
        "query_bucket_distribution": counter_to_ordered_dict(bucket_counter, QUERY_BUCKET_ORDER),
        "evidence_strength_distribution": counter_to_ordered_dict(strength_counter, EVIDENCE_STRENGTH_ORDER),
        "query_category_distribution": counter_to_ordered_dict(category_counter, QUERY_CATEGORY_ORDER),
        "provider_distribution": dict(sorted((key, int(value)) for key, value in provider_counter.items())),
        "model_distribution": dict(sorted((key, int(value)) for key, value in model_counter.items())),
        "llm_status_distribution": dict(sorted((key, int(value)) for key, value in llm_status_counter.items())),
        "cache_status_distribution": dict(sorted((key, int(value)) for key, value in cache_status_counter.items())),
        "fallback_triggered_count": fallback_count,
        "fallback_triggered_ratio": round(fallback_count / len(records), 4) if records else 0.0,
    }


def build_sampling_notes(
    output_path: Path,
    stats_path: Path,
    notes_path: Path,
    llm_config_path: Path,
    provider: str,
    seed: int,
    top_k: int,
    candidate_count: int,
    pool: Dict[Tuple[str, str], List[PlannedSample]],
    selected_samples: List[PlannedSample],
    stats: Dict[str, object],
) -> str:
    pool_counts = {f"{bucket}-{strength}": len(pool.get((bucket, strength), [])) for bucket, strength in DEFAULT_QUOTAS.keys()}
    selected_category_counter = Counter(sample.query_category for sample in selected_samples)
    command = (
        f"python -B -X utf8 {AD_DIR / 'eval_v5' / 'build_v5_eval_set.py'} "
        f"--provider {provider} "
        f"--output_path {output_path} "
        f"--stats_path {stats_path} "
        f"--notes_path {notes_path}"
    )

    lines = [
        f"# {output_path.stem} 抽样明细说明",
        "",
        "## 1. 文件对应关系",
        "",
        f"- 评测主文件：`{output_path}`",
        f"- 统计文件：`{stats_path}`",
        f"- 抽样说明：`{notes_path}`",
        f"- 抽样脚本：`{AD_DIR / 'eval_v5' / 'build_v5_eval_set.py'}`",
        "",
        "## 2. 抽样来源",
        "",
        f"- 候选池文件：`{AD_DIR / 'data' / 'train_pairs.jsonl'}`",
        f"- 去重口径：按 `normalized_query` 去重，同一个标准化 query 只保留第一次出现的样本",
        f"- top-k 口径：`top_k={top_k}`",
        f"- 候选数口径：`candidate_count={candidate_count}`",
        f"- 随机种子：`seed={seed}`",
        "",
        "## 3. 分层规则",
        "",
        "### 3.1 Query 长度桶",
        "",
        "- `short`：`len(normalized_query) <= 6`",
        "- `medium`：`7 <= len(normalized_query) <= 15`",
        "- `long`：`len(normalized_query) >= 16`",
        "",
        "### 3.2 Evidence 强度桶",
        "",
        "- 命中字段：`item_title / category_path / brand_name(非占位) / seller_name / ranking_signal / relevance_signal`",
        "- 分数公式：`evidence_score = 命中字段总数 / (top_k * 6)`",
        "- `high`：`evidence_score >= 0.75`",
        "- `low`：`evidence_score < 0.75`",
        "",
        "## 4. 目标配额",
        "",
        "| 分层格子 | 配额 |",
        "| --- | ---: |",
    ]

    for bucket in QUERY_BUCKET_ORDER:
        for strength in EVIDENCE_STRENGTH_ORDER:
            lines.append(f"| {bucket}-{strength} | {DEFAULT_QUOTAS[(bucket, strength)]} |")

    lines.extend(
        [
            "",
            "## 5. 候选池覆盖情况",
            "",
            "| 分层格子 | 候选量 |",
            "| --- | ---: |",
        ]
    )

    for bucket in QUERY_BUCKET_ORDER:
        for strength in EVIDENCE_STRENGTH_ORDER:
            key = f"{bucket}-{strength}"
            lines.append(f"| {key} | {pool_counts[key]} |")

    lines.extend(
        [
            "",
            "## 6. 本次实际抽样结果",
            "",
            f"- 总样本数：`{stats['total_samples']}`",
            f"- 请求 provider：`{provider}`",
            f"- 实际 provider 分布：`{stats['provider_distribution']}`",
            f"- model 分布：`{stats['model_distribution']}`",
            f"- llm_status 分布：`{stats['llm_status_distribution']}`",
            "",
            "| 分层格子 | 实际条数 |",
            "| --- | ---: |",
        ]
    )

    for bucket in QUERY_BUCKET_ORDER:
        for strength in EVIDENCE_STRENGTH_ORDER:
            key = f"{bucket}-{strength}"
            lines.append(f"| {key} | {stats['actual_quota_distribution'][key]} |")

    lines.extend(
        [
            "",
            "长度分布：",
            f"- `short`：{stats['query_bucket_distribution'].get('short', 0)}",
            f"- `medium`：{stats['query_bucket_distribution'].get('medium', 0)}",
            f"- `long`：{stats['query_bucket_distribution'].get('long', 0)}",
            "",
            "证据强度分布：",
            f"- `high`：{stats['evidence_strength_distribution'].get('high', 0)}",
            f"- `low`：{stats['evidence_strength_distribution'].get('low', 0)}",
            "",
            "抽中样本类目分布：",
            f"- `digital`：{selected_category_counter.get('digital', 0)}",
            f"- `apparel`：{selected_category_counter.get('apparel', 0)}",
            f"- `food`：{selected_category_counter.get('food', 0)}",
            f"- `health`：{selected_category_counter.get('health', 0)}",
            f"- `misc`：{selected_category_counter.get('misc', 0)}",
            "",
            "## 7. 环境与配置说明",
            "",
            f"- `llm_config_path`：`{llm_config_path}`",
            f"- `DEEPSEEK_API_KEY` 当前进程可见：`{stats['deepseek_api_key_present']}`",
            f"- `SFT_MODEL_PATH` 当前进程可见：`{stats['sft_model_path_present']}`",
            "",
            "## 8. 复现命令",
            "",
            "```powershell",
            "$env:PYTHONDONTWRITEBYTECODE='1'",
            command,
            "```",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_exists(args.pairs_path, "pairs_path")
    ensure_exists(args.corpus_path, "corpus_path")
    ensure_exists(args.rank_path, "rank_path")
    ensure_exists(args.users_path, "users_path")

    llm_config_path = resolve_llm_config(args.provider, args.llm_config_path)
    ensure_exists(llm_config_path, "llm_config_path")

    pool = build_candidate_pool(args.pairs_path, top_k=args.top_k)
    selected_samples = select_stratified_samples(pool=pool, quotas=DEFAULT_QUOTAS, seed=args.seed)
    notes_path = resolve_notes_path(args.output_path, args.notes_path)

    runtime_context = build_v5_runtime_context(
        corpus_path=args.corpus_path,
        rank_path=args.rank_path,
        users_path=args.users_path,
        pairs_path=args.pairs_path,
    )
    warmup = warm_retrieval_cache(
        queries=[sample.query for sample in selected_samples],
        runtime_context=runtime_context,
        top_k=args.top_k,
        allow_expensive_fallback=False,
    )

    print(
        json.dumps(
            {
                "selected_samples": len(selected_samples),
                "warm_cache": warmup,
                "provider": args.provider,
                "llm_config_path": str(llm_config_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    records: List[Dict[str, object]] = []
    total = len(selected_samples)
    for index, sample in enumerate(selected_samples, start=1):
        result = v5_llm_dynamic_creative(
            query=sample.query,
            corpus_path=args.corpus_path,
            rank_path=args.rank_path,
            users_path=args.users_path,
            pairs_path=args.pairs_path,
            llm_config_path=llm_config_path,
            top_k=args.top_k,
            candidate_count=args.candidate_count,
            requested_tone="creative",
            runtime_context=runtime_context,
            provider_override=args.provider,
        )
        sample_record = build_sample_record(
            sample_id=f"v5eval_{index:03d}",
            planned_sample=sample,
            result=result,
            top_k=args.top_k,
        )
        records.append(sample_record)
        if index == 1 or index % 10 == 0 or index == total:
            print(
                f"[{index}/{total}] {sample_record['query_bucket']}-{sample_record['evidence_strength']} "
                f"{sample_record['normalized_query']} -> {sample_record['final_ad_copy']}"
            )

    stats = build_stats(records=records, quotas=DEFAULT_QUOTAS, provider=args.provider, llm_config_path=llm_config_path)
    write_jsonl(records, args.output_path)
    args.stats_path.parent.mkdir(parents=True, exist_ok=True)
    with args.stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_content = build_sampling_notes(
        output_path=args.output_path,
        stats_path=args.stats_path,
        notes_path=notes_path,
        llm_config_path=llm_config_path,
        provider=args.provider,
        seed=args.seed,
        top_k=args.top_k,
        candidate_count=args.candidate_count,
        pool=pool,
        selected_samples=selected_samples,
        stats=stats,
    )
    with notes_path.open("w", encoding="utf-8") as handle:
        handle.write(notes_content)

    print("")
    print("build complete")
    print(f"output_path: {args.output_path}")
    print(f"stats_path: {args.stats_path}")
    print(f"notes_path: {notes_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
