"""Build a small eval set and compare ad-generation baselines.

Usage example:

python ad_generation/evaluate_samples.py ^
  --pairs_path ad_generation/data/train_pairs.jsonl ^
  --eval_cases_path ad_generation/eval_cases.jsonl ^
  --report_path ad_generation/eval_report.md ^
  --sample_size 39 ^
  --include_llm ^
  --llm_config ad_generation/llm_config.example.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from build_train_pairs import clean_text, iter_jsonl
from infer import build_generation_output


DOMAIN_ORDER = ["digital", "apparel", "food", "health", "misc"]
LENGTH_ORDER = ["short", "medium", "long"]
PRIMARY_DOMAINS = ["digital", "apparel", "food", "health"]
BUCKET_QUOTA = 3
MISC_QUOTA = 1

DOMAIN_KEYWORDS = {
    "digital": ["手机/数码/电脑办公", "家用电器", "电脑", "数码", "手机", "耳机", "相机", "键盘", "显示器", "充电"],
    "apparel": ["女装", "男装", "内衣", "童装", "女鞋", "男鞋", "箱包", "饰品", "套装", "裤", "裙", "上衣"],
    "food": ["零食", "生鲜", "速食", "干货", "饮料", "酒水", "粮油", "食品", "坚果", "特产", "饼干", "奶茶"],
    "health": ["营养健康", "医疗保健", "个护清洁", "美容护肤", "美妆", "身体护理", "口腔护理", "钙片", "氨糖", "保健"],
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the evaluation workflow."""

    parser = argparse.ArgumentParser(description="Generate eval cases and a markdown comparison report.")
    parser.add_argument(
        "--pairs_path",
        type=Path,
        default=Path("ad_generation/data/train_pairs.jsonl"),
        help="Path to train_pairs JSONL.",
    )
    parser.add_argument(
        "--eval_cases_path",
        type=Path,
        default=Path("ad_generation/eval_cases.jsonl"),
        help="Path to write eval cases JSONL.",
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=Path("ad_generation/eval_report.md"),
        help="Path to write the evaluation markdown report.",
    )
    parser.add_argument("--sample_size", type=int, default=39, help="Target number of eval cases.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic bucket sampling.")
    parser.add_argument(
        "--include_llm",
        action="store_true",
        help="Whether to generate the optional baseline_llm_topk outputs.",
    )
    parser.add_argument(
        "--llm_config",
        type=Path,
        default=Path("ad_generation/llm_config.example.json"),
        help="Path to LLM config JSON, used when --include_llm is enabled.",
    )
    return parser.parse_args()


def query_length_bucket(query: str) -> str:
    """Bucket queries into short / medium / long groups."""

    length = len(clean_text(query))
    if length <= 6:
        return "short"
    if length <= 15:
        return "medium"
    return "long"


def infer_domain(query: str, target_item: Optional[Dict[str, object]]) -> str:
    """Map a case to a coarse domain label used by the eval set."""

    target_item = target_item or {}
    category_path = clean_text(target_item.get("category_path"))
    title = clean_text(target_item.get("item_title") or target_item.get("title"))
    haystack = f"{category_path} {title} {clean_text(query)}".lower()

    for domain in PRIMARY_DOMAINS:
        keywords = DOMAIN_KEYWORDS[domain]
        if any(keyword.lower() in haystack for keyword in keywords):
            return domain
    return "misc"


def reservoir_add(
    store: Dict[Tuple[str, str], List[Dict[str, object]]],
    seen: Dict[Tuple[str, str], int],
    key: Tuple[str, str],
    item: Dict[str, object],
    limit: int,
    rng: random.Random,
) -> None:
    """Reservoir-sample a bounded number of items per bucket."""

    seen[key] = seen.get(key, 0) + 1
    bucket = store.setdefault(key, [])
    if len(bucket) < limit:
        bucket.append(item)
        return

    replace_index = rng.randint(0, seen[key] - 1)
    if replace_index < limit:
        bucket[replace_index] = item


def slim_case(record: Dict[str, object], domain: str, length_bucket: str) -> Dict[str, object]:
    """Keep only the fields needed for evaluation and reporting."""

    return {
        "query": record.get("query", ""),
        "user_id": record.get("user_id"),
        "session_id": record.get("session_id"),
        "user_profile": record.get("user_profile") or {},
        "recent_behavior_titles": record.get("recent_behavior_titles") or [],
        "evidence_items": record.get("evidence_items") or [],
        "target_item": record.get("target_item") or {},
        "target_copy": record.get("target_copy", ""),
        "domain": domain,
        "query_length_bucket": length_bucket,
    }


def sample_eval_cases(pairs_path: Path, sample_size: int, seed: int) -> List[Dict[str, object]]:
    """Sample a small, diverse evaluation set from train_pairs."""

    if not pairs_path.exists():
        raise FileNotFoundError(f"train_pairs file not found: {pairs_path}")

    rng = random.Random(seed)
    bucketed: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    seen: Dict[Tuple[str, str], int] = {}
    fallback_bucketed: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    fallback_seen: Dict[Tuple[str, str], int] = {}

    for record in iter_jsonl(pairs_path):
        evidence_items = record.get("evidence_items") or []
        target_item = record.get("target_item") or {}
        if not evidence_items or not target_item:
            continue

        query = clean_text(record.get("query"))
        if not query:
            continue

        domain = infer_domain(query, target_item)
        length_bucket = query_length_bucket(query)
        case = slim_case(record, domain=domain, length_bucket=length_bucket)

        if domain in PRIMARY_DOMAINS:
            reservoir_add(
                store=bucketed,
                seen=seen,
                key=(domain, length_bucket),
                item=case,
                limit=BUCKET_QUOTA,
                rng=rng,
            )
        else:
            reservoir_add(
                store=fallback_bucketed,
                seen=fallback_seen,
                key=(domain, length_bucket),
                item=case,
                limit=MISC_QUOTA,
                rng=rng,
            )

    selected: List[Dict[str, object]] = []
    seen_query: set[str] = set()

    def append_unique(cases: Iterable[Dict[str, object]]) -> None:
        for case in cases:
            query = str(case.get("query") or "")
            if query and query not in seen_query:
                selected.append(case)
                seen_query.add(query)

    for domain in DOMAIN_ORDER:
        for length_bucket in LENGTH_ORDER:
            append_unique(bucketed.get((domain, length_bucket), []))

    for length_bucket in LENGTH_ORDER:
        append_unique(fallback_bucketed.get(("misc", length_bucket), []))

    if len(selected) < 30:
        raise RuntimeError(
            f"Only sampled {len(selected)} eval cases from the configured buckets, which is below the required minimum."
        )

    selected.sort(
        key=lambda case: (
            DOMAIN_ORDER.index(str(case.get("domain", "misc"))) if str(case.get("domain", "misc")) in DOMAIN_ORDER else 99,
            LENGTH_ORDER.index(str(case.get("query_length_bucket", "medium")))
            if str(case.get("query_length_bucket", "medium")) in LENGTH_ORDER
            else 99,
            str(case.get("query")),
        )
    )
    return selected[:sample_size]


def write_jsonl(records: Iterable[Dict[str, object]], path: Path) -> None:
    """Write JSONL records using Windows-friendly UTF-8-SIG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_result_payload(
    case: Dict[str, object],
    top_k: int,
    source: str,
) -> Dict[str, object]:
    """Convert one eval case back into the inference payload schema."""

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
    """Choose the train_pairs source label that best matches the eval case."""

    if case.get("user_id") is not None:
        return "train_pairs_user_exact"
    if case.get("session_id") is not None:
        return "train_pairs_session_exact"
    return "train_pairs_query_exact"


def evaluate_case(
    case: Dict[str, object],
    include_llm: bool = False,
    llm_config_path: Optional[Path] = None,
) -> Dict[str, object]:
    """Generate the baseline outputs for a single eval case."""

    query = str(case.get("query") or "")
    user_id = case.get("user_id")

    template_top1 = build_generation_output(
        raw_query=query,
        result=build_result_payload(case=case, top_k=1, source="train_pairs_query_exact"),
        copy_style="template",
    )
    template_personalized = build_generation_output(
        raw_query=query,
        result=build_result_payload(case=case, top_k=1, source=personalized_source(case)),
        copy_style="template",
        user_id=int(user_id) if user_id is not None else None,
    )
    summary_topk = build_generation_output(
        raw_query=query,
        result=build_result_payload(case=case, top_k=3, source=personalized_source(case)),
        copy_style="summary",
        user_id=int(user_id) if user_id is not None else None,
    )

    result_payload = {
        "query": query,
        "domain": case.get("domain", "misc"),
        "query_length_bucket": case.get("query_length_bucket", "medium"),
        "baseline_template_top1": template_top1,
        "baseline_template_personalized": template_personalized,
        "baseline_summary_topk": summary_topk,
    }

    if include_llm:
        llm_topk = build_generation_output(
            raw_query=query,
            result=build_result_payload(case=case, top_k=3, source=personalized_source(case)),
            copy_style="llm",
            user_id=int(user_id) if user_id is not None else None,
            llm_config_path=llm_config_path,
        )
        result_payload["baseline_llm_topk"] = llm_topk
        result_payload["llm_provider"] = llm_topk.get("llm_provider", "")
        result_payload["llm_status"] = llm_topk.get("llm_status", "")
        result_payload["llm_fallback_triggered"] = bool(llm_topk.get("fallback_triggered"))
        result_payload["llm_fallback_reason"] = llm_topk.get("fallback_reason", "")

    return result_payload


def build_report_text(
    eval_cases_path: Path,
    report_path: Path,
    sample_size: int,
    case_results: List[Dict[str, object]],
    include_llm: bool = False,
) -> str:
    """Render the markdown report."""

    fallback_count = sum(1 for item in case_results if item["baseline_summary_topk"]["fallback_triggered"])
    summary_ratio = fallback_count / sample_size if sample_size else 0.0
    personalized_changed = sum(
        1
        for item in case_results
        if item["baseline_template_top1"]["final_ad_copy"] != item["baseline_template_personalized"]["final_ad_copy"]
    )

    domain_counts = Counter(str(item.get("domain", "misc")) for item in case_results)
    length_counts = Counter(str(item.get("query_length_bucket", "medium")) for item in case_results)
    summary_reason_counts = Counter(
        str(item["baseline_summary_topk"]["fallback_reason"] or "none") for item in case_results
    )
    llm_provider_counts = Counter(
        str(item.get("llm_provider") or "none") for item in case_results if include_llm
    )
    llm_fallback_count = sum(1 for item in case_results if include_llm and item.get("llm_fallback_triggered"))
    llm_ratio = llm_fallback_count / sample_size if sample_size and include_llm else 0.0
    llm_vs_summary_changed = sum(
        1
        for item in case_results
        if include_llm
        and item["baseline_llm_topk"]["final_ad_copy"] != item["baseline_summary_topk"]["final_ad_copy"]
    )
    llm_vs_template_changed = sum(
        1
        for item in case_results
        if include_llm
        and item["baseline_llm_topk"]["final_ad_copy"] != item["baseline_template_top1"]["final_ad_copy"]
    )

    lines: List[str] = [
        "# Ad Generation Eval Report",
        "",
        "## 实验设置",
        "",
        f"- 评估样本文件：`{eval_cases_path.as_posix()}`",
        f"- 报告文件：`{report_path.as_posix()}`",
        f"- 样本数：`{sample_size}`",
        "- baseline 1 `baseline_template_top1`：`template` 模式，`top_k=1`，只按 query 级证据生成。",
        "- baseline 2 `baseline_template_personalized`：`template` 模式，`top_k=1`，优先带 `user_id / session_id` 的弱个性化来源标签。",
        "- baseline 3 `baseline_summary_topk`：`summary` 模式，`top_k=3`，输出 fallback 标记与原因。",
        (
            "- baseline 4 `baseline_llm_topk`：`llm` 模式，`top_k=3`，"
            "通过 `llm_generator.py` 生成文案，并记录 provider / status / fallback。"
            if include_llm
            else "- baseline 4 `baseline_llm_topk`：未开启（传 `--include_llm` 可加入 LLM 对照）。"
        ),
        "",
        "## 样本覆盖",
        "",
        f"- 域覆盖：`{dict(domain_counts)}`",
        f"- query 长度覆盖：`{dict(length_counts)}`",
        "",
        "## 基线统计",
        "",
        f"- `baseline_template_top1` 样本数：`{sample_size}`",
        f"- `baseline_template_personalized` 样本数：`{sample_size}`",
        f"- `baseline_summary_topk` 样本数：`{sample_size}`",
        f"- `baseline_llm_topk` 样本数：`{sample_size if include_llm else 0}`",
        f"- `summary` fallback 数：`{fallback_count}/{sample_size}`",
        f"- `summary` fallback 比例：`{summary_ratio:.2%}`",
        f"- 个性化文案与非个性化文案不同的样本数：`{personalized_changed}/{sample_size}`",
        f"- `summary` 原因分布：`{dict(summary_reason_counts)}`",
        "",
        "## 对照样本（前 10 条）",
        "",
    ]

    if include_llm:
        lines.extend(
            [
                "## LLM 统计",
                "",
                f"- LLM provider 分布：`{dict(llm_provider_counts)}`",
                f"- `llm` fallback 数：`{llm_fallback_count}/{sample_size}`",
                f"- `llm` fallback 比例：`{llm_ratio:.2%}`",
                f"- `baseline_llm_topk` 与 `baseline_summary_topk` 不同的样本数：`{llm_vs_summary_changed}/{sample_size}`",
                f"- `baseline_llm_topk` 与 `baseline_template_top1` 不同的样本数：`{llm_vs_template_changed}/{sample_size}`",
                "",
            ]
        )

    for index, item in enumerate(case_results[:10], start=1):
        template_top1 = item["baseline_template_top1"]
        template_personalized = item["baseline_template_personalized"]
        summary_topk = item["baseline_summary_topk"]
        lines.extend(
            [
                f"### Case {index}",
                f"- query：`{item['query']}`",
                f"- domain / length：`{item['domain']}` / `{item['query_length_bucket']}`",
                f"- baseline_template_top1：{template_top1['final_ad_copy']}",
                f"- baseline_template_personalized：{template_personalized['final_ad_copy']}",
                (
                    f"- baseline_summary_topk：{summary_topk['final_ad_copy']} "
                    f"(fallback={summary_topk['fallback_triggered']}, reason={summary_topk['fallback_reason'] or 'none'})"
                ),
                (
                    f"- baseline_llm_topk：{item['baseline_llm_topk']['final_ad_copy']} "
                    f"(provider={item.get('llm_provider') or 'none'}, "
                    f"status={item.get('llm_status') or 'none'}, "
                    f"fallback={item.get('llm_fallback_triggered')}, "
                    f"reason={item.get('llm_fallback_reason') or 'none'})"
                    if include_llm
                    else "- baseline_llm_topk：未开启"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## 定性总结",
            "",
            "- `baseline_template_top1` 最像商品复述，因为它稳定围绕 top-1 商品标题、店铺和类目展开。",
            "- `baseline_template_personalized` 在当前评估集上通常只改变命中标签，不一定改变最终文案；这说明弱个性化输入已经接通，但受限于 `train_pairs` 基本是一条 query 对应一条样本。",
            "- `baseline_summary_topk` 在不 fallback 时更自然，也更适合表达“先看哪类商品方向”；它不强行复述某一个完整商品标题。",
            (
                f"- 本次 `summary` 的 fallback 比例是 `{summary_ratio:.2%}`。"
                " fallback 主要发生在证据不足或 top-k 之间缺少稳定共性时，此时系统会回退到稳妥的 template 文案。"
            ),
            (
                f"- `baseline_llm_topk` 的 fallback 比例是 `{llm_ratio:.2%}`。"
                " 它的语言更自然，更接近真实广告文案，但需要真实性约束和 fallback 机制。"
                if include_llm
                else "- `baseline_llm_topk` 本次未开启；如需比较语言自然度，可带 `--include_llm` 重新生成报告。"
            ),
            "- `mock` provider 当前只用于验证链路，后续可以替换成本地 Qwen 或真实 API。",
            "- 如果后续课程展示希望突出‘个性化确实生效’，更适合补一个重复 query 更多、同 query 多用户的评估子集。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the full evaluation workflow: sample cases, score baselines, and write the report."""

    args = parse_args()
    eval_cases = sample_eval_cases(args.pairs_path, sample_size=args.sample_size, seed=args.seed)
    write_jsonl(eval_cases, args.eval_cases_path)

    case_results = [
        evaluate_case(case, include_llm=args.include_llm, llm_config_path=args.llm_config)
        for case in eval_cases
    ]
    report_text = build_report_text(
        eval_cases_path=args.eval_cases_path,
        report_path=args.report_path,
        sample_size=len(case_results),
        case_results=case_results,
        include_llm=args.include_llm,
    )
    args.report_path.write_text(report_text, encoding="utf-8-sig")

    fallback_count = sum(1 for item in case_results if item["baseline_summary_topk"]["fallback_triggered"])
    llm_fallback_count = sum(1 for item in case_results if args.include_llm and item.get("llm_fallback_triggered"))
    print(f"Generated eval cases: {args.eval_cases_path}")
    print(f"Generated eval report: {args.report_path}")
    print(f"Sample count: {len(case_results)}")
    print(f"Summary fallback ratio: {fallback_count}/{len(case_results)} = {fallback_count / len(case_results):.2%}")
    if args.include_llm:
        print(f"LLM fallback ratio: {llm_fallback_count}/{len(case_results)} = {llm_fallback_count / len(case_results):.2%}")


if __name__ == "__main__":
    main()
