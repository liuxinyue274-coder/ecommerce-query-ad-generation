"""Run a tiny LLM A/B test for ad-copy generation.

This script compares a small set of configured LLMs on three fixed queries
using the existing inference pipeline. It intentionally keeps the test small
to avoid accidental full-volume API use.

Example:
    python ad_generation/compare_llm_models.py ^
      --corpus_path items_lite/train.jsonl ^
      --rank_path rank_lite/train.jsonl ^
      --users_path users_lite/train.jsonl ^
      --pairs_path ad_generation/data/train_pairs.jsonl ^
      --report_path ad_generation/model_compare_report.md
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from infer import infer_once


DEFAULT_QUERIES = [
    "华为手环",
    "宿舍吹风机",
    "iqooz10x手机壳动漫",
]

DEFAULT_QWEN_MODELS: List[Tuple[str, Path]] = [
    ("qwen-flash", Path("ad_generation/llm_config.qwen_flash.json")),
    ("qwen-plus", Path("ad_generation/llm_config.qwen_plus.json")),
    ("qwen-max", Path("ad_generation/llm_config.qwen_max.json")),
]

DEFAULT_DEEPSEEK_MODEL: Tuple[str, Path] = (
    "deepseek-chat",
    Path("ad_generation/llm_config.deepseek_chat.json"),
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the tiny model comparison run."""

    parser = argparse.ArgumentParser(description="Compare a small set of LLMs on three ad-generation queries.")
    parser.add_argument("--corpus_path", type=Path, default=Path("items_lite/train.jsonl"), help="Path to items/corpus JSONL.")
    parser.add_argument("--rank_path", type=Path, default=Path("rank_lite/train.jsonl"), help="Path to rank JSONL.")
    parser.add_argument("--users_path", type=Path, default=Path("users_lite/train.jsonl"), help="Optional users JSONL.")
    parser.add_argument(
        "--pairs_path",
        type=Path,
        default=Path("ad_generation/data/train_pairs.jsonl"),
        help="Path to prebuilt train_pairs JSONL.",
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=Path("ad_generation/model_compare_report.md"),
        help="Output Markdown report path.",
    )
    parser.add_argument(
        "--include_deepseek",
        action="store_true",
        help="Include deepseek-chat in the comparison set.",
    )
    parser.add_argument(
        "--deepseek_config",
        type=Path,
        default=Path("ad_generation/llm_config.deepseek_chat.json"),
        help="Path to deepseek-chat config JSON when --include_deepseek is enabled.",
    )
    return parser.parse_args()


def ensure_required_keys_present(model_specs: List[Tuple[str, Path]]) -> None:
    """Fail fast when the current process cannot access required API keys."""

    missing: List[str] = []
    for _, config_path in model_specs:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        env_name = str(config.get("api_key_env") or "LLM_API_KEY").strip()
        if not os.environ.get(env_name):
            missing.append(env_name)

    if missing:
        unique_missing = sorted(set(missing))
        raise EnvironmentError(
            "Missing required environment variable(s): "
            + ", ".join(unique_missing)
            + ". Please set them in the same PowerShell session before running compare_llm_models.py."
        )


def detect_suspicious_claims(copy_text: str) -> List[str]:
    """Flag lightweight signs of unsupported pricing/sales/efficacy claims."""

    suspicious_tokens = [
        "元",
        "¥",
        "折",
        "优惠",
        "立减",
        "销量",
        "热销",
        "爆款",
        "治",
        "改善",
        "修复",
        "药效",
        "功效",
        "续航",
        "认证",
    ]
    findings: List[str] = []
    for token in suspicious_tokens:
        if token in copy_text:
            findings.append(token)
    return findings


def render_case_block(query: str, model_outputs: Dict[str, Dict[str, object]]) -> str:
    """Render one query block in the final Markdown report."""

    lines = [f"## Query: {query}", ""]
    for model_name, output in model_outputs.items():
        suspicious = output["suspicious_claims"]
        lines.extend(
            [
                f"### {model_name}",
                f"- final ad copy: {output['final_ad_copy']}",
                f"- llm_status: {output['llm_status']}",
                f"- fallback_triggered: {output['fallback_triggered']}",
                f"- bad_copy_detected: {output['bad_copy_detected']}",
                f"- suspicious_claims: {'无' if not suspicious else ', '.join(suspicious)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def build_report(results: Dict[str, Dict[str, Dict[str, object]]], model_names: List[str]) -> str:
    """Build the final Markdown report body."""

    model_label = "、".join(f"`{name}`" for name in model_names)
    lines = [
        "# LLM Model Compare Report",
        "",
        "本报告只对 3 条 query 做最小 A/B 测试，保持：`top_k=3`、`copy_style=llm`、`copy_tone=creative`。",
        "",
        f"对比模型：{model_label}",
        "",
    ]
    for query, model_outputs in results.items():
        lines.append(render_case_block(query, model_outputs))
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- `fallback_triggered=True` 说明该模型在当前调用中没有成功给出最终 LLM 文案，结果已走现有 fallback 机制。",
            "- `suspicious_claims` 只是轻量规则提示，不等价于严格事实核验。",
            "- 这份报告不覆盖 `eval_report.md`，仅用于小规模模型对比。",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    """Run the tiny three-model comparison and write a Markdown report."""

    args = parse_args()

    model_specs = list(DEFAULT_QWEN_MODELS)
    if args.include_deepseek:
        model_specs.append((DEFAULT_DEEPSEEK_MODEL[0], args.deepseek_config))

    if not args.corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {args.corpus_path}")
    if not args.rank_path.exists():
        raise FileNotFoundError(f"Rank file not found: {args.rank_path}")
    if not args.pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {args.pairs_path}")
    for _, config_path in model_specs:
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    ensure_required_keys_present(model_specs)

    results: Dict[str, Dict[str, Dict[str, object]]] = {}

    for query in DEFAULT_QUERIES:
        results[query] = {}
        for model_name, config_path in model_specs:
            output = infer_once(
                query=query,
                corpus_path=args.corpus_path,
                rank_path=args.rank_path,
                users_path=args.users_path if args.users_path.exists() else None,
                pairs_path=args.pairs_path,
                top_k=3,
                copy_style="llm",
                copy_tone="creative",
                llm_config_path=config_path,
            )
            results[query][model_name] = {
                "final_ad_copy": str(output.get("final_ad_copy") or ""),
                "llm_status": str(output.get("llm_status") or ""),
                "fallback_triggered": bool(output.get("fallback_triggered")),
                "bad_copy_detected": bool(output.get("bad_copy_detected")),
                "suspicious_claims": detect_suspicious_claims(str(output.get("final_ad_copy") or "")),
            }

    report_body = build_report(results, [model_name for model_name, _ in model_specs])
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(report_body, encoding="utf-8")
    print(f"Model comparison report written to: {args.report_path}")


if __name__ == "__main__":
    main()
