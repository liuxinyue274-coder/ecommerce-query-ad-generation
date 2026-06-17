"""Convert train_pairs.jsonl into chat-style SFT data for later instruction tuning.

Usage example:

python ad_generation/build_sft_data.py ^
  --input_path ad_generation/data/train_pairs.jsonl ^
  --output_path ad_generation/data/sft_train.jsonl ^
  --sample_path ad_generation/data/sft_sample.jsonl ^
  --max_samples 5000 ^
  --top_k 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from build_train_pairs import clean_text, iter_jsonl
from prompt_builder import build_evidence_block, build_user_context


SYSTEM_PROMPT = "你是电商搜索广告文案生成助手。"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for SFT data conversion."""

    parser = argparse.ArgumentParser(description="Build SFT-style JSONL from train_pairs pseudo labels.")
    parser.add_argument(
        "--input_path",
        type=Path,
        default=Path("ad_generation/data/train_pairs.jsonl"),
        help="Path to the source train_pairs JSONL.",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("ad_generation/data/sft_train.jsonl"),
        help="Path to write the SFT training JSONL.",
    )
    parser.add_argument(
        "--sample_path",
        type=Path,
        default=Path("ad_generation/data/sft_sample.jsonl"),
        help="Path to write a small SFT sample JSONL for manual inspection.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5000,
        help="Maximum number of SFT records to export.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=3,
        help="Number of evidence items to include in each SFT prompt.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=20,
        help="Number of records to write into sft_sample.jsonl.",
    )
    return parser.parse_args()


def _format_user_content(record: Dict[str, object], top_k: int) -> str:
    """Assemble the user message for one SFT training example."""

    query = clean_text(record.get("query"))
    user_context = build_user_context(
        user_profile=record.get("user_profile") or {},
        recent_behavior_titles=record.get("recent_behavior_titles") or [],
    )
    evidence_block = build_evidence_block((record.get("evidence_items") or [])[:top_k])

    parts: List[str] = [
        "根据以下用户搜索词和商品证据，生成一条真实、自然、简洁的中文广告文案。",
        "",
        f"query:\n{query}",
    ]

    if user_context:
        parts.extend(["", f"用户信息:\n{user_context}"])

    if evidence_block:
        parts.extend(["", f"top-k 商品证据:\n{evidence_block}"])

    parts.extend(
        [
            "",
            "生成约束：",
            "1. 不编造价格、折扣、销量。",
            "2. 不编造商品不存在的功效。",
            "3. 只输出一条 20~50 字中文广告文案。",
            "4. 语气自然，不要夸张营销。",
            "",
            "只输出文案本身，不要解释。",
        ]
    )
    return "\n".join(parts)


def build_sft_record(record: Dict[str, object], top_k: int) -> Dict[str, object]:
    """Convert one train_pairs record into a chat-style SFT sample."""

    target_copy = clean_text(record.get("target_copy"))
    return {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": _format_user_content(record, top_k=top_k),
            },
            {
                "role": "assistant",
                "content": target_copy,
            },
        ],
        "metadata": {
            "query": clean_text(record.get("query")),
            "user_id": record.get("user_id"),
            "session_id": record.get("session_id"),
            "source": "template_pseudo_label",
        },
    }


def write_jsonl(records: Iterable[Dict[str, object]], path: Path) -> None:
    """Write JSONL records using Windows-friendly UTF-8-SIG."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    """Read train_pairs, convert them to SFT records, and write full/sample outputs."""

    args = parse_args()
    if not args.input_path.exists():
        raise FileNotFoundError(f"Input train_pairs file not found: {args.input_path}")

    converted: List[Dict[str, object]] = []
    for record in iter_jsonl(args.input_path):
        target_copy = clean_text(record.get("target_copy"))
        if not target_copy:
            continue

        converted.append(build_sft_record(record, top_k=max(1, args.top_k)))
        if len(converted) >= max(1, args.max_samples):
            break

    write_jsonl(converted, args.output_path)
    write_jsonl(converted[: max(1, args.sample_size)], args.sample_path)

    print(f"Generated SFT train file: {args.output_path}")
    print(f"Generated SFT sample file: {args.sample_path}")
    print(f"Sample count: {len(converted)}")


if __name__ == "__main__":
    main()
