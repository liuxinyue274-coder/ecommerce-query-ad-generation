"""Validate and preview the SFT training pipeline for ad-copy data.

Usage:
python -X utf8 ad_generation/train_sft_ad_copy.py --data outputs/real_sft_ad_copy.jsonl --dry_run
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple


NOTICE = (
    "当前 real_sft_ad_copy.jsonl 是由 deepseek_chat 生成并经过规则筛选的高质量伪标注数据，"
    "不是人工金标，主要用于 SFT 链路验证。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate ad-copy SFT data and preview training text.")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to the SFT JSONL data, e.g. outputs/real_sft_ad_copy.jsonl.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Validate data and build training text without real training.")
    parser.add_argument("--model_name_or_path", type=str, default="", help="Reserved model path for future real training.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/sft_ad_copy_lora"),
        help="Reserved output directory for future real training.",
    )
    parser.add_argument("--max_steps", type=int, default=50, help="Reserved training max steps.")
    parser.add_argument("--batch_size", type=int, default=1, help="Reserved training batch size.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Reserved training learning rate.")
    parser.add_argument(
        "--use_lora",
        type=str,
        default="true",
        help="Reserved LoRA flag. Accepts true/false.",
    )
    return parser.parse_args()


def _parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _message_map(messages: Iterable[Dict[str, object]]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for message in messages:
        role = str(message.get("role") or "").strip()
        content = str(message.get("content") or "")
        if role:
            mapping[role] = content
    return mapping


def _build_training_text(messages: Iterable[Dict[str, object]]) -> str:
    mapping = _message_map(messages)
    return "\n".join(
        [
            "<|system|>",
            mapping.get("system", ""),
            "<|user|>",
            mapping.get("user", ""),
            "<|assistant|>",
            mapping.get("assistant", ""),
        ]
    )


def _validate_record(record: Dict[str, object]) -> Tuple[bool, List[str], Optional[Dict[str, object]]]:
    errors: List[str] = []
    messages = record.get("messages")
    metadata = record.get("metadata")

    if not isinstance(messages, list) or not messages:
        errors.append("missing_messages")
        return False, errors, None
    if not isinstance(metadata, dict):
        errors.append("missing_metadata")
        return False, errors, None

    mapping = _message_map(messages)
    for role in ("system", "user", "assistant"):
        if role not in mapping:
            errors.append(f"missing_{role}")

    assistant_content = mapping.get("assistant", "").strip()
    if not assistant_content:
        errors.append("empty_assistant")

    for key in ("provider", "score", "evidence_confidence"):
        if key not in metadata:
            errors.append(f"missing_metadata_{key}")

    if errors:
        return False, errors, None

    normalized = {
        "messages": messages,
        "metadata": metadata,
        "training_text": _build_training_text(messages),
        "assistant_length": len(assistant_content),
        "provider": str(metadata.get("provider") or ""),
        "evidence_confidence": str(metadata.get("evidence_confidence") or ""),
        "score": float(metadata.get("score") or 0.0),
    }
    return True, [], normalized


def _dependency_status() -> Dict[str, bool]:
    required = ["torch", "transformers", "peft", "trl"]
    status: Dict[str, bool] = {}
    for name in required:
        try:
            importlib.import_module(name)
            status[name] = True
        except Exception:
            status[name] = False
    return status


def _render_preview_md(
    valid_samples: List[Dict[str, object]],
    stats: Dict[str, object],
    config: Dict[str, object],
    preview_path: Path,
) -> None:
    lines: List[str] = [
        "# SFT Train Preview",
        "",
        NOTICE,
        "",
        "## Stats",
        "",
        f"- total_samples: `{stats['total_samples']}`",
        f"- valid_samples: `{stats['valid_samples']}`",
        f"- invalid_samples: `{stats['invalid_samples']}`",
        f"- avg_assistant_length: `{stats['avg_assistant_length']}`",
        f"- provider_distribution: `{stats['provider_distribution']}`",
        f"- evidence_confidence_distribution: `{stats['evidence_confidence_distribution']}`",
        f"- avg_score: `{stats['avg_score']}`",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Training Text Preview",
        "",
    ]

    for index, sample in enumerate(valid_samples[:2], start=1):
        lines.extend(
            [
                f"### Sample {index}",
                "",
                "```text",
                str(sample.get("training_text") or ""),
                "```",
                "",
            ]
        )

    preview_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_config(config_path: Path, config: Dict[str, object]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.data.exists():
        raise FileNotFoundError(f"SFT data file not found: {args.data}")

    use_lora = _parse_bool(args.use_lora)
    records = _load_jsonl(args.data)

    valid_samples: List[Dict[str, object]] = []
    invalid_reasons: Counter[str] = Counter()
    provider_counts: Counter[str] = Counter()
    confidence_counts: Counter[str] = Counter()
    scores: List[float] = []
    assistant_lengths: List[int] = []

    for record in records:
        is_valid, errors, normalized = _validate_record(record)
        if not is_valid or normalized is None:
            for error in errors:
                invalid_reasons[error] += 1
            continue

        valid_samples.append(normalized)
        provider_counts[normalized["provider"]] += 1
        confidence_counts[normalized["evidence_confidence"]] += 1
        scores.append(float(normalized["score"]))
        assistant_lengths.append(int(normalized["assistant_length"]))

    stats = {
        "total_samples": len(records),
        "valid_samples": len(valid_samples),
        "invalid_samples": len(records) - len(valid_samples),
        "avg_assistant_length": round(mean(assistant_lengths), 2) if assistant_lengths else 0.0,
        "provider_distribution": dict(provider_counts),
        "evidence_confidence_distribution": dict(confidence_counts),
        "avg_score": round(mean(scores), 2) if scores else 0.0,
        "invalid_reason_distribution": dict(invalid_reasons),
    }

    dependency_status = _dependency_status()
    config = {
        "notice": NOTICE,
        "data_path": str(args.data),
        "dry_run": bool(args.dry_run),
        "model_name_or_path": args.model_name_or_path,
        "output_dir": str(args.output_dir),
        "max_steps": int(args.max_steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "use_lora": bool(use_lora),
        "dependency_status": dependency_status,
        "stats": stats,
    }

    config_path = Path("outputs/sft_train_config.json")
    preview_path = Path("outputs/sft_train_preview.md")
    _write_config(config_path, config)
    _render_preview_md(valid_samples=valid_samples, stats=stats, config=config, preview_path=preview_path)

    print(NOTICE)
    print("")
    print(f"total_samples: {stats['total_samples']}")
    print(f"valid_samples: {stats['valid_samples']}")
    print(f"invalid_samples: {stats['invalid_samples']}")
    print(f"avg_assistant_length: {stats['avg_assistant_length']}")
    print(f"provider_distribution: {stats['provider_distribution']}")
    print(f"evidence_confidence_distribution: {stats['evidence_confidence_distribution']}")
    print(f"avg_score: {stats['avg_score']}")
    if invalid_reasons:
        print(f"invalid_reason_distribution: {dict(invalid_reasons)}")

    print("")
    print("=== Training Text Sample 1 ===")
    print(valid_samples[0]["training_text"] if valid_samples else "(empty)")
    print("")
    print("=== Training Text Sample 2 ===")
    print(valid_samples[1]["training_text"] if len(valid_samples) > 1 else "(empty)")
    print("")
    print(f"Output config: {config_path}")
    print(f"Output preview: {preview_path}")

    if args.dry_run:
        print("Dry run mode: training was not started.")
        return

    missing = [name for name, ok in dependency_status.items() if not ok]
    if missing:
        print("")
        print(
            "Non-dry-run requested, but required training dependencies are missing: "
            + ", ".join(missing)
            + "."
        )
        print("Please install the missing packages first, then rerun without --dry_run.")
        return

    print("")
    print("Real training entry is reserved but not enabled by default in this validation script.")
    print("Current run stops after dependency check and config generation.")


if __name__ == "__main__":
    main()
