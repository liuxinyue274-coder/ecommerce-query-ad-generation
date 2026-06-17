"""Build first-version query-to-evidence training pairs for ad generation.

Example:
    python ad_generation/build_train_pairs.py \
      --corpus_path data/corpus.jsonl \
      --rank_path data/rank.jsonl \
      --users_path data/users.jsonl \
      --output_path ad_generation/data/train_pairs.jsonl \
      --top_k 3

The script only uses local JSONL files and writes JSONL output. It does not
modify any existing recall / relevance / ranking module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from prompt_builder import render_template_copy


def clean_text(value: object) -> str:
    """Convert a value to a stripped string."""

    if value is None:
        return ""
    return str(value).strip()


def first_non_empty(data: Optional[Dict[str, object]], keys: Iterable[str], default: str = "") -> str:
    """Read the first non-empty field from a mapping."""

    if not data:
        return default
    for key in keys:
        value = clean_text(data.get(key))
        if value:
            return value
    return default


def safe_int(value: object, default: int = 0) -> int:
    """Convert a value to int without raising."""

    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


def safe_float(value: object, default: float = 0.0) -> float:
    """Convert a value to float without raising."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def iter_jsonl(path: Path) -> Iterator[Dict[str, object]]:
    """Yield JSON objects from a JSONL file."""

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                yield json.loads(raw)
            except json.JSONDecodeError:
                print(f"[WARN] Skip invalid JSON at {path}:{line_number}")


def build_category_path(item: Dict[str, object]) -> str:
    """Compose a category path from compatible category fields."""

    parts: List[str] = []
    for keys in (
        ["category_level1_name", "category_l1_name", "category1_name"],
        ["category_level2_name", "category_l2_name", "category2_name"],
        ["category_level3_name", "category_l3_name", "category3_name"],
    ):
        value = first_non_empty(item, keys)
        if value and value.upper() != "UNKNOWN":
            parts.append(value)
    return " > ".join(parts)


def canonicalize_item(item: Dict[str, object]) -> Dict[str, object]:
    """Normalize item metadata with tolerant field-name lookup."""

    item_id = safe_int(item.get("item_id", item.get("target_item_id", 0)))
    category_path = first_non_empty(item, ["category_path"])
    if not category_path:
        category_path = build_category_path(item)
    return {
        "item_id": item_id,
        "item_title": first_non_empty(item, ["item_title", "title", "item_name", "name"]),
        "brand_name": first_non_empty(item, ["brand_name", "brand"]),
        "seller_name": first_non_empty(item, ["seller_name", "seller", "shop_name"]),
        "category_path": category_path,
    }


def load_corpus_map(corpus_path: Path) -> Dict[int, Dict[str, object]]:
    """Load item metadata as an item_id keyed dictionary."""

    corpus_map: Dict[int, Dict[str, object]] = {}
    for item in iter_jsonl(corpus_path):
        item_id = safe_int(item.get("item_id"))
        if item_id <= 0:
            continue
        corpus_map[item_id] = canonicalize_item(item)
    return corpus_map


def load_user_map(users_path: Optional[Path]) -> Dict[int, Dict[str, object]]:
    """Load optional user profile data."""

    if not users_path or not users_path.exists():
        return {}

    user_map: Dict[int, Dict[str, object]] = {}
    for user in iter_jsonl(users_path):
        user_id = safe_int(user.get("user_id"))
        if user_id <= 0:
            continue
        user_map[user_id] = {
            "gender": first_non_empty(user, ["gender"]),
            "age_bucket": first_non_empty(user, ["age_bucket", "age"]),
            "city": first_non_empty(user, ["fre_city", "city"]),
            "province": first_non_empty(user, ["fre_province", "province"]),
            "country": first_non_empty(user, ["fre_country", "country"]),
        }
    return user_map


def extract_recent_behavior_titles(
    record: Dict[str, object],
    corpus_map: Dict[int, Dict[str, object]],
    limit: int = 5,
) -> List[str]:
    """Map recent clicked/purchased item IDs to readable titles."""

    item_ids: List[int] = []
    for field_name in ("recently_clicked_item_ids", "recently_purchased_item_ids"):
        values = record.get(field_name) or []
        if not isinstance(values, list):
            continue
        for value in values:
            item_id = safe_int(value)
            if item_id > 0:
                item_ids.append(item_id)

    titles: List[str] = []
    seen = set()
    for item_id in item_ids:
        title = clean_text(corpus_map.get(item_id, {}).get("item_title"))
        if title and title not in seen:
            seen.add(title)
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def is_positive_record(record: Dict[str, object]) -> bool:
    """Return True when the rank log contains a positive interaction."""

    return safe_int(record.get("is_purchased")) == 1 or safe_int(record.get("is_clicked")) == 1


def compute_sample_ranking_signal(record: Dict[str, object]) -> float:
    """Compute a deterministic ranking signal from existing rank fields."""

    item_stats = record.get("target_item_statistical_features") or {}
    return (
        5.0 * safe_int(record.get("is_purchased"))
        + 2.0 * safe_int(record.get("is_clicked"))
        + 0.001 * safe_float(item_stats.get("item_show_cnt_30d_hist"))
        + 0.01 * safe_float(item_stats.get("item_click_cnt_30d_hist"))
        + 0.05 * safe_float(item_stats.get("item_order_cnt_30d_hist"))
    )


def collect_query_groups(
    rank_path: Path,
    corpus_map: Dict[int, Dict[str, object]],
    users_map: Optional[Dict[int, Dict[str, object]]] = None,
    query_filter: Optional[str] = None,
) -> Dict[str, Dict[str, object]]:
    """Aggregate positive query-item evidence from rank logs."""

    del users_map  # Reserved for future use without changing the function signature.
    groups: Dict[str, Dict[str, object]] = {}

    for record in iter_jsonl(rank_path):
        query = first_non_empty(record, ["query"])
        if not query:
            continue
        if query_filter and query != query_filter:
            continue
        if not is_positive_record(record):
            continue

        item_id = safe_int(record.get("target_item_id", record.get("item_id")))
        if item_id <= 0:
            continue

        group = groups.setdefault(
            query,
            {
                "query": query,
                "items": {},
                "best_record": None,
                "best_record_key": None,
            },
        )

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

        candidate_best_key: Tuple[float, int, int, int] = (
            sample_signal,
            purchase,
            click,
            -item_id,
        )
        if group["best_record"] is None or candidate_best_key > group["best_record_key"]:
            group["best_record"] = record
            group["best_record_key"] = candidate_best_key

    return groups


def sort_evidence_items(item_entries: Iterable[Dict[str, object]], corpus_map: Dict[int, Dict[str, object]]) -> List[Dict[str, object]]:
    """Normalize and sort evidence items with the required stable ordering."""

    normalized: List[Dict[str, object]] = []
    for entry in item_entries:
        item_id = safe_int(entry.get("item_id"))
        item_meta = corpus_map.get(item_id, {"item_id": item_id})
        purchase_cnt = safe_int(entry.get("purchase_cnt"))
        click_cnt = safe_int(entry.get("click_cnt"))
        ranking_signal = round(safe_float(entry.get("ranking_signal")), 6)

        evidence_item = {
            **canonicalize_item(item_meta),
            "purchase_cnt": purchase_cnt,
            "click_cnt": click_cnt,
            "ranking_signal": ranking_signal,
            "relevance_signal": 2 if purchase_cnt > 0 else 1,
        }
        normalized.append(evidence_item)

    normalized.sort(
        key=lambda item: (
            -safe_float(item.get("ranking_signal")),
            -safe_int(item.get("purchase_cnt")),
            -safe_int(item.get("click_cnt")),
            safe_int(item.get("item_id")),
        )
    )
    return normalized


def build_query_record(
    query: str,
    group: Dict[str, object],
    corpus_map: Dict[int, Dict[str, object]],
    users_map: Optional[Dict[int, Dict[str, object]]] = None,
    top_k: int = 3,
) -> Dict[str, object]:
    """Convert one aggregated query group into the final JSONL training sample."""

    evidence_items = sort_evidence_items(group.get("items", {}).values(), corpus_map)[:top_k]
    target_item = dict(evidence_items[0]) if evidence_items else {}

    best_record = group.get("best_record") or {}
    user_id = safe_int(best_record.get("user_id"))
    session_id = safe_int(best_record.get("session_id"))
    user_profile = dict((users_map or {}).get(user_id, {}))
    recent_behavior_titles = extract_recent_behavior_titles(best_record, corpus_map)

    return {
        "query": query,
        "user_id": user_id,
        "session_id": session_id,
        "user_profile": user_profile,
        "recent_behavior_titles": recent_behavior_titles,
        "evidence_items": evidence_items,
        "target_item": target_item,
        "target_copy": render_template_copy(query=query, item=target_item, user_profile=user_profile),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Build first-version ad generation training pairs from KuaiSearch rank logs.",
        epilog=(
            "Example:\n"
            "  python ad_generation/build_train_pairs.py "
            "--corpus_path demo/items.jsonl --rank_path demo/rank.jsonl "
            "--users_path demo/users.jsonl --output_path ad_generation/data/train_pairs.jsonl --top_k 3"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--corpus_path", type=Path, default=Path("data/corpus.jsonl"), help="Path to corpus/items JSONL.")
    parser.add_argument("--rank_path", type=Path, default=Path("data/rank.jsonl"), help="Path to ranking log JSONL.")
    parser.add_argument("--users_path", type=Path, default=Path("data/users.jsonl"), help="Optional path to users JSONL.")
    parser.add_argument(
        "--output_path",
        type=Path,
        default=Path("ad_generation/data/train_pairs.jsonl"),
        help="Output JSONL path for training pairs.",
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of evidence items to keep per query.")
    return parser.parse_args()


def main() -> None:
    """Build and write `train_pairs.jsonl`."""

    args = parse_args()
    corpus_path = args.corpus_path
    rank_path = args.rank_path
    users_path = args.users_path if args.users_path and args.users_path.exists() else None
    output_path = args.output_path

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")
    if not rank_path.exists():
        raise FileNotFoundError(f"Rank file not found: {rank_path}")

    print(f"[INFO] Loading corpus from {corpus_path}")
    corpus_map = load_corpus_map(corpus_path)
    print(f"[INFO] Loaded {len(corpus_map)} items")

    users_map = load_user_map(users_path)
    if users_path:
        print(f"[INFO] Loaded {len(users_map)} users from {users_path}")
    else:
        print("[INFO] No users file found, continue without user profiles")

    print(f"[INFO] Aggregating positive samples from {rank_path}")
    query_groups = collect_query_groups(rank_path=rank_path, corpus_map=corpus_map, users_map=users_map)
    print(f"[INFO] Built groups for {len(query_groups)} queries")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for query in sorted(query_groups):
            record = build_query_record(
                query=query,
                group=query_groups[query],
                corpus_map=corpus_map,
                users_map=users_map,
                top_k=max(1, args.top_k),
            )
            if not record["evidence_items"]:
                continue
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[INFO] Wrote {written} training pairs to {output_path}")


if __name__ == "__main__":
    main()
