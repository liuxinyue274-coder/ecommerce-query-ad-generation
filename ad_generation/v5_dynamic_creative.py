"""V5 dynamic creative pipeline for LLM-backed ad generation.

This module adds a fuller generation framework on top of the existing
template / summary / fallback baseline without changing the old path.

Pipeline:
query normalization
-> intent_enricher
-> evidence_selector
-> user_profile_builder
-> style_retriever
-> prompt_builder
-> llm_provider
-> llm_output_parser
-> copy_validator
-> copy_ranker
-> copy_rewriter
-> final_ad_copy
"""

from __future__ import annotations

import json
import importlib
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from build_train_pairs import clean_text, first_non_empty, safe_float, safe_int
from infer import normalize_query, retrieve_evidence_bundle
from llm_generator import generate_with_llm, generate_with_llm_result, load_llm_config
from prompt_builder import (
    build_user_context,
    render_creative_template_copy,
    render_summary_copy,
    render_template_copy,
)
from v5_runtime import V5RuntimeContext, retrieve_evidence_bundle_cached


MODULE_SCHEMAS: Dict[str, Dict[str, object]] = {
    "query_normalization": {
        "input": {
            "raw_query": "str",
        },
        "output": {
            "raw_query": "str",
            "normalized_query": "str",
            "query_length": "int",
            "alpha_num_tokens": "list[str]",
            "han_tokens": "list[str]",
            "flags": "dict[str, bool]",
        },
    },
    "intent_enricher": {
        "input": {
            "query_bundle": "query_normalization.output",
            "retrieval_result": "infer.retrieve_evidence_bundle.result",
        },
        "output": {
            "query_text": "str",
            "category_hint": "str",
            "brand_hint": "str",
            "model_hint": "str",
            "scenario": "str",
            "audience": "str",
            "purchase_focus": "str",
            "scene_hint": "str",
            "crowd_hint": "str",
            "price_intent": "str",
            "attribute_hints": "list[str]",
            "search_focus": "list[str]",
            "intent_summary": "str",
        },
    },
    "evidence_selector": {
        "input": {
            "intent_bundle": "intent_enricher.output",
            "retrieval_result": "infer.retrieve_evidence_bundle.result",
            "top_k": "int",
        },
        "output": {
            "retrieval_source": "str",
            "anchor_item": "dict[str, object]",
            "selected_evidence_items": "list[dict[str, object]]",
            "common_selling_points": "list[str]",
            "fact_block": "list[str]",
            "selection_reason": "str",
        },
    },
    "user_profile_builder": {
        "input": {
            "user_id": "Optional[int]",
            "retrieval_result": "infer.retrieve_evidence_bundle.result",
            "users_map": "Optional[dict[int, dict[str, object]]]",
        },
        "output": {
            "user_id": "Optional[int]",
            "profile_available": "bool",
            "persona_summary": "str",
            "demographic_tags": "list[str]",
            "behavior_tags": "list[str]",
            "interest_tags": "list[str]",
            "recent_behavior_titles": "list[str]",
            "personalization_strength": "str",
            "user_profile_raw": "dict[str, object]",
        },
    },
    "style_retriever": {
        "input": {
            "query_bundle": "query_normalization.output",
            "intent_bundle": "intent_enricher.output",
            "user_profile_bundle": "user_profile_builder.output",
            "requested_tone": "str",
        },
        "output": {
            "style_id": "str",
            "tone": "str",
            "length_range": "str",
            "style_rules": "list[str]",
            "negative_rules": "list[str]",
            "style_examples": "list[str]",
            "example_pattern": "str",
        },
    },
    "prompt_builder": {
        "input": {
            "query_bundle": "query_normalization.output",
            "intent_bundle": "intent_enricher.output",
            "evidence_bundle": "evidence_selector.output",
            "user_profile_bundle": "user_profile_builder.output",
            "style_bundle": "style_retriever.output",
            "candidate_count": "int",
        },
        "output": {
            "system_prompt": "str",
            "user_prompt": "str",
            "full_prompt": "str",
            "output_format": "str",
            "candidate_count": "int",
            "prompt_sections": "dict[str, object]",
        },
    },
    "llm_provider": {
        "input": {
            "prompt_bundle": "prompt_builder.output",
            "llm_config_path": "Optional[path]",
            "candidate_count": "int",
        },
        "output": {
            "provider": "str",
            "model_name": "str",
            "status": "str",
            "requested_candidates": "int",
            "raw_generations": "list[dict[str, str]]",
            "fallback_used": "bool",
        },
    },
    "llm_output_parser": {
        "input": {
            "provider_bundle": "llm_provider.output",
        },
        "output": {
            "parse_status": "str",
            "parsed_candidates": "list[dict[str, object]]",
        },
    },
    "copy_validator": {
        "input": {
            "parsed_bundle": "llm_output_parser.output",
            "query_bundle": "query_normalization.output",
            "intent_bundle": "intent_enricher.output",
            "evidence_bundle": "evidence_selector.output",
            "style_bundle": "style_retriever.output",
        },
        "output": {
            "validated_candidates": "list[dict[str, object]]",
            "validator_summary": "dict[str, object]",
        },
    },
    "copy_ranker": {
        "input": {
            "validated_bundle": "copy_validator.output",
            "intent_bundle": "intent_enricher.output",
            "evidence_bundle": "evidence_selector.output",
        },
        "output": {
            "ranked_candidates": "list[dict[str, object]]",
            "top_candidate": "dict[str, object]",
        },
    },
    "copy_rewriter": {
        "input": {
            "ranked_bundle": "copy_ranker.output",
            "query_bundle": "query_normalization.output",
            "intent_bundle": "intent_enricher.output",
            "evidence_bundle": "evidence_selector.output",
            "user_profile_bundle": "user_profile_builder.output",
        },
        "output": {
            "rewritten": "bool",
            "rewrite_reason": "str",
            "final_candidate": "dict[str, object]",
        },
    },
    "final_ad_copy": {
        "input": {
            "rewriter_bundle": "copy_rewriter.output",
        },
        "output": {
            "final_ad_copy": "str",
            "final_source": "str",
            "final_candidate": "dict[str, object]",
        },
    },
}


BAD_PATTERNS = [
    "当前候选",
    "建议先从这类商品里继续筛选",
    "当前候选主要集中在",
    "信息清晰，适合先了解",
    "推荐先看",
]

BANNED_TOKENS = [
    "全网最低",
    "爆款必买",
    "疯抢",
    "百分百有效",
]

LOCAL_FAKE_BANNED_WORDS = [
    "最",
    "第一",
    "100%",
    "绝对",
    "永久",
    "根治",
    "必买",
    "神器",
    "闭眼入",
    "全网最低",
    "官方唯一",
]

LOCAL_FAKE_BAD_FRAGMENTS = [
    "让步少率",
    "先看出手",
    "更抗能用",
    "更爽能用",
    "帅气来",
    "这类目前更省心",
    "加上更上手",
    "少刚刚好",
]

STRATEGY_ORDER = [
    "场景型",
    "痛点型",
    "卖点型",
    "人群型",
    "轻网感型",
]

SPECIAL_QUERY_PROFILES = {
    "运动手环": {
        "scenario": "跑步通勤",
        "audience": "运动人群",
        "purchase_focus": "计步心率",
    },
    "宿舍吹风机": {
        "scenario": "宿舍洗头",
        "audience": "学生党",
        "purchase_focus": "小巧速干",
    },
    "通勤双肩包": {
        "scenario": "地铁通勤",
        "audience": "上班通学",
        "purchase_focus": "分层能装",
    },
    "敏感肌面霜": {
        "scenario": "换季维稳",
        "audience": "敏感肌",
        "purchase_focus": "保湿修护",
    },
    "零食礼包": {
        "scenario": "追剧分享",
        "audience": "囤零食人群",
        "purchase_focus": "多口味分享",
    },
    "手机壳": {
        "scenario": "日常通勤",
        "audience": "手机党",
        "purchase_focus": "防摔贴手",
    },
    "连衣裙": {
        "scenario": "通勤约会",
        "audience": "女生",
        "purchase_focus": "版型轻盈",
    },
}

SCENE_KEYWORDS = {
    "student_dorm": ["宿舍", "学生"],
    "commute": ["通勤", "上班", "出门"],
    "sports": ["运动", "跑步", "健身"],
    "gift": ["礼物", "送礼"],
    "anime_style": ["动漫", "二次元", "卡通"],
}

CROWD_KEYWORDS = {
    "female": ["女", "女生", "女士", "妈妈"],
    "male": ["男", "男生", "男士", "爸爸"],
    "kids": ["儿童", "小孩", "幼儿", "童"],
    "teen": ["学生", "青少年"],
}

NATURAL_SCENE_TERMS = ["跑步", "通勤", "宿舍", "早八", "换季", "追剧", "办公室", "日常", "出门", "上班"]
GENERIC_TAIL_PHRASES = ["更省心", "更实用", "都在线", "更适合"]


def _unique_preserve(items: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        clean = clean_text(item)
        if clean and clean not in seen:
            seen.add(clean)
            output.append(clean)
    return output


def _tokenize_text(text: str) -> List[str]:
    return re.findall(r"[A-Za-z]+[A-Za-z0-9\-]*|\d+[A-Za-z0-9\-]*|[\u4e00-\u9fff]{1,4}", clean_text(text))


def _short_title(title: str, limit: int = 18) -> str:
    clean = clean_text(title)
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip("，。；、 ") + "等"


def _contains_any(text: str, candidates: Iterable[str]) -> bool:
    haystack = clean_text(text)
    return any(candidate and candidate in haystack for candidate in candidates)


def _find_special_query_profile(query: str) -> Dict[str, str]:
    normalized = clean_text(query)
    for key, profile in SPECIAL_QUERY_PROFILES.items():
        if key in normalized:
            return dict(profile)
    return {}


def _extract_interest_tags(recent_behavior_titles: List[str]) -> List[str]:
    keyword_groups = [
        "运动",
        "动漫",
        "通勤",
        "收纳",
        "护肤",
        "面霜",
        "零食",
        "手机壳",
        "吹风机",
        "双肩包",
        "裙",
        "手环",
    ]
    tags: List[str] = []
    for title in recent_behavior_titles[:5]:
        for keyword in keyword_groups:
            if keyword in title:
                tags.append(keyword)
    return _unique_preserve(tags)


def _extract_common_selling_points(
    selected_evidence_items: List[Dict[str, object]],
    query: str,
    category_hint: str,
) -> List[str]:
    titles = [first_non_empty(item, ["item_title", "title"]) for item in selected_evidence_items]
    joined = " ".join(titles)
    keyword_pool = [
        "计步",
        "心率",
        "睡眠",
        "运动",
        "轻巧",
        "小巧",
        "速干",
        "折叠",
        "便携",
        "大容量",
        "分层",
        "收纳",
        "电脑",
        "通勤",
        "保湿",
        "修护",
        "舒缓",
        "面霜",
        "多口味",
        "独立包装",
        "分享",
        "防摔",
        "贴手",
        "液态硅胶",
        "适配",
        "机型",
        "垂感",
        "显瘦",
        "收腰",
        "轻盈",
        "连衣裙",
        "动漫",
        "卡通",
    ]
    selling_points = [keyword for keyword in keyword_pool if keyword in joined]

    special_profile = _find_special_query_profile(query)
    if special_profile:
        focus_tokens = _tokenize_text(special_profile.get("purchase_focus", ""))
        for token in focus_tokens:
            if token in joined or token in clean_text(category_hint):
                selling_points.append(token)

    if not selling_points:
        if "手环" in query or "手环" in category_hint:
            selling_points.extend(["计步", "运动"])
        elif "吹风机" in query:
            selling_points.extend(["小巧", "便携"])
        elif "双肩包" in query:
            selling_points.extend(["分层", "能装"])
        elif "面霜" in query:
            selling_points.extend(["保湿", "修护"])
        elif "零食" in query:
            selling_points.extend(["多口味", "分享"])
        elif "手机壳" in query:
            selling_points.extend(["防摔", "适配"])
        elif "连衣裙" in query:
            selling_points.extend(["轻盈", "版型"])

    return _unique_preserve(selling_points)[:4]


def _pick_query_term(query: str) -> str:
    special_profile = _find_special_query_profile(query)
    if special_profile:
        for key in SPECIAL_QUERY_PROFILES:
            if key in query:
                return key
    tokens = [token for token in _tokenize_text(query) if len(token) >= 2]
    return tokens[0] if tokens else clean_text(query)


def _sanitize_copy(text: str) -> str:
    copy = clean_text(text).replace("“ ", "“").replace(" ”", "”")
    for token in LOCAL_FAKE_BANNED_WORDS:
        copy = copy.replace(token, "")
    for fragment in LOCAL_FAKE_BAD_FRAGMENTS:
        copy = copy.replace(fragment, "")
    copy = copy.replace("这款商品适合", "")
    copy = copy.replace("优质好物值得拥有", "")
    copy = copy.replace("当前候选主要集中在", "")
    copy = copy.replace("建议先从", "")
    copy = copy.replace("适合使用", "更合适")
    copy = copy.replace("品质好物", "")
    copy = re.sub(r"\s+", "", copy)
    copy = re.sub(r"([一-龥]{2,4})\1+", r"\1", copy)
    copy = re.sub(r"(更顺手){2,}", "更顺手", copy)
    copy = re.sub(r"(更省心){2,}", "更省心", copy)
    copy = re.sub(r"(刚刚好){2,}", "刚刚好", copy)
    copy = re.sub(r"[、，]{2,}", "，", copy)
    copy = re.sub(r"[，。]{2,}", "。", copy)
    copy = copy.strip("，。；、 ")
    if len(copy) > 32:
        copy = copy[:32].rstrip("，。；、 ") + "。"
    if len(copy) < 12 and copy:
        copy = copy + "，日常更顺手。"
    if copy and not re.search(r"[。！？]$", copy):
        copy += "。"
    return copy


def _drop_redundant_tail(text: str) -> str:
    clean = _sanitize_copy(text).rstrip("。")
    for pair in [
        ("够用", "实用"),
        ("省心", "安心"),
        ("好用", "实用"),
    ]:
        if pair[0] in clean and pair[1] in clean:
            clean = clean.replace(f"更{pair[1]}", "")
            clean = clean.replace(pair[1], "")
    clean = re.sub(r"(，)?(更省心|更实用|都在线|更适合)$", "", clean)
    clean = re.sub(r"(，)?(更省心|更实用|都在线|更适合)(，)?(更省心|更实用|都在线|更适合)$", "", clean)
    clean = re.sub(r"，日常更顺手$", "", clean)
    clean = re.sub(r"，([^，。]{1,4})款，", r"，\1款", clean)
    clean = re.sub(r"[，、]{2,}", "，", clean).strip("，、 ")
    return _sanitize_copy(clean)


def _infer_local_fake_vertical(context: Dict[str, object]) -> str:
    query = clean_text(context.get("normalized_query") or context.get("query"))
    category_hint = clean_text((context.get("intent") or {}).get("category_hint"))
    title_text = " ".join(first_non_empty(item, ["item_title", "title"]) for item in context.get("selected_evidence") or [])
    haystack = f"{query} {category_hint} {title_text}"

    if _contains_any(haystack, ["手机壳", "壳", "硅胶壳", "配件壳"]):
        return "phone_case"
    if _contains_any(haystack, ["手环", "智能", "穿戴", "腕表", "手表"]):
        return "wearable"
    if _contains_any(haystack, ["吹风机", "宿舍", "小电器", "电吹风"]):
        return "dorm_appliance"
    if _contains_any(haystack, ["双肩包", "书包", "背包", "斜挎包", "托特包"]):
        return "bag"
    if _contains_any(haystack, ["连衣裙", "鞋", "外套", "裤", "服", "穿搭"]):
        return "apparel"
    if _contains_any(haystack, ["零食", "礼包", "饼干", "糖", "食品", "吃"]):
        return "food"
    if _contains_any(haystack, ["面霜", "敏感肌", "保湿", "修护", "护肤", "个护"]):
        return "health"
    return "misc"


def _local_fake_default_scene(vertical: str) -> str:
    scene_map = {
        "wearable": "跑步通勤",
        "phone_case": "日常出门",
        "dorm_appliance": "宿舍早八",
        "bag": "通勤出门",
        "apparel": "通勤出门",
        "food": "追剧囤货",
        "health": "换季护理",
        "misc": "日常出门",
    }
    return scene_map.get(vertical, "日常出门")


def _local_fake_default_audience(vertical: str) -> str:
    audience_map = {
        "wearable": "运动党",
        "phone_case": "手机党",
        "dorm_appliance": "学生党",
        "bag": "通勤人群",
        "apparel": "通勤人群",
        "food": "囤货党",
        "health": "敏感肌",
        "misc": "日常用户",
    }
    return audience_map.get(vertical, "日常用户")


def _local_fake_pick_points(context: Dict[str, object], vertical: str) -> List[str]:
    keyword_pool = [
        "心率",
        "睡眠",
        "运动记录",
        "健康监测",
        "计步",
        "小巧",
        "速干",
        "低功率",
        "折叠",
        "便携",
        "好收纳",
        "分层",
        "能装",
        "好搭",
        "舒适",
        "显瘦",
        "百搭",
        "小包装",
        "解馋",
        "方便",
        "组合装",
        "温和",
        "修护",
        "保湿",
        "舒缓",
        "防摔",
        "贴手",
        "液态硅胶",
        "适配",
        "轻盈",
        "版型",
    ]

    def _explode_point(text: str) -> List[str]:
        clean_point = clean_text(text)
        if not clean_point:
            return []
        exploded = [keyword for keyword in keyword_pool if keyword in clean_point]
        return exploded or [clean_point]

    selling_points: List[str] = []
    for point in list(context.get("evidence_summary") or []):
        selling_points.extend(_explode_point(str(point)))
    purchase_focus_tokens = _tokenize_text(clean_text((context.get("intent") or {}).get("purchase_focus")))
    for token in purchase_focus_tokens:
        if len(token) >= 2:
            selling_points.extend(_explode_point(token))
    fallback_map = {
        "wearable": ["心率", "睡眠", "运动记录", "健康监测"],
        "phone_case": ["防摔", "贴手", "液态硅胶", "适配"],
        "dorm_appliance": ["低功率", "小巧", "折叠", "好收纳"],
        "bag": ["分层", "能装", "好收纳", "通勤"],
        "apparel": ["好搭", "舒适", "显瘦", "百搭"],
        "food": ["小包装", "解馋", "方便", "组合装"],
        "health": ["温和", "修护", "保湿", "舒缓"],
        "misc": ["实用", "方便", "耐用", "好收纳"],
    }
    filtered: List[str] = []
    for point in selling_points:
        clean_point = clean_text(point)
        if len(clean_point) < 2:
            continue
        if clean_point in {"运动", "面霜", "连衣裙", "手机壳", "双肩包", "手环", "通勤"}:
            continue
        filtered.append(clean_point)
    if not filtered:
        filtered = list(fallback_map.get(vertical, fallback_map["misc"]))
    return _unique_preserve(filtered)[:4]


def _local_fake_preferred_points(vertical: str, points: List[str]) -> List[str]:
    preferred_map = {
        "wearable": ["心率", "睡眠", "运动记录", "健康监测", "计步"],
        "dorm_appliance": ["低功率", "小巧", "折叠", "便携", "好收纳", "速干"],
        "bag": ["分层", "能装", "好收纳", "轻便"],
        "health": ["温和", "修护", "保湿", "舒缓"],
        "food": ["小包装", "解馋", "方便", "组合装"],
        "phone_case": ["防摔", "贴手", "液态硅胶", "适配"],
        "apparel": ["好搭", "舒适", "显瘦", "百搭", "轻盈"],
        "misc": ["实用", "方便", "耐用", "好收纳"],
    }
    preferred = preferred_map.get(vertical, preferred_map["misc"])
    ordered = [point for point in preferred if point in points]
    if not ordered:
        ordered = preferred[:2]
    return _unique_preserve(ordered + points)[:4]


def _local_fake_personalized_tags(context: Dict[str, object], vertical: str, audience: str) -> List[str]:
    user_profile = context.get("user_profile") or {}
    interest_tags = list(user_profile.get("interest_tags") or [])
    demographic_tags = list(user_profile.get("demographic_tags") or [])
    personalized = _unique_preserve(interest_tags + demographic_tags + [audience])
    if not personalized:
        personalized = [_local_fake_default_audience(vertical)]
    return personalized[:3]


def _local_fake_trim(copy: str, vertical: str) -> str:
    trimmed = _sanitize_copy(copy)
    if len(trimmed) >= 12:
        return trimmed
    suffix_map = {
        "wearable": "，看数据更直观。",
        "phone_case": "，拿着更贴手。",
        "dorm_appliance": "，收着也不占地。",
        "bag": "，背着更利落。",
        "apparel": "，出门搭着不费劲。",
        "food": "，囤着吃更方便。",
        "health": "，日常用着更稳妥。",
        "misc": "，日常用着更顺手。",
    }
    return _sanitize_copy(trimmed.rstrip("。") + suffix_map.get(vertical, "，日常用着更顺手。"))


def _local_fake_join_points(primary_point: str, secondary_point: str) -> str:
    if secondary_point and secondary_point != primary_point:
        return f"{primary_point}{secondary_point}"
    return primary_point


def _local_fake_mentions_core_term(text: str, query_term: str, scenario: str, audience: str) -> bool:
    def _char_terms(value: str) -> List[str]:
        clean_value = clean_text(value)
        extra_terms: List[str] = []
        for size in (2, 3):
            for index in range(0, max(0, len(clean_value) - size + 1)):
                piece = clean_value[index : index + size]
                if re.search(r"[\u4e00-\u9fff]", piece):
                    extra_terms.append(piece)
        return extra_terms

    tokens = _unique_preserve(
        _tokenize_text(query_term)[:3]
        + _tokenize_text(scenario)[:3]
        + _tokenize_text(audience)[:3]
        + _char_terms(query_term)
        + _char_terms(scenario)
        + _char_terms(audience)
        + NATURAL_SCENE_TERMS
    )
    return any(token and token in text for token in tokens)


def _local_fake_generate_copy(
    strategy: str,
    vertical: str,
    query_term: str,
    scenario: str,
    audience: str,
    primary_point: str,
    secondary_point: str,
) -> str:
    dual_point = _local_fake_join_points(primary_point, secondary_point)
    templates = {
        "wearable": {
            "场景型": "跑步通勤都能戴，心率睡眠随手看",
            "痛点型": "运动别只靠感觉，数据戴手上更直观",
            "卖点型": "日常佩戴不累赘，运动睡眠都能记",
            "人群型": "想看睡眠和心率，手环帮你记清楚",
            "轻网感型": "通勤运动都适合，健康数据随手看",
        },
        "phone_case": {
            "场景型": "日常防摔先安排，硅胶手感更顺",
            "痛点型": "出门不怕小磕碰，防摔款更省心",
            "卖点型": "手机壳先看防护，手感舒服更常用",
            "人群型": "手滑党也能安心，软壳防护更贴手",
            "轻网感型": "日常通勤带着用，防摔手感都兼顾",
        },
        "dorm_appliance": {
            "场景型": "宿舍用刚刚好，吹完一折不占地",
            "痛点型": "早八前快速吹干，小巧款更好收",
            "卖点型": "小空间也好放，低功率用着更稳",
            "人群型": "学生宿舍常备，小巧好收不占桌",
            "轻网感型": "吹完随手一收，桌面不再乱糟糟",
        },
        "bag": {
            "场景型": "电脑水杯都装下，通勤一包就够",
            "痛点型": "上班路上少点乱，多隔层更好找",
            "卖点型": "通勤背着不累，电脑文件都能放",
            "人群型": "早高峰也好背，轻便容量更实用",
            "轻网感型": "日常出门不纠结，一个包就够用",
        },
        "apparel": {
            "场景型": f"{scenario}不想纠结，{dual_point}穿着就能出门",
            "痛点型": f"赶时间也不用换来换去，{primary_point}更好搭",
            "卖点型": f"{query_term}先看{primary_point}，{secondary_point}加上更耐穿",
            "人群型": f"{audience}穿{query_term}，{dual_point}更贴合日常",
            "轻网感型": f"{scenario}穿它就行，{dual_point}不容易出错",
        },
        "food": {
            "场景型": f"追剧嘴巴别闲着，{primary_point}囤着刚好",
            "痛点型": f"办公室抽屉备点，下午饿了有得吃",
            "卖点型": f"解馋不用纠结，{secondary_point or primary_point}更适合分享",
            "人群型": f"周末宅家追剧，零食礼包刚好安排",
            "轻网感型": f"小包装更方便，出门分享都顺手",
        },
        "health": {
            "场景型": "换季脸泛红？温和修护更稳妥",
            "痛点型": "日常护理别太猛，保湿修护慢慢来",
            "卖点型": "敏感肌也想保湿，温和配方更安心",
            "人群型": "干燥起皮别硬扛，先把保湿补上",
            "轻网感型": "换季护肤求稳，温和修护更适合",
        },
        "misc": {
            "场景型": f"日常用得上，{primary_point}更省心",
            "痛点型": f"出门带着方便，{primary_point}不累赘",
            "卖点型": f"家用刚刚好，{primary_point}更实在",
            "人群型": f"不想挑太久，先看{query_term}这一类",
            "轻网感型": f"场景合适更重要，{primary_point}更对路",
        },
    }
    return templates.get(vertical, templates["misc"]).get(strategy, f"{scenario}用{query_term}，{primary_point}更顺手")


def _build_local_fake_context(
    query_bundle: Dict[str, object],
    intent_bundle: Dict[str, object],
    evidence_bundle: Dict[str, object],
    user_profile_bundle: Dict[str, object],
    style_bundle: Dict[str, object],
) -> Dict[str, object]:
    return {
        "query": str(query_bundle.get("raw_query") or ""),
        "normalized_query": str(query_bundle.get("normalized_query") or ""),
        "intent": dict(intent_bundle),
        "selected_evidence": list(evidence_bundle.get("selected_evidence_items") or []),
        "user_profile": {
            "interest_tags": list(user_profile_bundle.get("interest_tags") or []),
            "demographic_tags": list(user_profile_bundle.get("demographic_tags") or []),
            "persona_summary": str(user_profile_bundle.get("persona_summary") or ""),
            "personalization_strength": str(user_profile_bundle.get("personalization_strength") or ""),
        },
        "retrieved_style_examples": list(style_bundle.get("style_examples") or []),
        "evidence_summary": list(evidence_bundle.get("common_selling_points") or []),
    }


def _local_fake_json_response(
    prompt: str,
    context: Dict[str, object],
    candidate_count: int,
) -> str:
    del prompt

    query = clean_text(context.get("normalized_query") or context.get("query"))
    query_term = _pick_query_term(query)
    intent = dict(context.get("intent") or {})
    vertical = _infer_local_fake_vertical(context)
    scenario = clean_text(intent.get("scenario")) or _local_fake_default_scene(vertical)
    audience = clean_text(intent.get("audience")) or _local_fake_default_audience(vertical)
    points = _local_fake_preferred_points(vertical, _local_fake_pick_points(context, vertical))
    primary_point = points[0]
    secondary_point = points[1] if len(points) > 1 else points[0]
    style_examples = list(context.get("retrieved_style_examples") or [])
    personalized_tags = _local_fake_personalized_tags(context, vertical, audience)
    style_hint = style_examples[0] if style_examples else ""

    strategy_order = STRATEGY_ORDER[: max(3, min(candidate_count, 5))]
    copies: List[Dict[str, object]] = []
    for index, strategy in enumerate(strategy_order):
        point_a = points[index % len(points)]
        point_b = points[(index + 1) % len(points)] if len(points) > 1 else point_a
        if strategy in {"场景型", "人群型", "轻网感型", "卖点型"}:
            point_a = primary_point
            point_b = secondary_point
        draft_copy = _local_fake_generate_copy(
            strategy=strategy,
            vertical=vertical,
            query_term=query_term,
            scenario=scenario,
            audience=audience,
            primary_point=point_a,
            secondary_point=point_b,
        )
        if not _local_fake_mentions_core_term(draft_copy, query_term, scenario, audience):
            draft_copy = f"{scenario}{draft_copy}"
        copy = _drop_redundant_tail(_local_fake_trim(draft_copy, vertical))
        if vertical == "misc" and not any(point and point in copy for point in [point_a, point_b]):
            copy = _drop_redundant_tail(_local_fake_trim(copy.rstrip("。") + f"，{point_a}更对路", vertical))
        used_evidence = _unique_preserve([point_a, point_b])[:2]
        rationale = f"结合{strategy}表达，落到{scenario}和{_local_fake_join_points(point_a, point_b)}。"
        if style_hint:
            rationale += f" 参考风格示例：{style_hint}"
        copies.append(
            {
                "copy": copy,
                "strategy": strategy,
                "used_evidence": used_evidence,
                "personalized_tags": personalized_tags,
                "rationale": rationale,
            }
        )

    return json.dumps({"copies": copies}, ensure_ascii=False)


def _extract_model_hint(text: str) -> str:
    query = clean_text(text)
    patterns = [
        r"[A-Za-z]+[A-Za-z0-9\-]*\d+[A-Za-z0-9\-]*",
        r"\d+[A-Za-z]+[A-Za-z0-9\-]*",
        r"[A-Za-z0-9\-]{3,}",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, query)
        if matches:
            for match in matches:
                if any(char.isdigit() for char in match):
                    return match
    return ""


def _extract_category_hint(items: List[Dict[str, object]]) -> str:
    leaves: List[str] = []
    for item in items:
        category_path = clean_text(item.get("category_path"))
        if category_path:
            leaves.append(category_path.split(">")[-1].strip())
    if not leaves:
        return ""
    return Counter(leaves).most_common(1)[0][0]


def _extract_brand_hint(query: str, items: List[Dict[str, object]]) -> str:
    query_lower = clean_text(query).lower()
    brands = []
    for item in items:
        brand = first_non_empty(item, ["brand_name", "brand"])
        if brand:
            brands.append(brand)
    for brand in _unique_preserve(brands):
        aliases = [part.strip() for part in re.split(r"[\\/|]", brand) if part.strip()]
        for alias in aliases:
            if alias.lower() in query_lower:
                return brand
    if brands:
        return Counter(brands).most_common(1)[0][0]
    return ""


def _detect_scene_hint(query: str) -> str:
    for scene_name, keywords in SCENE_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            return scene_name
    return ""


def _detect_crowd_hint(query: str) -> str:
    for crowd_name, keywords in CROWD_KEYWORDS.items():
        if any(keyword in query for keyword in keywords):
            return crowd_name
    return ""


def _normalize_provider_config(config: Dict[str, object]) -> Dict[str, object]:
    normalized = dict(config)
    api_block = normalized.get("api")
    if isinstance(api_block, dict):
        if not normalized.get("base_url") and api_block.get("base_url"):
            normalized["base_url"] = api_block.get("base_url")
        if not normalized.get("api_key_env") and api_block.get("api_key_env"):
            normalized["api_key_env"] = api_block.get("api_key_env")
    return normalized


def query_normalization_step(raw_query: str) -> Dict[str, object]:
    normalized_query = normalize_query(raw_query)
    tokens = _tokenize_text(normalized_query)
    alpha_num_tokens = [token for token in tokens if re.search(r"[A-Za-z0-9]", token)]
    han_tokens = [token for token in tokens if re.search(r"[\u4e00-\u9fff]", token)]
    flags = {
        "has_alpha_num_token": bool(alpha_num_tokens),
        "has_model_hint": bool(_extract_model_hint(normalized_query)),
        "has_scene_hint": bool(_detect_scene_hint(normalized_query)),
        "has_crowd_hint": bool(_detect_crowd_hint(normalized_query)),
    }
    return {
        "raw_query": clean_text(raw_query),
        "normalized_query": normalized_query,
        "query_length": len(normalized_query),
        "alpha_num_tokens": alpha_num_tokens,
        "han_tokens": han_tokens,
        "flags": flags,
    }


def intent_enricher_step(
    query_bundle: Dict[str, object],
    retrieval_result: Dict[str, object],
) -> Dict[str, object]:
    query = str(query_bundle.get("normalized_query") or "")
    evidence_items = list(retrieval_result.get("evidence_items") or [])
    brand_hint = _extract_brand_hint(query, evidence_items)
    category_hint = _extract_category_hint(evidence_items)
    model_hint = _extract_model_hint(query)
    scene_hint = _detect_scene_hint(query)
    crowd_hint = _detect_crowd_hint(query)
    price_intent = "budget" if any(token in query for token in ["平价", "便宜", "低价"]) else "unknown"
    special_profile = _find_special_query_profile(query)

    attribute_hints: List[str] = []
    for token in _tokenize_text(query):
        if len(token) >= 2 and token not in attribute_hints:
            if token not in [brand_hint, category_hint, model_hint]:
                attribute_hints.append(token)
    attribute_hints = attribute_hints[:6]

    search_focus = _unique_preserve([brand_hint, model_hint, category_hint, scene_hint, crowd_hint] + attribute_hints)[:8]
    summary_parts = [part for part in [brand_hint, model_hint, category_hint, scene_hint, crowd_hint] if part]
    intent_summary = " / ".join(summary_parts) if summary_parts else query

    return {
        "query_text": query,
        "category_hint": category_hint,
        "brand_hint": brand_hint,
        "model_hint": model_hint,
        "scenario": special_profile.get("scenario", "") or scene_hint,
        "audience": special_profile.get("audience", "") or crowd_hint,
        "purchase_focus": special_profile.get("purchase_focus", "") or " / ".join(attribute_hints[:2]),
        "scene_hint": scene_hint,
        "crowd_hint": crowd_hint,
        "price_intent": price_intent,
        "attribute_hints": attribute_hints,
        "search_focus": search_focus,
        "intent_summary": intent_summary,
    }


def evidence_selector_step(
    intent_bundle: Dict[str, object],
    retrieval_result: Dict[str, object],
    top_k: int = 3,
) -> Dict[str, object]:
    query_focus = set(str(value).lower() for value in intent_bundle.get("search_focus") or [] if value)
    evidence_items = list(retrieval_result.get("evidence_items") or [])
    category_hint = str(intent_bundle.get("category_hint") or "")
    brand_hint = str(intent_bundle.get("brand_hint") or "")

    rescored: List[Tuple[float, Dict[str, object]]] = []
    for item in evidence_items:
        title = first_non_empty(item, ["item_title", "title"])
        brand = first_non_empty(item, ["brand_name", "brand"])
        category = clean_text(item.get("category_path"))
        score = safe_float(item.get("ranking_signal"))
        haystack = f"{title} {brand} {category}".lower()
        overlap = sum(1 for token in query_focus if token and token in haystack)
        if brand_hint and brand_hint.lower() in haystack:
            score += 2.0
        if category_hint and category_hint.lower() in haystack:
            score += 1.0
        score += overlap
        rescored.append((score, item))

    rescored.sort(key=lambda entry: (-entry[0], -safe_float(entry[1].get("ranking_signal")), safe_int(entry[1].get("item_id"))))
    selected_evidence_items = [dict(item) for _, item in rescored[: max(1, top_k)]]
    anchor_item = dict(selected_evidence_items[0]) if selected_evidence_items else dict(retrieval_result.get("target_item") or {})
    common_selling_points = _extract_common_selling_points(
        selected_evidence_items=selected_evidence_items,
        query=str(intent_bundle.get("query_text") or intent_bundle.get("intent_summary") or ""),
        category_hint=category_hint,
    )

    fact_block: List[str] = []
    for item in selected_evidence_items:
        title = first_non_empty(item, ["item_title", "title"])
        brand = first_non_empty(item, ["brand_name", "brand"])
        category_path = clean_text(item.get("category_path"))
        leaf_category = category_path.split(">")[-1].strip() if category_path else ""
        if title:
            fact_block.append(f"title={_short_title(title)}")
        if brand:
            fact_block.append(f"brand={brand}")
        if leaf_category:
            fact_block.append(f"category={leaf_category}")
    fact_block = _unique_preserve(fact_block)[:8]

    selection_reason = "rescored_by_intent_focus"
    return {
        "retrieval_source": str(retrieval_result.get("source") or "unknown"),
        "anchor_item": anchor_item,
        "selected_evidence_items": selected_evidence_items,
        "common_selling_points": common_selling_points,
        "fact_block": fact_block,
        "selection_reason": selection_reason,
    }


def user_profile_builder_step(
    user_id: Optional[int],
    retrieval_result: Dict[str, object],
    users_map: Optional[Dict[int, Dict[str, object]]] = None,
) -> Dict[str, object]:
    user_profile_raw = dict(retrieval_result.get("user_profile") or {})
    if not user_profile_raw and user_id is not None and users_map:
        user_profile_raw = dict(users_map.get(user_id, {}))

    recent_behavior_titles = list(retrieval_result.get("recent_behavior_titles") or [])
    demographic_tags: List[str] = []
    behavior_tags: List[str] = []

    gender = clean_text(user_profile_raw.get("gender"))
    age = clean_text(user_profile_raw.get("age_bucket") or user_profile_raw.get("age"))
    city = clean_text(user_profile_raw.get("city"))

    if gender:
        demographic_tags.append(f"gender={gender}")
    if age:
        demographic_tags.append(f"age={age}")
    if city:
        demographic_tags.append(f"city={city}")

    for title in recent_behavior_titles[:5]:
        tokens = _tokenize_text(title)
        if tokens:
            behavior_tags.append(tokens[0])
    behavior_tags = _unique_preserve(behavior_tags)
    interest_tags = _extract_interest_tags(recent_behavior_titles)

    persona_summary = build_user_context(
        user_profile=user_profile_raw,
        recent_behavior_titles=recent_behavior_titles,
    )
    if user_profile_raw and recent_behavior_titles:
        personalization_strength = "medium"
    elif user_profile_raw or recent_behavior_titles:
        personalization_strength = "weak"
    else:
        personalization_strength = "none"

    return {
        "user_id": user_id,
        "profile_available": bool(user_profile_raw or recent_behavior_titles),
        "persona_summary": persona_summary,
        "demographic_tags": demographic_tags,
        "behavior_tags": behavior_tags,
        "interest_tags": interest_tags,
        "recent_behavior_titles": recent_behavior_titles,
        "personalization_strength": personalization_strength,
        "user_profile_raw": user_profile_raw,
    }


def style_retriever_step(
    query_bundle: Dict[str, object],
    intent_bundle: Dict[str, object],
    user_profile_bundle: Dict[str, object],
    requested_tone: str = "creative",
) -> Dict[str, object]:
    query_length = safe_int(query_bundle.get("query_length"))
    scene_hint = str(intent_bundle.get("scene_hint") or "")
    category_hint = str(intent_bundle.get("category_hint") or "")

    tone = requested_tone
    if requested_tone == "creative" and query_length <= 6 and not scene_hint:
        style_id = "creative_compact"
        length_range = "20-32字"
    elif scene_hint:
        style_id = "scene_led"
        length_range = "24-42字"
    elif category_hint:
        style_id = "fact_led"
        length_range = "22-38字"
    else:
        style_id = "balanced_generic"
        length_range = "22-40字"

    style_rules = [
        "先点出需求场景或选择方向，再自然带出商品事实。",
        "不要照抄完整商品标题。",
        "如果证据不足，就写稳妥推荐，不硬造卖点。",
    ]
    if user_profile_bundle.get("personalization_strength") in {"weak", "medium"}:
        style_rules.append("可以轻度借用用户最近行为，但不能压过当前 query。")

    negative_rules = [
        "不要写价格、优惠、销量。",
        "不要写全网最低、爆款必买。",
        "不要写搜索摘要口吻。",
    ]

    if style_id == "scene_led":
        example_pattern = "场景开头 + 品类方向 + 自然引导"
    elif style_id == "creative_compact":
        example_pattern = "需求点 + 适配方向"
    else:
        example_pattern = "选择理由 + 商品方向"
    style_examples = [
        "晨跑想看运动手环，计步心率一眼更省事。",
        "宿舍吹风机别太占地，小巧速干拿着更顺手。",
        "通勤双肩包装电脑和杂物，分层收纳背着更利落。",
        "敏感肌面霜换季脸干时，保湿修护这类更安心。",
        "手机壳别只看花样，防摔贴手的款日常更耐用。",
    ]

    return {
        "style_id": style_id,
        "tone": tone,
        "length_range": length_range,
        "style_rules": style_rules,
        "negative_rules": negative_rules,
        "style_examples": style_examples,
        "example_pattern": example_pattern,
    }


def prompt_builder_step(
    query_bundle: Dict[str, object],
    intent_bundle: Dict[str, object],
    evidence_bundle: Dict[str, object],
    user_profile_bundle: Dict[str, object],
    style_bundle: Dict[str, object],
    candidate_count: int = 3,
) -> Dict[str, object]:
    normalized_query = str(query_bundle.get("normalized_query") or "")
    fact_block = list(evidence_bundle.get("fact_block") or [])
    prompt_sections = {
        "query": normalized_query,
        "intent_summary": str(intent_bundle.get("intent_summary") or ""),
        "intent": {
            "scenario": str(intent_bundle.get("scenario") or ""),
            "audience": str(intent_bundle.get("audience") or ""),
            "purchase_focus": str(intent_bundle.get("purchase_focus") or ""),
        },
        "style_id": str(style_bundle.get("style_id") or ""),
        "style_examples": list(style_bundle.get("style_examples") or []),
        "facts": fact_block,
        "common_selling_points": list(evidence_bundle.get("common_selling_points") or []),
        "interest_tags": list(user_profile_bundle.get("interest_tags") or []),
        "persona_summary": str(user_profile_bundle.get("persona_summary") or ""),
    }

    system_prompt = "\n".join(
        [
            "你是电商搜索广告文案生成助手。",
            "你的任务是根据 query 意图、商品证据和弱个性化信息，生成真实、自然、可点击的一行中文广告文案。",
            "不要编造价格、折扣、销量、功效、认证、官方身份。",
            "不要写成搜索结果摘要或商品标题复制。",
        ]
    )

    user_lines = [
        f"Query: {normalized_query}",
        f"Intent Summary: {intent_bundle.get('intent_summary') or normalized_query}",
        f"Scene Hint: {intent_bundle.get('scene_hint') or 'none'}",
        f"Crowd Hint: {intent_bundle.get('crowd_hint') or 'none'}",
        f"Style ID: {style_bundle.get('style_id') or 'balanced_generic'}",
        f"Length Range: {style_bundle.get('length_range') or '22-40字'}",
        "",
        "Evidence Facts:",
    ]
    for index, fact in enumerate(fact_block, start=1):
        user_lines.append(f"{index}. {fact}")

    if user_profile_bundle.get("persona_summary"):
        user_lines.extend(["", "Weak Personalization:", str(user_profile_bundle.get("persona_summary") or "")])

    user_lines.extend(
        [
            "",
            "Generation Rules:",
            "- 只输出中文广告文案本身。",
            "- 候选之间要有表达差异，但都要围绕同一 query 意图。",
            "- 优先写适合谁、适合什么场景、为什么值得先看。",
            "- 如果证据主要是类目和适配信息，就写稳妥推荐。",
            "",
            f"Please produce {candidate_count} candidates, one per line, using this format:",
            "候选1：...",
            "候选2：...",
            "候选3：...",
        ]
    )

    user_prompt = "\n".join(user_lines)
    full_prompt = system_prompt + "\n\n" + user_prompt
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "full_prompt": full_prompt,
        "output_format": "one-line-candidates",
        "candidate_count": max(1, candidate_count),
        "prompt_sections": prompt_sections,
    }


def _synthesized_candidates(
    query_bundle: Dict[str, object],
    intent_bundle: Dict[str, object],
    evidence_bundle: Dict[str, object],
    user_profile_bundle: Dict[str, object],
    limit: int,
) -> List[str]:
    query = str(query_bundle.get("normalized_query") or "")
    selected_evidence_items = list(evidence_bundle.get("selected_evidence_items") or [])
    anchor_item = dict(evidence_bundle.get("anchor_item") or {})
    user_profile_raw = dict(user_profile_bundle.get("user_profile_raw") or {})
    category_hint = str(intent_bundle.get("category_hint") or "")
    scene_hint = str(intent_bundle.get("scene_hint") or "")
    crowd_hint = str(intent_bundle.get("crowd_hint") or "")

    summary_copy = render_summary_copy(query=query, evidence_items=selected_evidence_items, user_profile=user_profile_raw)["copy"]
    creative_copy = render_creative_template_copy(
        query=query,
        evidence_items=selected_evidence_items,
        item=anchor_item,
        user_profile=user_profile_raw,
    )
    template_copy = render_template_copy(query=query, item=anchor_item, user_profile=user_profile_raw)

    category_piece = category_hint or first_non_empty(anchor_item, ["category_path"]).split(">")[-1].strip()
    crowd_piece = crowd_hint.replace("_", " ") if crowd_hint else ""
    scene_piece = scene_hint.replace("_", " ") if scene_hint else ""
    anchor_title = _short_title(first_non_empty(anchor_item, ["item_title", "title"]))

    extras = [
        f"想找“{query}”？可以先看看{category_piece or anchor_title}这类方向，更容易挑到贴合需求的一款。",
        f"围绕“{query}”这类需求，不妨优先留意{anchor_title or category_piece}，表达更贴近当前搜索意图。",
        (
            f"{crowd_piece}如果在找“{query}”，可以先从{scene_piece or category_piece or anchor_title}相关款里挑，"
            "更容易找到顺手又合适的选择。"
        ).strip(),
    ]
    candidates = _unique_preserve([creative_copy, summary_copy, template_copy] + extras)
    return candidates[: max(1, limit)]


def llm_provider_step(
    prompt_bundle: Dict[str, object],
    query_bundle: Dict[str, object],
    intent_bundle: Dict[str, object],
    evidence_bundle: Dict[str, object],
    user_profile_bundle: Dict[str, object],
    style_bundle: Dict[str, object],
    llm_config_path: Optional[Path],
    candidate_count: int = 3,
    provider_override: Optional[str] = None,
) -> Dict[str, object]:
    if llm_config_path is not None and Path(llm_config_path).exists():
        config = _normalize_provider_config(load_llm_config(Path(llm_config_path)))
    else:
        config = {
            "provider": "local_fake",
            "model_name": "local-fake-dynamic-creative",
            "temperature": 0.7,
            "max_new_tokens": 128,
            "timeout": 30,
        }

    provider = clean_text(provider_override) or clean_text(config.get("provider")) or "local_fake"
    config["provider"] = provider
    model_name = clean_text(config.get("model_name")) or "unknown"
    variant_notes = [
        "版本1：偏场景表达。",
        "版本2：偏商品事实表达。",
        "版本3：偏简洁转化表达。",
        "版本4：偏轻个性化表达。",
        "版本5：偏自然推荐表达。",
    ]

    raw_generations: List[Dict[str, str]] = []
    fallback_used = False
    statuses: List[str] = []
    local_fake_context = _build_local_fake_context(
        query_bundle=query_bundle,
        intent_bundle=intent_bundle,
        evidence_bundle=evidence_bundle,
        user_profile_bundle=user_profile_bundle,
        style_bundle=style_bundle,
    )

    def _build_local_fake_bundle(status: str, requested_provider: str, requested_model: str, reason: str = "") -> Dict[str, object]:
        candidate_total = max(3, min(candidate_count, 5))
        local_raw_generations = [
            {
                "candidate_id": "batch_1",
                "raw_text": _local_fake_json_response(
                    prompt=str(prompt_bundle.get("full_prompt") or ""),
                    context=local_fake_context,
                    candidate_count=candidate_total,
                ),
            }
        ]
        return {
            "provider": "local_fake",
            "model_name": "local-fake-dynamic-creative",
            "status": status,
            "requested_provider": requested_provider,
            "requested_model": requested_model,
            "requested_candidates": candidate_total,
            "raw_generations": local_raw_generations,
            "fallback_used": True,
            "local_fake_context": local_fake_context,
            "fallback_reason": reason,
            "api_results": [],
        }

    if provider in {"mock", "local_fake"}:
        bundle = _build_local_fake_bundle(
            status="local_fake_json_ok",
            requested_provider=provider,
            requested_model=model_name or "local-fake-dynamic-creative",
        )
        bundle["provider"] = provider
        bundle["model_name"] = model_name or "local-fake-dynamic-creative"
        return bundle

    if provider == "deepseek_chat" and not clean_text(os.environ.get("DEEPSEEK_API_KEY")):
        return _build_local_fake_bundle(
            status="skipped_no_api_key",
            requested_provider="deepseek_chat",
            requested_model=clean_text(os.environ.get("DEEPSEEK_MODEL")) or model_name or "deepseek-chat",
            reason="missing_DEEPSEEK_API_KEY",
        )
    if provider == "sft_local" and not clean_text(os.environ.get("SFT_MODEL_PATH")):
        return _build_local_fake_bundle(
            status="skipped_no_sft_model",
            requested_provider="sft_local",
            requested_model="",
            reason="missing_SFT_MODEL_PATH",
        )
    if provider == "sft_local":
        try:
            importlib.import_module("transformers")
            importlib.import_module("torch")
        except Exception:
            return _build_local_fake_bundle(
                status="skipped_missing_dependencies",
                requested_provider="sft_local",
                requested_model=clean_text(os.environ.get("SFT_MODEL_PATH")),
                reason="missing_transformers_or_torch",
            )

    api_results: List[Dict[str, object]] = []
    for index in range(max(1, candidate_count)):
        prompt = str(prompt_bundle.get("full_prompt") or "")
        if provider in {"deepseek_chat", "sft_local"}:
            prompt = (
                prompt
                + "\n\n"
                + variant_notes[index % len(variant_notes)]
                + "\n只输出1条中文广告文案，不要解释，不要JSON，不要编号。"
            )
            api_result = generate_with_llm_result(prompt, config)
            api_results.append(api_result)
            candidate_text = clean_text(api_result.get("text"))
            if api_result.get("ok"):
                statuses.append("ok")
            else:
                failure_status = clean_text(api_result.get("status")) or "failed"
                statuses.append(f"{failure_status}:{clean_text(api_result.get('error'))}")
                fallback_used = True
        else:
            prompt = prompt + "\n\n" + variant_notes[index % len(variant_notes)]
            try:
                candidate_text = generate_with_llm(prompt, config)
                statuses.append("ok")
            except Exception as exc:
                candidate_text = ""
                statuses.append(f"failed:{exc}")
                fallback_used = True
        raw_generations.append(
            {
                "candidate_id": f"candidate_{index + 1}",
                "raw_text": candidate_text,
            }
        )

    if all(not item["raw_text"] for item in raw_generations):
        fallback_used = True
        synthetic = _synthesized_candidates(
            query_bundle=query_bundle,
            intent_bundle=intent_bundle,
            evidence_bundle=evidence_bundle,
            user_profile_bundle=user_profile_bundle,
            limit=max(3, candidate_count),
        )
        raw_generations = [
            {
                "candidate_id": f"candidate_{index + 1}",
                "raw_text": text,
            }
            for index, text in enumerate(synthetic[: max(1, candidate_count)])
        ]
        status = "synthetic_candidates"
    else:
        if len({item["raw_text"] for item in raw_generations if item["raw_text"]}) <= 1:
            fallback_used = True
            synthetic = _synthesized_candidates(
                query_bundle=query_bundle,
                intent_bundle=intent_bundle,
                evidence_bundle=evidence_bundle,
                user_profile_bundle=user_profile_bundle,
                limit=max(3, candidate_count),
            )
            for index, text in enumerate(synthetic):
                if index >= len(raw_generations):
                    break
                if not raw_generations[index]["raw_text"] or raw_generations[index]["raw_text"] == raw_generations[0]["raw_text"]:
                    raw_generations[index]["raw_text"] = text
        status = "provider_ok" if all(item == "ok" for item in statuses) else "provider_partial"

    return {
        "provider": provider,
        "model_name": model_name,
        "status": status,
        "requested_provider": provider,
        "requested_model": model_name,
        "requested_candidates": max(1, candidate_count),
        "raw_generations": raw_generations,
        "fallback_used": fallback_used,
        "fallback_reason": "",
        "api_results": api_results,
    }


def llm_output_parser_step(provider_bundle: Dict[str, object]) -> Dict[str, object]:
    parsed_candidates: List[Dict[str, object]] = []
    for record in provider_bundle.get("raw_generations") or []:
        candidate_id = str(record.get("candidate_id") or "")
        raw_text = clean_text(record.get("raw_text"))
        if not raw_text:
            continue
        try:
            payload = json.loads(raw_text)
            if isinstance(payload, dict) and isinstance(payload.get("copies"), list):
                for index, item in enumerate(payload["copies"], start=1):
                    if not isinstance(item, dict):
                        continue
                    parsed_candidates.append(
                        {
                            "candidate_id": f"{candidate_id}_{index}",
                            "text": clean_text(item.get("copy")),
                            "raw_text": raw_text,
                            "source": "llm_provider_json",
                            "strategy": clean_text(item.get("strategy")),
                            "used_evidence": list(item.get("used_evidence") or []),
                            "personalized_tags": list(item.get("personalized_tags") or []),
                            "rationale": clean_text(item.get("rationale")),
                        }
                    )
                continue
        except json.JSONDecodeError:
            pass
        lines = [clean_text(line) for line in raw_text.splitlines() if clean_text(line)]
        extracted = False
        for line in lines:
            normalized_line = re.sub(r"^候选\d+[：:]\s*", "", line)
            if normalized_line:
                parsed_candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "text": normalized_line,
                        "raw_text": raw_text,
                        "source": "llm_provider",
                    }
                )
                extracted = True
                break
        if not extracted:
            parsed_candidates.append(
                {
                    "candidate_id": candidate_id,
                    "text": raw_text,
                    "raw_text": raw_text,
                    "source": "llm_provider",
                }
            )
    return {
        "parse_status": "ok" if parsed_candidates else "empty",
        "parsed_candidates": parsed_candidates,
    }


def _clean_candidate_text(text: str) -> str:
    clean = clean_text(text)
    clean = re.sub(r"^候选\d+[：:]\s*", "", clean)
    clean = re.sub(r"\s+", "", clean)
    clean = clean.strip("：:;；")
    return clean


def copy_validator_step(
    parsed_bundle: Dict[str, object],
    query_bundle: Dict[str, object],
    intent_bundle: Dict[str, object],
    evidence_bundle: Dict[str, object],
    style_bundle: Dict[str, object],
) -> Dict[str, object]:
    del style_bundle

    normalized_query = str(query_bundle.get("normalized_query") or "")
    brand_hint = str(intent_bundle.get("brand_hint") or "")
    category_hint = str(intent_bundle.get("category_hint") or "")
    anchor_title = first_non_empty(evidence_bundle.get("anchor_item") or {}, ["item_title", "title"])

    validated_candidates: List[Dict[str, object]] = []
    for record in parsed_bundle.get("parsed_candidates") or []:
        text = _clean_candidate_text(str(record.get("text") or ""))
        issues: List[str] = []
        if not text:
            issues.append("empty")
        if len(text) < 12:
            issues.append("too_short")
        if len(text) > 60:
            issues.append("too_long")
        if any(pattern in text for pattern in BAD_PATTERNS):
            issues.append("search_summary_tone")
        for token in BANNED_TOKENS:
            if token in text:
                issues.append(f"banned_token:{token}")
        if "价格" in text or "优惠" in text or "销量" in text:
            issues.append("unsupported_claim")

        relevance_hits = 0
        for token in _unique_preserve([normalized_query, brand_hint, category_hint, anchor_title]):
            if token and token in text:
                relevance_hits += 1

        validated_candidates.append(
            {
                "candidate_id": str(record.get("candidate_id") or ""),
                "text": text,
                "strategy": record.get("strategy"),
                "used_evidence": list(record.get("used_evidence") or []),
                "personalized_tags": list(record.get("personalized_tags") or []),
                "rationale": record.get("rationale") or "",
                "issues": issues,
                "is_valid": not issues,
                "length": len(text),
                "relevance_hits": relevance_hits,
            }
        )

    validator_summary = {
        "candidate_count": len(validated_candidates),
        "valid_count": sum(1 for item in validated_candidates if item["is_valid"]),
    }
    return {
        "validated_candidates": validated_candidates,
        "validator_summary": validator_summary,
    }


def copy_ranker_step(
    validated_bundle: Dict[str, object],
    intent_bundle: Dict[str, object],
    evidence_bundle: Dict[str, object],
) -> Dict[str, object]:
    selling_points = list(evidence_bundle.get("common_selling_points") or [])
    scene_hint = str(intent_bundle.get("scene_hint") or "")
    crowd_hint = str(intent_bundle.get("crowd_hint") or "")
    ranked_candidates: List[Dict[str, object]] = []
    for record in validated_bundle.get("validated_candidates") or []:
        text = str(record.get("text") or "")
        issues = list(record.get("issues") or [])
        relevance_score = safe_int(record.get("relevance_hits")) * 8
        validity_score = 48 if not issues else max(0, 16 - 8 * len(issues))

        scenario_hits = sum(1 for token in NATURAL_SCENE_TERMS if token in text)
        evidence_hits = sum(1 for token in selling_points if token and token in text)
        personalization_visible = sum(1 for tag in record.get("personalized_tags") or [] if clean_text(tag) and clean_text(tag).split("=")[-1] in text)
        comma_count = text.count("，") + text.count("、")
        title_like_score = 0
        if not re.search(r"[，。！？]", text):
            title_like_score += 4
        if len(re.findall(r"[\u4e00-\u9fff]{2,4}", text)) >= 6 and scenario_hits == 0:
            title_like_score += 4
        if any(token in text for token in ["适配", "promax", "iphone", "官方", "型号", "套装"]) and scenario_hits == 0:
            title_like_score += 3

        template_penalty = 0
        if "这类" in text:
            template_penalty += 4
        if "更省心" in text and scenario_hits == 0:
            template_penalty += 3
        if any(token in text for token in ["适合使用", "品质好物", "当前候选", "建议先从"]):
            template_penalty += 6
        if comma_count > 2:
            template_penalty += (comma_count - 2) * 3
        if re.search(r"，[^，。]{1,4}款，", text):
            template_penalty += 4
        if any(pair[0] in text and pair[1] in text for pair in [("够用", "实用"), ("省心", "安心"), ("好用", "实用")]):
            template_penalty += 5
        template_penalty += sum(2 for token in GENERIC_TAIL_PHRASES if token in text)
        if "更顺手" in text:
            template_penalty += 2

        naturalness_score = 18
        if "？" in text:
            naturalness_score += 3
        if scenario_hits:
            naturalness_score += min(10, scenario_hits * 4)
        if scene_hint and scene_hint.replace("_", "")[:2] in text:
            naturalness_score += 3
        if crowd_hint and crowd_hint.replace("_", "")[:2] in text:
            naturalness_score += 2
        naturalness_score += min(6, evidence_hits * 3)
        naturalness_score += min(4, personalization_visible * 2)
        naturalness_score -= title_like_score
        naturalness_score -= template_penalty

        brevity_score = 16 if 12 <= len(text) <= 20 else 14 if 21 <= len(text) <= 26 else 8 if 27 <= len(text) <= 30 else 0
        rank_score = validity_score + relevance_score + naturalness_score + brevity_score
        ranked_candidates.append(
            {
                **record,
                "rank_score": rank_score,
                "subscores": {
                    "validity": validity_score,
                    "relevance": relevance_score,
                    "naturalness": naturalness_score,
                    "brevity": brevity_score,
                    "scenario_hits": scenario_hits,
                    "evidence_hits": evidence_hits,
                    "title_like_penalty": title_like_score,
                    "template_penalty": template_penalty,
                },
            }
        )

    ranked_candidates.sort(key=lambda item: (-safe_float(item.get("rank_score")), len(item.get("issues") or []), item.get("candidate_id", "")))
    top_candidate = ranked_candidates[0] if ranked_candidates else {}
    return {
        "ranked_candidates": ranked_candidates,
        "top_candidate": top_candidate,
    }


def _rule_rewrite(text: str) -> str:
    rewritten = clean_text(text)
    for pattern in BAD_PATTERNS:
        rewritten = rewritten.replace(pattern, "")
    rewritten = rewritten.replace("，，", "，")
    rewritten = re.sub(r"[，。]{2,}", "。", rewritten)
    rewritten = rewritten.strip("，。； ")
    if len(rewritten) > 48:
        rewritten = rewritten[:48].rstrip("，。；、 ") + "。"
    if rewritten and not re.search(r"[。！？]$", rewritten):
        rewritten += "。"
    return _drop_redundant_tail(rewritten)


def _compress_long_copy(text: str) -> str:
    base = _drop_redundant_tail(text).rstrip("。")
    if len(base) <= 28:
        return _sanitize_copy(base)

    parts = [part.strip() for part in re.split(r"[，、]", base) if part.strip()]
    if not parts:
        return _sanitize_copy(base)

    scored_parts: List[Tuple[int, str]] = []
    for part in parts:
        score = 0
        if any(token in part for token in NATURAL_SCENE_TERMS):
            score += 4
        if any(token in part for token in ["心率", "睡眠", "保湿", "修护", "防摔", "手感", "小巧", "低功率", "分层", "能装"]):
            score += 3
        if "更" in part:
            score -= 1
        scored_parts.append((score, part))
    scored_parts.sort(key=lambda item: (-item[0], len(item[1])))

    selected: List[str] = []
    for _, part in scored_parts:
        if len(selected) >= 2:
            break
        selected.append(part)
    if not selected:
        selected = parts[:2]

    compressed = "，".join(selected[:2])
    compressed = re.sub(r"(一个包)够用", r"\1就够用", compressed)
    compressed = re.sub(r"(防摔)贴手都在线", r"\1手感都兼顾", compressed)
    compressed = _drop_redundant_tail(compressed)
    if len(compressed.rstrip("。")) > 26 and "，" in compressed:
        compressed = compressed.split("，", 1)[0]
    return _sanitize_copy(compressed)


def copy_rewriter_step(
    ranked_bundle: Dict[str, object],
    query_bundle: Dict[str, object],
    intent_bundle: Dict[str, object],
    evidence_bundle: Dict[str, object],
    user_profile_bundle: Dict[str, object],
) -> Dict[str, object]:
    top_candidate = dict(ranked_bundle.get("top_candidate") or {})
    selected_evidence_items = list(evidence_bundle.get("selected_evidence_items") or [])
    anchor_item = dict(evidence_bundle.get("anchor_item") or {})
    user_profile_raw = dict(user_profile_bundle.get("user_profile_raw") or {})
    query = str(query_bundle.get("normalized_query") or "")

    if not top_candidate:
        summary_result = render_summary_copy(query=query, evidence_items=selected_evidence_items, user_profile=user_profile_raw)
        return {
            "rewritten": False,
            "rewrite_reason": "empty_ranker_result_then_summary_fallback",
            "final_candidate": {
                "text": summary_result["copy"],
                "source": "summary_fallback",
                "rank_score": 0,
            },
        }

    issues = list(top_candidate.get("issues") or [])
    rewritten_text = _rule_rewrite(str(top_candidate.get("text") or ""))
    if len(rewritten_text.rstrip("。")) > 28:
        rewritten_text = _compress_long_copy(rewritten_text)

    if not issues and rewritten_text:
        final_text = rewritten_text
        rewrite_reason = "top_candidate_kept"
        rewritten = final_text != str(top_candidate.get("text") or "")
        source = "llm_ranker_top1"
    else:
        summary_result = render_summary_copy(query=query, evidence_items=selected_evidence_items, user_profile=user_profile_raw)
        creative_fallback = render_creative_template_copy(
            query=query,
            evidence_items=selected_evidence_items,
            item=anchor_item,
            user_profile=user_profile_raw,
        )
        template_fallback = render_template_copy(query=query, item=anchor_item, user_profile=user_profile_raw)

        if rewritten_text and len(rewritten_text) >= 12 and "unsupported_claim" not in issues:
            final_text = rewritten_text
            rewrite_reason = "rule_rewrite_from_invalid_candidate"
            rewritten = True
            source = "llm_rule_rewrite"
        elif clean_text(summary_result.get("copy")):
            final_text = str(summary_result.get("copy") or "")
            rewrite_reason = f"summary_fallback:{summary_result.get('reason') or 'unknown'}"
            rewritten = True
            source = "summary_fallback"
        elif creative_fallback:
            final_text = creative_fallback
            rewrite_reason = "creative_template_fallback"
            rewritten = True
            source = "creative_template_fallback"
        else:
            final_text = template_fallback
            rewrite_reason = "template_fallback"
            rewritten = True
            source = "template_fallback"

    return {
        "rewritten": rewritten,
        "rewrite_reason": rewrite_reason,
        "final_candidate": {
            **top_candidate,
            "text": final_text,
            "source": source,
        },
    }


def final_ad_copy_step(rewriter_bundle: Dict[str, object]) -> Dict[str, object]:
    final_candidate = dict(rewriter_bundle.get("final_candidate") or {})
    return {
        "final_ad_copy": str(final_candidate.get("text") or ""),
        "final_source": str(final_candidate.get("source") or "unknown"),
        "final_candidate": final_candidate,
    }


def llm_full_pipeline(
    query: str,
    corpus_path: Path,
    rank_path: Path,
    users_path: Optional[Path],
    pairs_path: Optional[Path],
    llm_config_path: Optional[Path] = None,
    top_k: int = 3,
    candidate_count: int = 5,
    requested_tone: str = "creative",
    user_id: Optional[int] = None,
    session_id: Optional[int] = None,
    runtime_context: Optional[V5RuntimeContext] = None,
    provider_override: Optional[str] = None,
) -> Dict[str, object]:
    if runtime_context is not None:
        retrieval_bundle = retrieve_evidence_bundle_cached(
            query=query,
            runtime_context=runtime_context,
            top_k=top_k,
            user_id=user_id,
            session_id=session_id,
        )
    else:
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
    query_bundle = query_normalization_step(str(retrieval_bundle.get("raw_query") or query))
    intent_bundle = intent_enricher_step(query_bundle=query_bundle, retrieval_result=dict(retrieval_bundle.get("result") or {}))
    evidence_bundle = evidence_selector_step(
        intent_bundle=intent_bundle,
        retrieval_result=dict(retrieval_bundle.get("result") or {}),
        top_k=top_k,
    )
    user_profile_bundle = user_profile_builder_step(
        user_id=user_id,
        retrieval_result=dict(retrieval_bundle.get("result") or {}),
        users_map=retrieval_bundle.get("users_map"),
    )
    style_bundle = style_retriever_step(
        query_bundle=query_bundle,
        intent_bundle=intent_bundle,
        user_profile_bundle=user_profile_bundle,
        requested_tone=requested_tone,
    )
    prompt_bundle = prompt_builder_step(
        query_bundle=query_bundle,
        intent_bundle=intent_bundle,
        evidence_bundle=evidence_bundle,
        user_profile_bundle=user_profile_bundle,
        style_bundle=style_bundle,
        candidate_count=candidate_count,
    )
    provider_bundle = llm_provider_step(
        prompt_bundle=prompt_bundle,
        query_bundle=query_bundle,
        intent_bundle=intent_bundle,
        evidence_bundle=evidence_bundle,
        user_profile_bundle=user_profile_bundle,
        style_bundle=style_bundle,
        llm_config_path=llm_config_path,
        candidate_count=candidate_count,
        provider_override=provider_override,
    )
    parsed_bundle = llm_output_parser_step(provider_bundle)
    validated_bundle = copy_validator_step(
        parsed_bundle=parsed_bundle,
        query_bundle=query_bundle,
        intent_bundle=intent_bundle,
        evidence_bundle=evidence_bundle,
        style_bundle=style_bundle,
    )
    ranked_bundle = copy_ranker_step(
        validated_bundle=validated_bundle,
        intent_bundle=intent_bundle,
        evidence_bundle=evidence_bundle,
    )
    rewriter_bundle = copy_rewriter_step(
        ranked_bundle=ranked_bundle,
        query_bundle=query_bundle,
        intent_bundle=intent_bundle,
        evidence_bundle=evidence_bundle,
        user_profile_bundle=user_profile_bundle,
    )
    final_bundle = final_ad_copy_step(rewriter_bundle)
    retrieval_runtime_cache = dict(retrieval_bundle.get("runtime_cache") or {})

    return {
        "module_schemas": MODULE_SCHEMAS,
        "query_normalization": query_bundle,
        "intent_enricher": intent_bundle,
        "evidence_selector": evidence_bundle,
        "user_profile_builder": user_profile_bundle,
        "style_retriever": style_bundle,
        "prompt_builder": prompt_bundle,
        "llm_provider": provider_bundle,
        "provider": str(provider_bundle.get("provider") or ""),
        "model": str(provider_bundle.get("model_name") or ""),
        "llm_status": str(provider_bundle.get("status") or ""),
        "llm_output_parser": parsed_bundle,
        "copy_validator": validated_bundle,
        "copy_ranker": ranked_bundle,
        "copy_rewriter": rewriter_bundle,
        "final_ad_copy": final_bundle,
        "retrieval_bundle": retrieval_bundle,
        "runtime_cache": {
            "enabled": runtime_context is not None,
            "cache_status": retrieval_runtime_cache.get("cache_status", "disabled" if runtime_context is None else "unknown"),
            "cache_hits": runtime_context.cache_hits if runtime_context is not None else 0,
            "cache_misses": runtime_context.cache_misses if runtime_context is not None else 0,
            "retrieval_cache_size": retrieval_runtime_cache.get("retrieval_cache_size", len(runtime_context.retrieval_cache) if runtime_context is not None else 0),
            "rank_loaded": retrieval_runtime_cache.get("rank_loaded", False),
            "corpus_loaded": retrieval_runtime_cache.get("corpus_loaded", False),
        },
    }


def v5_llm_dynamic_creative(**kwargs: object) -> Dict[str, object]:
    """Alias entry point for the full V5 dynamic creative pipeline."""

    return llm_full_pipeline(**kwargs)


def format_pipeline_trace(pipeline_output: Dict[str, object]) -> str:
    ordered_steps = [
        "query_normalization",
        "intent_enricher",
        "evidence_selector",
        "user_profile_builder",
        "style_retriever",
        "prompt_builder",
        "llm_provider",
        "llm_output_parser",
        "copy_validator",
        "copy_ranker",
        "copy_rewriter",
        "final_ad_copy",
    ]
    lines: List[str] = []
    for step in ordered_steps:
        lines.append(f"=== {step} ===")
        lines.append(json.dumps(pipeline_output.get(step) or {}, ensure_ascii=False, indent=2))
        lines.append("")
    return "\n".join(lines).rstrip()
