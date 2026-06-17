"""Utilities for prompt assembly and first-version template ad copy."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional


PLACEHOLDER_BRANDS = {
    "",
    "无品牌",
    "其他",
    "其他/other",
    "其它",
    "其它/other",
    "unknown",
    "none",
    "null",
    "nan",
    "other",
}


def _clean_text(value: object) -> str:
    """Convert a value to a stripped string, returning an empty string for blanks."""

    if value is None:
        return ""
    return str(value).strip()


def _first_non_empty(data: Optional[Dict[str, object]], keys: Iterable[str]) -> str:
    """Return the first non-empty value from a mapping for the given keys."""

    if not data:
        return ""
    for key in keys:
        value = _clean_text(data.get(key))
        if value:
            return value
    return ""


def _normalize_compare_text(text: str) -> str:
    """Normalize text for lightweight containment checks."""

    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", _clean_text(text)).lower()


def _is_placeholder_brand(brand: str) -> bool:
    """Return True when brand is effectively empty or just a placeholder."""

    if not brand:
        return True
    normalized = _normalize_compare_text(brand)
    placeholder_tokens = {_normalize_compare_text(item) for item in PLACEHOLDER_BRANDS}
    return normalized in placeholder_tokens


def _title_already_contains_brand(title: str, brand: str) -> bool:
    """Check whether title already contains the brand string or one of its aliases."""

    title_norm = _normalize_compare_text(title)
    if not title_norm:
        return False

    candidates = [brand]
    candidates.extend(re.split(r"[\\/|｜]+", brand))

    for candidate in candidates:
        candidate_norm = _normalize_compare_text(candidate)
        if candidate_norm and candidate_norm in title_norm:
            return True
    return False


def build_user_context(
    user_profile: Optional[Dict[str, object]] = None,
    recent_behavior_titles: Optional[Iterable[object]] = None,
) -> str:
    """Build a compact user-context block for prompting."""

    lines: List[str] = []
    profile = user_profile or {}

    gender = _first_non_empty(profile, ["gender"])
    age = _first_non_empty(profile, ["age_bucket", "age"])
    city = _first_non_empty(profile, ["city", "fre_city"])
    province = _first_non_empty(profile, ["province", "fre_province"])

    traits: List[str] = []
    if gender:
        traits.append(f"性别：{gender}")
    if age:
        traits.append(f"年龄段：{age}")
    if city:
        traits.append(f"城市：{city}")
    elif province:
        traits.append(f"地区：{province}")

    if traits:
        lines.append("用户画像：" + "，".join(traits))

    cleaned_titles: List[str] = []
    for title in recent_behavior_titles or []:
        clean = _clean_text(title)
        if clean and clean not in cleaned_titles:
            cleaned_titles.append(clean)

    if cleaned_titles:
        lines.append("最近行为：" + "；".join(cleaned_titles[:5]))

    return "\n".join(lines)


def build_evidence_block(evidence_items: Optional[Iterable[Dict[str, object]]]) -> str:
    """Render a readable evidence block from top-k items."""

    lines: List[str] = []
    for index, item in enumerate(evidence_items or [], start=1):
        title = _first_non_empty(item, ["item_title", "title"])
        brand = _first_non_empty(item, ["brand_name", "brand"])
        seller = _first_non_empty(item, ["seller_name", "seller"])
        category_path = _first_non_empty(item, ["category_path"])

        fields: List[str] = []
        if title:
            fields.append(title)
        if brand and not _is_placeholder_brand(brand):
            fields.append(f"品牌：{brand}")
        if category_path:
            fields.append(f"类目：{category_path}")
        if seller:
            fields.append(f"店铺：{seller}")

        if fields:
            lines.append(f"{index}. " + " | ".join(fields))

    return "\n".join(lines)


def build_llm_evidence_block(evidence_items: Optional[Iterable[Dict[str, object]]]) -> str:
    """Render a richer evidence block for LLM prompting, including ranking/relevance signals."""

    lines: List[str] = []
    for index, item in enumerate(evidence_items or [], start=1):
        title = _first_non_empty(item, ["item_title", "title"])
        brand = _first_non_empty(item, ["brand_name", "brand"])
        seller = _first_non_empty(item, ["seller_name", "seller"])
        category_path = _first_non_empty(item, ["category_path"])
        ranking_signal = _clean_text(item.get("ranking_signal"))
        relevance_signal = _clean_text(item.get("relevance_signal"))

        fields: List[str] = []
        if title:
            fields.append(f"title={title}")
        if brand:
            fields.append(f"brand={brand}")
        if seller:
            fields.append(f"seller={seller}")
        if category_path:
            fields.append(f"category_path={category_path}")
        if ranking_signal:
            fields.append(f"ranking_signal={ranking_signal}")
        if relevance_signal:
            fields.append(f"relevance_signal={relevance_signal}")

        if fields:
            lines.append(f"{index}. " + " | ".join(fields))

    return "\n".join(lines)


def build_prompt(
    query: str,
    evidence_items: Optional[Iterable[Dict[str, object]]] = None,
    user_profile: Optional[Dict[str, object]] = None,
    recent_behavior_titles: Optional[Iterable[object]] = None,
) -> str:
    """Build the final prompt string for ad generation."""

    clean_query = _clean_text(query)
    user_context = build_user_context(user_profile=user_profile, recent_behavior_titles=recent_behavior_titles)
    evidence_block = build_evidence_block(evidence_items)

    prompt_parts: List[str] = [
        "你是电商广告文案助手。",
        "请根据给定 query 和候选商品证据，输出一条简洁自然的中文广告文案。",
        "不要编造价格、优惠、销量、品牌承诺或不存在的商品卖点。",
        "优先使用商品标题、品牌、类目、店铺等显式信息。",
        "",
        f"Query:\n{clean_query}",
    ]

    if user_context:
        prompt_parts.extend(["", f"User Context:\n{user_context}"])

    if evidence_block:
        prompt_parts.extend(["", f"Evidence Items:\n{evidence_block}"])

    prompt_parts.extend(["", "Output:\n只输出一条中文广告文案，不要解释。"])
    return "\n".join(prompt_parts)


def build_llm_prompt(
    query: str,
    evidence_items: Optional[Iterable[Dict[str, object]]] = None,
    user_profile: Optional[Dict[str, object]] = None,
    recent_behavior_titles: Optional[Iterable[object]] = None,
    copy_tone: str = "creative",
    rewrite_request: bool = False,
) -> str:
    """Build a stricter prompt for LLM-based ad-copy generation."""

    clean_query = _clean_text(query)
    user_context = build_user_context(user_profile=user_profile, recent_behavior_titles=recent_behavior_titles)
    evidence_block = build_llm_evidence_block(evidence_items)

    tone_instructions = {
        "safe": [
            "风格目标：稳妥推荐型。",
            "更像可信的推荐说明，表达克制、完整、稳妥，突出适合谁、适合什么场景、为什么值得先看。",
            "优先写成一句自然说明，不要写成商品标题或口号。",
        ],
        "creative": [
            "风格目标：自然创意型。",
            "输出必须像电商搜索广告短文案，而不是检索解释、候选摘要或商品清单说明。",
            "文案最好包含：用户场景或需求 + 商品方向或核心证据 + 自然引导。",
            "句子必须完整自然、有轻微吸引力，像人在说话，不要只输出标题式短语，也不要只是拼接商品名。",
            "尽量把 query 意图转化成用户可感知的购买场景或选择理由。",
            "不要复述“当前候选主要集中在...”或“建议先从这类商品里继续筛选”这种搜索摘要语气。",
            "风格参考示例：Query：华为手环；输出：运动、通勤都想轻松记录？搜“华为手环”，可以先看看智能手环设备方向。",
            "风格参考示例：Query：宿舍吹风机；输出：宿舍吹发更看重轻便实用，可以优先看看小型家用吹风机。",
            "风格参考示例：Query：iqooz10x手机壳动漫；输出：想给 iQOO Z10x 换个动漫风外观？可以先看适配机型的个性手机壳。",
        ],
        "concise": [
            "风格目标：短句转化型。",
            "尽量压缩为 20~30 字，适合广告位短标题，但仍需自然、完整、可读、可信。",
            "即使更短，也不要只剩商品标题，要保留轻微引导感。",
        ],
    }
    selected_tone = tone_instructions.get(copy_tone, tone_instructions["creative"])

    prompt_parts: List[str] = [
        "角色：你是电商搜索广告文案生成助手。",
        "任务：根据 query、弱个性化用户上下文和 top-k 商品证据，先理解用户搜索意图，再生成一条真实、自然、简洁的中文广告文案。",
        "",
        "请先隐式理解 query 意图中的关键信息：",
        "- 品类",
        "- 品牌",
        "- 功能/卖点关键词",
        "- 使用场景",
        "- 目标人群",
        "- 型号/规格/系列",
        "",
        "请优先利用商品证据中的以下字段完成 Query 意图增强：",
        "- top-k title",
        "- brand",
        "- seller",
        "- category_path",
        "- ranking_signal",
        "- relevance_signal",
        "",
        "用户画像和最近行为只作为弱个性化参考，不能压过当前 query 和商品证据。",
        "",
        *selected_tone,
        "",
        "创意表达目标：",
        "- 文案需要自然、有吸引力，但不能夸张。",
        "- 不要机械复述完整商品标题。",
        "- 尽量把 query 意图转化成用户可感知的购买场景或选择理由。",
        "- 优先提炼‘适合谁、适合什么场景、为什么值得先看’，再组织成广告表达。",
        "- 如果证据里主要只有类目和标题，就做稳妥表达，不要硬造卖点。",
        "",
        "约束：",
        "1. 不编造价格、折扣、销量。",
        "2. 不编造商品不存在的功效。",
        "3. 不编造商品不存在的材质、认证或资质。",
        "4. 不要使用‘专业’‘官方’‘正品’‘高端’‘爆款’等证据不足的强化词。",
        "5. 文案长度控制在 20~50 字；如果 tone=concise，则尽量控制在 20~30 字。",
        "6. 只输出一条中文广告文案。",
        "7. 语气自然，不要夸张营销。",
        "",
        "禁止使用的表达：",
        "- 全网最低",
        "- 爆款必买",
        "- 疯抢",
        "- 百分百有效",
        "",
        "反例约束：不要输出以下模板化句式，只有系统 fallback 时才允许出现：",
        "- 可先关注xxx等商品，更贴近当前需求",
        "- 信息清楚，适合先了解",
        "- 当前候选主要集中在xxx方向",
        "- 推荐先看xxx",
        "- 建议先从这类商品里继续筛选",
        "- 更贴近当前需求",
        "- 可先关注xxx等商品",
        "",
        f"Query:\n{clean_query}",
    ]

    if user_context:
        prompt_parts.extend(["", f"用户信息:\n{user_context}"])

    if evidence_block:
        prompt_parts.extend(["", f"商品证据（按相关候选顺序）:\n{evidence_block}"])

    if rewrite_request:
        prompt_parts.extend(
            [
                "",
                "请改写为更自然的广告短文案，不要写成搜索摘要、候选解释或标题压缩句。",
                "必须避免：当前候选主要集中在...；建议先从这类商品里继续筛选；更贴近当前需求；信息清楚，适合先了解；可先关注xxx等商品。",
            ]
        )

    prompt_parts.extend(["", "输出格式：只输出文案本身，不要解释。"])
    return "\n".join(prompt_parts)


def render_template_copy(
    query: str,
    item: Optional[Dict[str, object]],
    user_profile: Optional[Dict[str, object]] = None,
) -> str:
    """Generate a single Chinese ad copy string from a target item."""

    del user_profile

    item = item or {}
    clean_query = _clean_text(query)
    title = _first_non_empty(item, ["item_title", "title"])
    brand = _first_non_empty(item, ["brand_name", "brand"])
    seller = _first_non_empty(item, ["seller_name", "seller"])
    category_path = _first_non_empty(item, ["category_path"])

    usable_brand = "" if _is_placeholder_brand(brand) else brand
    if usable_brand and _title_already_contains_brand(title, usable_brand):
        usable_brand = ""

    display_title = title
    if usable_brand and title:
        display_title = f"{usable_brand}{title}"

    category_short = ""
    if category_path:
        category_short = category_path.split(">")[-1].strip()

    if display_title and seller:
        return f"搜“{clean_query}”可以先看这款{display_title}，由{seller}在售，信息清楚，适合先了解。"
    if display_title and category_short:
        return f"搜“{clean_query}”可先关注这款{display_title}，属于{category_short}方向，和当前需求更贴近。"
    if display_title:
        return f"结合你搜索的“{clean_query}”，这款{display_title}是当前更匹配的选择，信息直观，适合优先浏览。"
    if category_short:
        return f"结合你搜索的“{clean_query}”，可先关注{category_short}方向的候选商品，便于更快缩小选择范围。"
    return f"结合你搜索的“{clean_query}”，推荐先查看当前匹配度更高的商品证据，快速缩小选择范围。"


def _extract_major_category(category_path: str) -> str:
    """Extract the first-level category from a category path."""

    if not category_path:
        return ""
    return _clean_text(category_path.split(">")[0])


def _extract_leaf_category(category_path: str) -> str:
    """Extract the leaf category from a category path."""

    if not category_path:
        return ""
    return _clean_text(category_path.split(">")[-1])


def _tokenize_title(text: str) -> List[str]:
    """Tokenize title text into mixed Chinese/alnum chunks."""

    clean = _clean_text(text)
    if not clean:
        return []
    tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]{2,}", clean)
    return [token for token in tokens if token]


def _common_title_hint(titles: List[str]) -> str:
    """Extract a lightweight common hint from multiple titles."""

    if len(titles) < 2:
        return ""

    token_sets = []
    for title in titles:
        token_sets.append(set(_tokenize_title(title)))

    common_tokens = set.intersection(*token_sets) if token_sets else set()
    common_tokens = [token for token in common_tokens if len(token) >= 2]
    if not common_tokens:
        return ""

    common_tokens.sort(key=lambda token: (-len(token), token))
    return common_tokens[0]


def _simplify_category_label(category_path: str) -> str:
    """Convert a category path into a shorter, more copy-friendly label."""

    label = _extract_leaf_category(category_path) or _extract_major_category(category_path)
    if not label:
        return ""

    generic_suffixes = ("设备", "配件", "仪器", "用品", "商品")
    for suffix in generic_suffixes:
        if label.endswith(suffix) and len(label) > len(suffix) + 1:
            candidate = label[: -len(suffix)].strip()
            if len(candidate) >= 2:
                return candidate
    return label


def render_creative_template_copy(
    query: str,
    evidence_items: Optional[Iterable[Dict[str, object]]] = None,
    item: Optional[Dict[str, object]] = None,
    user_profile: Optional[Dict[str, object]] = None,
) -> str:
    """Generate a more ad-like fallback copy for creative mode without unsupported claims."""

    del user_profile

    items = list(evidence_items or [])
    anchor_item = item or (items[0] if items else {}) or {}
    clean_query = _clean_text(query)

    titles = [_first_non_empty(entry, ["item_title", "title"]) for entry in items if entry]
    common_title = _common_title_hint([title for title in titles if title])
    if common_title and not re.search(r"[\u4e00-\u9fff]", common_title):
        common_title = ""

    title = _first_non_empty(anchor_item, ["item_title", "title"])

    category_label = _simplify_category_label(_first_non_empty(anchor_item, ["category_path"]))
    if not category_label:
        for evidence_item in items:
            category_label = _simplify_category_label(_first_non_empty(evidence_item, ["category_path"]))
            if category_label:
                break

    generic_labels = {"美发美容", "个护", "手机及", "智能设备", "家用电器"}
    if category_label in generic_labels:
        category_label = ""

    if common_title and len(common_title) <= 10:
        return f"想找“{clean_query}”？不妨先看{common_title}相关款，更容易挑到顺眼又合用的一款。"

    if category_label:
        return f"围绕“{clean_query}”这类需求，不妨先看{category_label}相关款，更容易找到贴合场景的选择。"

    if title:
        return f"想找“{clean_query}”？不妨先看看更贴合这类需求的相关款，更容易挑到顺手的一款。"

    return f"围绕“{clean_query}”的需求，可以先看更贴近搜索意图的相关款式。"


def render_summary_copy(
    query: str,
    evidence_items: Optional[Iterable[Dict[str, object]]],
    user_profile: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Generate a summary-style copy from top-k evidence items.

    Returns a payload with:
    - `copy`: final copy text
    - `fallback`: whether template fallback was used
    - `reason`: explanation for summary or fallback
    """

    del user_profile  # Reserved for later personalization without changing the signature.

    items = list(evidence_items or [])
    if len(items) < 2:
        fallback_copy = render_template_copy(query=query, item=items[0] if items else {}, user_profile=None)
        return {
            "copy": fallback_copy,
            "fallback": True,
            "reason": "insufficient_evidence_items",
        }

    clean_query = _clean_text(query)
    titles = [_first_non_empty(item, ["item_title", "title"]) for item in items]
    brands = [
        brand
        for brand in (_first_non_empty(item, ["brand_name", "brand"]) for item in items)
        if brand and not _is_placeholder_brand(brand)
    ]
    major_categories = [_extract_major_category(_first_non_empty(item, ["category_path"])) for item in items]
    leaf_categories = [_extract_leaf_category(_first_non_empty(item, ["category_path"])) for item in items]

    common_title = _common_title_hint(titles)
    brand_counter = Counter(brands)
    major_counter = Counter([item for item in major_categories if item])
    leaf_counter = Counter([item for item in leaf_categories if item])

    top_brand, top_brand_count = ("", 0)
    if brand_counter:
        top_brand, top_brand_count = brand_counter.most_common(1)[0]

    top_major, top_major_count = ("", 0)
    if major_counter:
        top_major, top_major_count = major_counter.most_common(1)[0]

    top_leaf, top_leaf_count = ("", 0)
    if leaf_counter:
        top_leaf, top_leaf_count = leaf_counter.most_common(1)[0]

    if top_brand and top_leaf and top_brand_count >= 2 and top_leaf_count >= 2:
        return {
            "copy": f"搜“{clean_query}”时，可先关注{top_brand}{top_leaf}方向，当前候选主要集中在这类商品，便于更快缩小选择范围。",
            "fallback": False,
            "reason": "shared_brand_and_leaf_category",
        }

    if top_major and top_major_count == len(items) and top_leaf:
        return {
            "copy": f"搜“{clean_query}”时，当前候选主要集中在{top_major}下的{top_leaf}方向，建议先从这类商品里继续筛选。",
            "fallback": False,
            "reason": "shared_major_category",
        }

    if top_major and top_major_count >= 2:
        return {
            "copy": f"结合“{clean_query}”的搜索意图，当前证据更多落在{top_major}相关商品上，可优先关注这一类方向。",
            "fallback": False,
            "reason": "major_category_majority",
        }

    if common_title:
        return {
            "copy": f"搜“{clean_query}”时，当前候选多围绕“{common_title}”相关商品，可先从这一方向继续筛选。",
            "fallback": False,
            "reason": "common_title_hint",
        }

    fallback_copy = render_template_copy(query=query, item=items[0], user_profile=None)
    return {
        "copy": fallback_copy,
        "fallback": True,
        "reason": "no_summary_signal",
    }
