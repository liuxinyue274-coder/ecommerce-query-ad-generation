"""V5 广告生成可视化看板 (Streamlit)

用途：
- 实时输入 Query，调用 V5 主链路（v5_llm_dynamic_creative），展示
  解析出的意图结构 -> 召回的广告商品 -> 候选广告文案 -> 最终文案。
- 离线浏览 outputs/demo_v5_outputs.jsonl 中的历史结果。
- 一键标记 badcase，落盘到本地 outputs/badcases.jsonl，方便日常调试与 SFT 数据筛选。

启动：
    streamlit run app.py
或在 Windows 下双击 run.bat。
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# UI 增强库（defensive import：未安装时自动降级，不影响主流程）
# ---------------------------------------------------------------------------
try:
    import streamlit_antd_components as sac  # type: ignore
    _HAS_SAC = True
except ImportError:
    sac = None  # type: ignore
    _HAS_SAC = False

try:
    import streamlit_shadcn_ui as sui  # type: ignore
    _HAS_SUI = True
except ImportError:
    sui = None  # type: ignore
    _HAS_SUI = False

try:
    from streamlit_extras.colored_header import colored_header  # type: ignore
    _HAS_EXTRAS = True
except ImportError:
    colored_header = None  # type: ignore
    _HAS_EXTRAS = False

# ---------------------------------------------------------------------------
# 路径与项目根
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = APP_DIR / "outputs"

DEFAULT_PROJECT_ROOT = Path("./v5_main_project")  # 设置环境变量 KS_PROJECT_ROOT 覆盖
PROJECT_ROOT = Path(os.environ.get("KS_PROJECT_ROOT") or DEFAULT_PROJECT_ROOT)
AD_GEN_DIR = PROJECT_ROOT / "ad_generation"

# 把 ad_generation 加到 sys.path 以便 import V5 模块
if AD_GEN_DIR.exists() and str(AD_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(AD_GEN_DIR))

st.set_page_config(
    page_title="Ecommerce CopyGen Studio · LLM 驱动的电商广告生成与诊断平台",
    page_icon="🎯",
    layout="wide",
)


# ---------------------------------------------------------------------------
# 全局视觉(Champagne Gold 高级金属感)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
:root {
  --bg-base: #FAFAF9;
  --bg-card: #FFFFFF;
  --bg-tint: #F5F5F4;
  --text-primary: #1C1917;
  --text-muted: #78716C;
  --border-default: #E7E5E4;
  --border-focus: #C9A876;
  --gold-light: #D4B27D;
  --gold-mid: #B8985C;
  --gold-deep: #8B6F3D;
  /* Rosegold & Coppers status palette */
  --rose-success: #D8A09C;     /* blush rosé,正向反馈 */
  --rose-success-deep: #B8786E;
  --copper-info: #9C8E84;      /* serenity 暖灰,中性提示 */
  --copper-warn: #C9A876;      /* champagne gold */
  --copper-danger: #7C5E50;    /* satin glow 深栗,负向警示 */
  --copper-accent: #C29488;    /* elegant copper 点缀 */
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: -apple-system, "PingFang SC", "Microsoft YaHei", "Inter", system-ui, sans-serif;
  background: var(--bg-base);
  color: var(--text-primary);
}

/* 限宽 + 顶部呼吸 */
.block-container {
  max-width: 1280px;
  padding-top: 1.2rem;
  padding-bottom: 4rem;
}

/* 数字等宽 */
[data-testid="stMetric"], [data-testid="stMetricValue"],
[data-testid="stDataFrame"] td, [data-testid="stTable"] td {
  font-variant-numeric: tabular-nums;
}

/* 默认卡片(container border + expander)收紧、不加阴影 */
[data-testid="stExpander"] {
  border: 1px solid var(--border-default) !important;
  border-radius: 10px !important;
  box-shadow: none !important;
}
[data-testid="stExpander"]:hover {
  border-color: #A8A29E !important;
}
[data-testid="stExpander"] summary {
  font-weight: 600;
  color: var(--text-primary);
}

/* st.container(border=True) */
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 10px !important;
}

/* Streamlit 主按钮:演示橙金 + 白字,统一所有 primary 按钮 */
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"] {
  background: #F2A93B !important;
  border: 1px solid #D88F1F !important;
  color: #FFFFFF !important;
  font-weight: 600 !important;
  letter-spacing: 0.01em;
  box-shadow: none !important;
  transition: background 120ms ease, border-color 120ms ease;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover {
  background: #E89B25 !important;
  border-color: #C07F18 !important;
  color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:active {
  background: #C07F18 !important;
  color: #FFFFFF !important;
}

/* secondary 按钮:克制描边 */
.stButton > button[kind="secondary"] {
  border: 1px solid var(--border-default) !important;
  background: var(--bg-card) !important;
  color: var(--text-primary) !important;
}
.stButton > button[kind="secondary"]:hover {
  border-color: var(--border-focus) !important;
  color: var(--gold-deep) !important;
}

/* 输入框聚焦金边 */
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus,
[data-baseweb="select"] > div:focus-within {
  border-color: var(--gold-mid) !important;
  box-shadow: 0 0 0 2px rgba(184,152,92,0.15) !important;
}

/* dataframe 行高、表头 */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border-default);
  border-radius: 8px;
  overflow: hidden;
}

/* inline code (反引号 / `code` / strategy chip) — 默认绿色改香槟金 */
.stMarkdown code, [data-testid="stMarkdownContainer"] code, code {
  color: #8B6F3D !important;
  background: #F5F0E5 !important;
  border: 1px solid #E7DFD0 !important;
  border-radius: 4px !important;
  padding: 1px 6px !important;
  font-size: 12.5px !important;
}

/* 自定义 valid 标记 */
.cg-valid { color: #C9A876 !important; font-weight: 700; }
.cg-invalid { color: #7C5E50 !important; font-weight: 700; }
.cg-pink-section { color: #D8A09C !important; }
.cg-gold-section { color: #8B6F3D !important; }

/* 结果条:浅米底 + 香槟金细边,彻底告别 success 绿 */
.cg-result-strip {
  background: #FAF4E8;
  border: 1px solid #E0D2B0;
  border-left: 3px solid #B8985C;
  border-radius: 6px;
  padding: 10px 16px;
  color: #4A3A1F;
  font-size: 14px;
  margin: 4px 0 12px 0;
}
.cg-result-strip code {
  background: rgba(184,152,92,0.10) !important;
  color: #6B5424 !important;
  border: 1px solid rgba(184,152,92,0.25) !important;
}

/* dataframe 内 valid 列绿色✅彻底替换:由 pandas Styler 着色;
   兜底:任何 td 内的纯文字 ✓ ✗ 也走金色 */
[data-testid="stDataFrame"] td { color: #1C1917; }

/* tabs 选中态金色下划线 + 玫瑰金底 */
[data-baseweb="tab-list"] [aria-selected="true"] {
  color: #4A2A26 !important;
  background: #F5E1DD !important;
  border-radius: 6px 6px 0 0;
}
[data-baseweb="tab-highlight"] {
  background: #C9A876 !important;
}

/* sac.steps 完成态铜色(尽力命中) */
.ant-steps-item-finish .ant-steps-item-icon {
  background: var(--gold-mid) !important;
  border-color: var(--gold-deep) !important;
}
.ant-steps-item-finish .ant-steps-item-icon > .ant-steps-icon {
  color: #FFFFFF !important;
}
.ant-steps-item-finish > .ant-steps-item-container > .ant-steps-item-tail::after {
  background: var(--gold-mid) !important;
}
.ant-steps-item-process .ant-steps-item-icon {
  background: var(--gold-light) !important;
  border-color: var(--gold-deep) !important;
}

/* sac.segmented 选中态纯香槟金 */
[data-testid="stSegmentedControl"] button[aria-checked="true"],
.ant-segmented-item-selected {
  background: #B8985C !important;
  color: #1C1917 !important;
}

/* Streamlit alert 状态色:取自 Rosegold & Coppers 图 */
.stAlert[data-baseweb="notification"][kind="success"],
[data-testid="stAlertContainer"][data-baseweb="notification"][kind="success"],
div[data-testid="stNotification"][kind="success"],
.stSuccess {
  background: #D8A09C !important;
  border: 1px solid #B8786E !important;
  color: #4A2A26 !important;
}
.stSuccess code, .stSuccess [data-testid="stMarkdownContainer"] code {
  color: #4A2A26 !important;
  background: rgba(255,255,255,0.35) !important;
}
.stAlert[kind="info"], .stInfo,
div[data-testid="stNotification"][kind="info"] {
  background: #F2EFEC !important;
  border: 1px solid #9C8E84 !important;
  color: #4A4039 !important;
}
.stAlert[kind="warning"], .stWarning,
div[data-testid="stNotification"][kind="warning"] {
  background: #FBF3E2 !important;
  border: 1px solid #C9A876 !important;
  color: #6B5424 !important;
}
.stAlert[kind="error"], .stError,
div[data-testid="stNotification"][kind="error"] {
  background: #F2E8E2 !important;
  border: 1px solid #7C5E50 !important;
  color: #4A3429 !important;
}

/* 通用兜底:任何 background 是 streamlit 默认绿/红/蓝的 alert,统一覆盖 */
[data-testid="stAlertContentSuccess"] { color: #6B3F38 !important; }
[data-testid="stAlertContentInfo"] { color: #4A4039 !important; }
[data-testid="stAlertContentWarning"] { color: #6B5424 !important; }
[data-testid="stAlertContentError"] { color: #4A3429 !important; }

/* Brand header 块 */
.brand-header {
  background: linear-gradient(135deg, #FFFFFF 0%, #FAFAF9 55%, #F5E6CC 100%);
  border: 1px solid #E7E5E4;
  border-radius: 12px;
  padding: 22px 26px;
  margin-bottom: 20px;
}
.brand-header .brand-title {
  font-size: 30px;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1.15;
  color: var(--text-primary);
  margin: 0;
}
.brand-header .brand-title .accent {
  color: var(--gold-deep);
  font-weight: 800;
}
.brand-header .brand-sub {
  font-size: 13.5px;
  color: var(--text-muted);
  margin-top: 6px;
}

/* Footer */
.app-footer {
  text-align: center;
  color: #A8A29E;
  font-size: 12px;
  margin: 40px 0 8px 0;
  letter-spacing: 0.04em;
}

/* Section 标题节奏 */
h3 {
  font-size: 18px !important;
  font-weight: 600 !important;
  letter-spacing: -0.005em;
  margin-top: 6px !important;
}

/* metric 卡边框 */
[data-testid="stMetric"] {
  border: 1px solid var(--border-default);
  border-radius: 10px;
  padding: 12px 14px;
  background: var(--bg-card);
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# V5 链路调用（带 runtime cache）
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="首次加载 V5 runtime 上下文（语料/排序/用户/配对），约 5-30 秒...")
def load_runtime() -> Dict[str, Any]:
    """构建一次 V5 runtime context，后续请求复用，避免每次扫描全量数据。"""
    from v5_runtime import build_v5_runtime_context  # type: ignore

    corpus_path = PROJECT_ROOT / "items_lite" / "train.jsonl"
    rank_path = PROJECT_ROOT / "rank_lite" / "train.jsonl"
    users_path = PROJECT_ROOT / "users_lite" / "train.jsonl"
    pairs_path = AD_GEN_DIR / "data" / "train_pairs.jsonl"

    runtime_context = build_v5_runtime_context(
        corpus_path=corpus_path,
        rank_path=rank_path,
        users_path=users_path,
        pairs_path=pairs_path,
    )
    return {
        "runtime_context": runtime_context,
        "corpus_path": corpus_path,
        "rank_path": rank_path,
        "users_path": users_path,
        "pairs_path": pairs_path,
    }


def run_query(query: str, provider: str, top_k: int = 3, candidate_count: int = 5) -> Dict[str, Any]:
    """实时调用 V5 主链路，返回完整 result。"""
    from v5_dynamic_creative import v5_llm_dynamic_creative  # type: ignore

    ctx = load_runtime()
    llm_config_path = AD_GEN_DIR / (
        "llm_config.deepseek_chat.json" if provider == "deepseek_chat" else "llm_config.mock.json"
    )

    return v5_llm_dynamic_creative(
        query=query,
        corpus_path=ctx["corpus_path"],
        rank_path=ctx["rank_path"],
        users_path=ctx["users_path"],
        pairs_path=ctx["pairs_path"],
        llm_config_path=llm_config_path,
        top_k=top_k,
        candidate_count=candidate_count,
        requested_tone="creative",
        runtime_context=ctx["runtime_context"],
        provider_override=provider,
    )


def _cache_key(query: str, provider: str, top_k: int, candidate_count: int) -> Tuple[str, str, int, int]:
    return (query.strip(), provider, int(top_k), int(candidate_count))


def run_query_timed(
    query: str,
    provider: str,
    top_k: int = 3,
    candidate_count: int = 5,
    use_cache: bool = True,
) -> Tuple[Dict[str, Any], float, bool]:
    """run_query + 计时 + 进程内缓存。返回 (result, elapsed_sec, cache_hit)。

    缓存放在 st.session_state 里（按 query+provider+top_k+candidate_count 维度），
    避免反复调试同一条 query 时重复打 deepseek_chat。
    """
    cache: Dict[Tuple[str, str, int, int], Dict[str, Any]] = st.session_state.setdefault("_run_cache", {})
    key = _cache_key(query, provider, top_k, candidate_count)
    if use_cache and key in cache:
        return cache[key], 0.0, True
    t0 = time.perf_counter()
    result = run_query(query, provider, top_k, candidate_count)
    elapsed = time.perf_counter() - t0
    cache[key] = result
    return result, elapsed, False


# ---------------------------------------------------------------------------
# 结果渲染
# ---------------------------------------------------------------------------
def _safe_get(d: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    cur: Any = d or {}
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


def render_pair_table(result: Dict[str, Any]) -> None:
    """🎯 系统产物：商品 ↔ 文案 配对表（替代之前的"最终广告文案"绿框）。

    核心观念：本系统的真实产物是「一组 (商品, 文案) 配对」，没有"唯一最终文案"。
    每个召回商品配上引用了它的最高 rank 候选文案，没被引用的走 Top1 候选兜底。
    """
    items: List[Dict[str, Any]] = (
        _safe_get(result, "evidence_selector", "selected_evidence_items", default=[]) or []
    )
    ranked: List[Dict[str, Any]] = (
        _safe_get(result, "copy_ranker", "ranked_candidates", default=[]) or []
    )
    pairs = _build_item_copy_pairs(items, ranked)

    n_real = sum(1 for p in pairs if p["source"] == "linked")
    n_distributed = sum(1 for p in pairs if p["source"] == "distributed")
    n_fallback = sum(1 for p in pairs if p["source"] == "fallback")
    n_total = len(pairs)
    st.markdown("### 🎯 系统产物：商品 ↔ 文案 配对")
    st.caption(
        f"召回 {len(items)} 条商品 · 生成 {len(ranked)} 条候选文案 · "
        f"配对 {n_total} 条（直接引用 {n_real}，按rank分配 {n_distributed}，兜底重复 {n_fallback}）"
    )
    if not pairs:
        st.info("当前 query 没有可展示的配对。")
        return

    rows = []
    for i, p in enumerate(pairs, 1):
        it = p["item"]
        rows.append(
            {
                "#": i,
                "商品 (item_title)": it.get("item_title") or it.get("title") or "(无标题)",
                "LLM 文案": p["text"] or "(空)",
                "rank": (
                    round(p["rank_score"], 4)
                    if isinstance(p["rank_score"], (int, float))
                    else None
                ),
                "valid": "✓" if p["is_valid"] else ("✗" if p["is_valid"] is False else "·"),
                "来源": {
                    "linked": "直接引用",
                    "distributed": "按rank分配",
                    "fallback": "兜底重复",
                    "none": "无候选",
                }.get(p["source"], p["source"]),
            }
        )
    df_pairs = pd.DataFrame(rows)

    def _valid_color(v: str) -> str:
        if v == "✓":
            return "color: #C9A876; font-weight: 700;"
        if v == "✗":
            return "color: #7C5E50; font-weight: 700;"
        return "color: #A8A29E;"

    styled = df_pairs.style.applymap(_valid_color, subset=["valid"])
    st.dataframe(styled, width="stretch", hide_index=True)


def _build_item_copy_pairs(
    items: List[Dict[str, Any]], ranked: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """为每个 item 找它最佳的候选文案。

    优先级：
      1. 引用了该 item 的候选中 rank_score 最高的那条 → source="linked"
      2. 没有候选引用该 item 时，从尚未被使用的候选里按 rank_score 顺序分配
         一条 → source="distributed"（保证 m≥n 时每个商品拿到不同文案）
      3. 候选不足以覆盖所有 item，剩余 item 用 ranked[0] 兜底 → source="fallback"
      4. 连一条候选都没有 → source="none"，text 为空
    """
    # 1) 收集每个 item_id 引用了它的候选列表
    refs: Dict[str, List[Dict[str, Any]]] = {}
    for c in ranked:
        if not isinstance(c, dict):
            continue
        used = c.get("used_evidence") or []
        if isinstance(used, str):
            used = [used]
        for uid in used:
            uid_s = str(uid)
            if uid_s:
                refs.setdefault(uid_s, []).append(c)

    # 用 id() 标识候选避免 dict 不可哈希
    used_ids: set = set()

    # 2) 第一遍：linked
    linked_choice: Dict[int, Dict[str, Any]] = {}
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        iid = str(it.get("item_id") or "")
        cand_list = refs.get(iid, [])
        if cand_list:
            best = max(cand_list, key=lambda c: c.get("rank_score") or 0)
            linked_choice[idx] = best
            used_ids.add(id(best))

    # 3) 剩余候选池（按 rank_score 降序，去掉已 linked 的）
    sorted_ranked = sorted(
        [c for c in ranked if isinstance(c, dict)],
        key=lambda c: c.get("rank_score") or 0,
        reverse=True,
    )
    pool = [c for c in sorted_ranked if id(c) not in used_ids]
    fallback = sorted_ranked[0] if sorted_ranked else None

    pairs: List[Dict[str, Any]] = []
    for idx, it in enumerate(items):
        if not isinstance(it, dict):
            continue
        if idx in linked_choice:
            best = linked_choice[idx]
            pairs.append(
                {
                    "item": it,
                    "candidate": best,
                    "text": (best.get("text") or "").strip(),
                    "rank_score": best.get("rank_score"),
                    "is_valid": best.get("is_valid"),
                    "source": "linked",
                }
            )
        elif pool:
            picked = pool.pop(0)
            pairs.append(
                {
                    "item": it,
                    "candidate": picked,
                    "text": (picked.get("text") or "").strip(),
                    "rank_score": picked.get("rank_score"),
                    "is_valid": picked.get("is_valid"),
                    "source": "distributed",
                }
            )
        elif fallback is not None:
            pairs.append(
                {
                    "item": it,
                    "candidate": fallback,
                    "text": (fallback.get("text") or "").strip(),
                    "rank_score": fallback.get("rank_score"),
                    "is_valid": fallback.get("is_valid"),
                    "source": "fallback",
                }
            )
        else:
            pairs.append(
                {
                    "item": it,
                    "candidate": None,
                    "text": "",
                    "rank_score": None,
                    "is_valid": None,
                    "source": "none",
                }
            )
    return pairs


def render_status_strip(result: Dict[str, Any]) -> None:
    fallback = bool(_safe_get(result, "llm_provider", "fallback_used", default=False))
    fallback_reason = _safe_get(result, "llm_provider", "fallback_reason", default="")
    cache_status = _safe_get(result, "runtime_cache", "cache_status", default="-")
    cols = st.columns(5)
    cols[0].metric("Provider", str(result.get("provider") or "-"))
    cols[1].metric("Model", str(result.get("model") or "-"))
    cols[2].metric("LLM Status", str(result.get("llm_status") or "-"))
    cols[3].metric("Fallback", "是" if fallback else "否")
    cols[4].metric("Cache", str(cache_status))
    if fallback and fallback_reason:
        st.warning(f"fallback 原因：{fallback_reason}")


def render_intent(result: Dict[str, Any]) -> None:
    intent = result.get("intent_enricher") or {}
    st.markdown("### 意图结构")
    payload = {
        "intent_summary": intent.get("intent_summary"),
        "scenario": intent.get("scenario"),
        "audience": intent.get("audience"),
        "purchase_focus": intent.get("purchase_focus"),
    }
    st.json(payload, expanded=True)


def _leaf_category(category_path: Any) -> str:
    """从 'A > B > C' 形式的 category_path 取叶子类目。"""
    if not category_path:
        return ""
    return str(category_path).split(">")[-1].strip()


def render_evidence(result: Dict[str, Any]) -> None:
    items: List[Dict[str, Any]] = (
        _safe_get(result, "evidence_selector", "selected_evidence_items", default=[]) or []
    )
    common_sp: List[str] = _safe_get(result, "evidence_selector", "common_selling_points", default=[]) or []

    st.markdown(
        f"<h3 class='cg-gold-section'>召回的广告商品（{len(items)}）</h3>",
        unsafe_allow_html=True,
    )
    if not items:
        st.caption("无")
        return

    # 品类分布统计（按叶子类目）
    leaf_counts: Dict[str, int] = {}
    for it in items:
        leaf = _leaf_category(it.get("category_path")) or "(unknown)"
        leaf_counts[leaf] = leaf_counts.get(leaf, 0) + 1

    if len(leaf_counts) == 1:
        only_cat = next(iter(leaf_counts))
        st.warning(
            f"⚠️ **召回扎堆**：{len(items)} 条证据全部是同一叶子类目 「{only_cat}」。"
            "对开放/探索型 query，这通常意味着 evidence_selector 的多样性不足。"
        )
    else:
        st.caption(f"叶子类目分布：{len(leaf_counts)} 个不同类目")
        st.bar_chart(leaf_counts, height=120, color="#B8786E")

    if common_sp:
        st.markdown("**common_selling_points**：" + " · ".join(f"`{s}`" for s in common_sp[:10]))

    rows: List[Dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        rows.append(
            {
                "item_id": it.get("item_id"),
                "title": it.get("item_title") or it.get("title"),
                "leaf_category": _leaf_category(it.get("category_path")),
                "brand": it.get("brand_name") or it.get("brand"),
                "ranking_signal": it.get("ranking_signal"),
                "relevance_signal": it.get("relevance_signal"),
                "purchase_cnt": it.get("purchase_cnt"),
                "click_cnt": it.get("click_cnt"),
            }
        )
    st.dataframe(rows, width="stretch", hide_index=True)


def render_candidates(result: Dict[str, Any]) -> None:
    ranked: List[Dict[str, Any]] = _safe_get(result, "copy_ranker", "ranked_candidates", default=[]) or []
    st.markdown("### 候选广告文案 Top3")
    if not ranked:
        st.caption("（无候选）")
        return
    for i, c in enumerate(ranked[:3], 1):
        score = c.get("rank_score")
        score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "-"
        text = c.get("text") or c.get("copy") or ""
        strategy = c.get("strategy") or "-"
        cid = c.get("candidate_id") or "-"
        is_valid = c.get("is_valid")
        if is_valid:
            valid_badge = "<span class='cg-valid'>✓</span>"
        elif is_valid is False:
            valid_badge = "<span class='cg-invalid'>✗</span>"
        else:
            valid_badge = "·"
        st.markdown(
            f"**#{i}** {valid_badge}  ·  rank_score = `{score_str}`  ·  策略：`{strategy}`  ·  id：`{cid}`",
            unsafe_allow_html=True,
        )
        st.markdown(f"> {text}")
        subscores = c.get("subscores") or {}
        if subscores and isinstance(subscores, dict):
            sub_str = "  ·  ".join(f"{k}=`{v}`" for k, v in subscores.items())
            st.caption(f"subscores：{sub_str}")


def render_copy_product_pairs(result: Dict[str, Any]) -> None:
    """🌟 醒目区：候选广告文案 ↔ 召回商品 配对卡片，类似淘宝商品标题列表。

    取 ranked_candidates 的 Top-N，每条文案旁边贴上它 used_evidence 引用的商品（item_title 等），
    没有引用的就在卡尾标注 "通用候选（未指定商品）"。
    """
    ranked: List[Dict[str, Any]] = _safe_get(result, "copy_ranker", "ranked_candidates", default=[]) or []
    items: List[Dict[str, Any]] = _safe_get(result, "evidence_selector", "selected_evidence_items", default=[]) or []
    if not ranked:
        return

    # 建索引：item_id -> item dict
    item_by_id: Dict[str, Dict[str, Any]] = {}
    for it in items:
        if isinstance(it, dict) and it.get("item_id") is not None:
            item_by_id[str(it.get("item_id"))] = it

    st.markdown("### 🛍️ 候选文案 ↔ 召回商品（配对预览）")
    st.caption(
        f"召回 {len(items)} 条商品，生成 {len(ranked)} 条候选文案。"
        "下面把 Top 候选与它引用的商品贴在一起，便于一眼看出"
        "「文案在卖什么 / 商品有没有被用上」。"
    )

    top_n = min(3, len(ranked))
    cols = st.columns(top_n) if top_n > 1 else [st.container()]
    for i, (col, c) in enumerate(zip(cols, ranked[:top_n]), 1):
        with col:
            with st.container(border=True):
                # 文案（醒目大字）
                text = c.get("text") or c.get("copy") or ""
                score = c.get("rank_score")
                score_str = f"{score:.2f}" if isinstance(score, (int, float)) else "-"
                is_valid = c.get("is_valid")
                if is_valid:
                    valid_badge = "<span class='cg-valid'>✓</span>"
                elif is_valid is False:
                    valid_badge = "<span class='cg-invalid'>✗</span>"
                else:
                    valid_badge = "·"
                st.markdown(
                    f"**#{i}** {valid_badge}  ·  rank=`{score_str}`  ·  策略 `{c.get('strategy') or '-'}`",
                    unsafe_allow_html=True,
                )
                st.markdown(f"#### 📣 {text}" if text else "#### _（空）_")

                # 引用的商品（淘宝标题样式）
                used_ids = c.get("used_evidence") or []
                if isinstance(used_ids, str):
                    used_ids = [used_ids]
                linked = [item_by_id.get(str(uid)) for uid in used_ids if str(uid) in item_by_id]
                linked = [x for x in linked if x]

                if linked:
                    st.markdown("**🛒 引用商品**")
                    for it in linked:
                        title = it.get("item_title") or it.get("title") or "(无标题)"
                        brand = it.get("brand_name") or it.get("brand") or ""
                        leaf = _leaf_category(it.get("category_path"))
                        purchase = it.get("purchase_cnt")
                        click = it.get("click_cnt")
                        # 商品标题（蓝色超链接样式）
                        st.markdown(f"🟦 **{title}**")
                        meta_bits = []
                        if brand:
                            meta_bits.append(f"品牌 `{brand}`")
                        if leaf:
                            meta_bits.append(f"类目 `{leaf}`")
                        if isinstance(purchase, (int, float)):
                            meta_bits.append(f"购买 `{int(purchase)}`")
                        if isinstance(click, (int, float)):
                            meta_bits.append(f"点击 `{int(click)}`")
                        if meta_bits:
                            st.caption(" · ".join(meta_bits))
                else:
                    st.caption("🟨 通用候选（未指定具体商品）")

                # 校验问题
                issues = c.get("issues") or []
                if issues:
                    st.caption("⚠ issues：" + " · ".join(f"`{x}`" for x in issues[:5]))

    # 整图：所有召回商品（即使没被任何候选引用，也展示）
    used_set = {
        str(uid)
        for c in ranked
        for uid in (c.get("used_evidence") or ([c.get("used_evidence")] if isinstance(c.get("used_evidence"), str) else []))
    }
    unused = [it for it in items if isinstance(it, dict) and str(it.get("item_id")) not in used_set]
    if unused:
        with st.expander(f"未被任何 Top 候选引用的商品（{len(unused)}）", expanded=False):
            for it in unused:
                title = it.get("item_title") or it.get("title") or "(无标题)"
                leaf = _leaf_category(it.get("category_path"))
                brand = it.get("brand_name") or it.get("brand") or ""
                meta = " · ".join(x for x in [f"品牌 `{brand}`" if brand else "", f"类目 `{leaf}`" if leaf else ""] if x)
                st.markdown(f"🟦 **{title}**")
                if meta:
                    st.caption(meta)


def render_validator(result: Dict[str, Any]) -> None:
    val = result.get("copy_validator") or {}
    summary = val.get("validator_summary") or {}
    cands: List[Dict[str, Any]] = val.get("validated_candidates") or []
    with st.expander(f"校验结果 / 风险提示（{len(cands)} 条候选）", expanded=False):
        st.markdown("**validator_summary**")
        st.json(summary, expanded=True)
        if cands:
            rows = []
            for c in cands:
                issues = c.get("issues")
                if isinstance(issues, list):
                    issues_str = "; ".join(str(x) for x in issues) if issues else ""
                else:
                    issues_str = str(issues) if issues else ""
                rows.append(
                    {
                        "candidate_id": c.get("candidate_id"),
                        "is_valid": c.get("is_valid"),
                        "issues": issues_str,
                        "text": c.get("text"),
                    }
                )
            st.dataframe(rows, width="stretch", hide_index=True)


def render_advanced(result: Dict[str, Any]) -> None:
    with st.expander("用户画像 / 风格样例 / 重写过程 / 原始 LLM 输出", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**user_profile_builder**")
            st.json(result.get("user_profile_builder") or {}, expanded=False)
            st.markdown("**style_retriever**")
            st.json(result.get("style_retriever") or {}, expanded=False)
        with c2:
            st.markdown("**copy_rewriter**")
            st.json(result.get("copy_rewriter") or {}, expanded=False)
            st.markdown("**llm_provider.raw_generations**")
            st.json(_safe_get(result, "llm_provider", "raw_generations", default=[]) or [], expanded=False)


def render_full_json(result: Dict[str, Any], key_prefix: str = "") -> None:
    with st.expander("完整 result（JSON / 可下载）", expanded=False):
        text = json.dumps(result, ensure_ascii=False, indent=2)
        st.download_button(
            "⬇️ 下载完整 result.json",
            data=text.encode("utf-8"),
            file_name="result.json",
            mime="application/json",
            key=f"{key_prefix}_dl_result",
        )
        preview_limit = 6000
        if len(text) > preview_limit:
            st.code(text[:preview_limit] + "\n...\n[已截断，请下载查看完整内容]", language="json")
        else:
            st.code(text, language="json")


# ---------------------------------------------------------------------------
# C 端用户视角预览（演示用，emoji 当商品图）
# ---------------------------------------------------------------------------
_CATEGORY_EMOJI_RULES: List[Tuple[Tuple[str, ...], str]] = [
    (("手环", "手表", "智能穿戴", "运动表"), "⌚"),
    (("耳机", "音响", "耳麦"), "🎧"),
    (("手机", "手机壳", "充电"), "📱"),
    (("电脑", "笔记本", "键盘", "鼠标", "数码"), "💻"),
    (("相机", "摄影"), "📷"),
    (("吹风机", "美容", "美妆", "面膜", "面霜", "护肤", "口红"), "💄"),
    (("零食", "饮料", "茶", "咖啡", "酒", "食品"), "🍫"),
    (("书", "图书", "教辅"), "📚"),
    (("玩具", "母婴"), "🧸"),
    (("鞋", "运动鞋", "球鞋"), "👟"),
    (("包", "双肩包", "背包", "箱包"), "🎒"),
    (("服饰", "服装", "T恤", "衣"), "👕"),
    (("家居", "家具", "厨房"), "🏠"),
    (("宠物",), "🐾"),
    (("健身", "运动", "户外"), "🏃"),
]


def _item_emoji(item: Dict[str, Any]) -> str:
    """根据 item 标题/类目猜一个 emoji 当商品图。"""
    hay = " ".join(
        str(item.get(k) or "")
        for k in ("item_title", "title", "category_path", "brand_name", "brand")
    )
    for keys, emoji in _CATEGORY_EMOJI_RULES:
        if any(k in hay for k in keys):
            return emoji
    return "🛍️"


def _stars_from_signals(item: Dict[str, Any]) -> Tuple[float, str]:
    """从 click/purchase 信号粗暴换算 4.x 颗星，纯演示。"""
    click = item.get("click_cnt") or item.get("click_rate") or 0
    purchase = item.get("purchase_cnt") or item.get("purchase_rate") or 0
    try:
        c = float(click)
        p = float(purchase)
    except (TypeError, ValueError):
        c, p = 0.0, 0.0
    base = 4.2 + min(0.7, (p * 0.0008 + c * 0.0002))
    base = max(3.5, min(5.0, base))
    full = int(base)
    half = 1 if base - full >= 0.5 else 0
    empty = 5 - full - half
    return base, "★" * full + ("☆" if half else "") + "·" * empty


def _fake_price(item: Dict[str, Any]) -> str:
    """没有真实价格，按 item_id hash 出一个稳定的 ¥XX-XXX 占位。"""
    if "price" in item and item.get("price"):
        return f"¥{item.get('price')}"
    seed = abs(hash(str(item.get("item_id") or item.get("item_title") or "x")))
    price = 49 + (seed % 950)
    return f"¥{price}"


def _sold_count(item: Dict[str, Any]) -> str:
    p = item.get("purchase_cnt")
    if isinstance(p, (int, float)) and p > 0:
        v = int(p)
        if v >= 10000:
            return f"已售 {v / 10000:.1f}w+"
        if v >= 1000:
            return f"已售 {v // 1000}k+"
        return f"已售 {v}"
    seed = abs(hash(str(item.get("item_id") or "x"))) % 9000 + 100
    return f"已售 {seed}+"


def render_consumer_view(query: str, result: Dict[str, Any], key_prefix: str = "consumer") -> None:
    """🛒 C 端用户视角的商品列表（Taobao 风格，emoji 占位图）。

    页面顶部的 Taobao 风格搜索框 / 排序 tabs 由 tab_live 自己绘制（因为搜索框是真 query 输入），
    这个函数只负责渲染商品卡列表。

    数据来源：复用 V5 evidence_selector + copy_ranker 结果，按 _build_item_copy_pairs 配对。
    """
    items: List[Dict[str, Any]] = (
        _safe_get(result, "evidence_selector", "selected_evidence_items", default=[]) or []
    )
    ranked: List[Dict[str, Any]] = (
        _safe_get(result, "copy_ranker", "ranked_candidates", default=[]) or []
    )
    pairs = _build_item_copy_pairs(items, ranked)

    if not pairs:
        st.info("当前 query 没有召回到商品；C 端预览暂无商品可展示。")
        return

    # 商品卡片 3 列网格
    cols_per_row = 3
    rows = [pairs[i : i + cols_per_row] for i in range(0, len(pairs), cols_per_row)]
    for row_idx, row_pairs in enumerate(rows):
        cols = st.columns(cols_per_row)
        for col_idx, pair in enumerate(row_pairs):
            it = pair["item"]
            ad_text = pair["text"]
            is_fallback = pair["source"] == "fallback"
            with cols[col_idx]:
                with st.container(border=True):
                    emoji = _item_emoji(it)
                    title = it.get("item_title") or it.get("title") or "(无标题)"
                    brand = it.get("brand_name") or it.get("brand") or ""
                    leaf = _leaf_category(it.get("category_path"))
                    rating, stars = _stars_from_signals(it)
                    sold = _sold_count(it)
                    price = _fake_price(it)

                    # emoji 大图
                    st.markdown(
                        f"<div style='font-size:84px;text-align:center;line-height:1.1;"
                        f"background:#FFF5EE;border-radius:8px;padding:18px 0;'>"
                        f"{emoji}</div>",
                        unsafe_allow_html=True,
                    )
                    # B 方案：LLM 文案当主标题（加粗），原 item_title 降级为副标题（灰小字）
                    if ad_text:
                        tag_html = (
                            "<span style='background:#E5E7EB;color:#6B7280;font-size:11px;"
                            "padding:1px 6px;border-radius:4px;margin-left:6px;'>通用</span>"
                            if is_fallback
                            else ""
                        )
                        st.markdown(
                            f"<div style='font-size:16px;font-weight:700;color:#1F2937;"
                            f"line-height:1.45;margin-top:6px;'>"
                            f"💡 {ad_text}{tag_html}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f"<div style='font-size:16px;font-weight:700;color:#1F2937;"
                            f"line-height:1.45;margin-top:6px;'>{title}</div>",
                            unsafe_allow_html=True,
                        )
                    # 原 item_title 副标题（灰小字）—— 仅当上面用了 ad_text 时才需要展示
                    if ad_text:
                        st.markdown(
                            f"<div style='font-size:12px;color:#9CA3AF;line-height:1.4;"
                            f"margin-top:2px;'>{title}</div>",
                            unsafe_allow_html=True,
                        )
                    # 品牌 + 类目
                    bits = [b for b in [brand, leaf] if b]
                    if bits:
                        st.caption(" · ".join(bits))
                    # 评分 + 销量
                    st.markdown(
                        f"<span style='color:#FFA500;'>{stars}</span> "
                        f"<span style='color:#6B7280;font-size:13px;'> {rating:.1f} · {sold}</span>",
                        unsafe_allow_html=True,
                    )
                    # 价格
                    st.markdown(
                        f"<span style='color:#FF2D2D;font-size:22px;font-weight:700;'>"
                        f"{price}</span>",
                        unsafe_allow_html=True,
                    )
                    # 操作按钮
                    bcols = st.columns(2)
                    btn_key1 = f"{key_prefix}_buy_{row_idx}_{col_idx}"
                    btn_key2 = f"{key_prefix}_cart_{row_idx}_{col_idx}"
                    if bcols[0].button("立即购买", key=btn_key1, type="primary", width="stretch"):
                        st.toast(f"🛒 演示模式：未对接交易系统（{title[:14]}…）", icon="✨")
                    if bcols[1].button("加购物车", key=btn_key2, width="stretch"):
                        st.toast(f"🛍️ 已加入购物车（演示）：{title[:14]}…", icon="✨")


def render_taobao_search_bar(default_query: str, key_prefix: str = "demo") -> Tuple[str, str, bool]:
    """演示模式顶部的 Taobao 风格搜索框：返回 (query, provider, run_clicked)。

    布局：[ ← ] [ 🔍 输入框 ] [ provider ▽ ] [ 搜索 ]
          [ 综合 销量 价格↑ 筛选 ▽ ] (装饰)
    """
    with st.container(border=True):
        cols = st.columns([0.6, 7, 1.6, 1.2])
        cols[0].markdown(
            "<div style='font-size:26px;text-align:center;color:#9CA3AF;"
            "padding-top:4px;'>←</div>",
            unsafe_allow_html=True,
        )
        with cols[1]:
            q = st.text_input(
                "搜索商品",
                value=default_query or "",
                placeholder="🔍 试试搜索：运动手环 / 宿舍吹风机 / 通勤双肩包 ...",
                label_visibility="collapsed",
                key=f"{key_prefix}_query",
            )
        with cols[2]:
            provider = st.selectbox(
                "provider",
                ["local_fake", "deepseek_chat", "sft_local"],
                index=["local_fake", "deepseek_chat", "sft_local"].index(
                    st.session_state.get(f"{key_prefix}_provider", "local_fake")
                ),
                label_visibility="collapsed",
                key=f"{key_prefix}_provider",
            )
        with cols[3]:
            run_clicked = st.button(
                "🔍 搜索", type="primary", width="stretch", key=f"{key_prefix}_run"
            )
        # 排序 tabs（纯装饰）
        st.markdown(
            "<div style='display:flex;gap:18px;padding:6px 4px 2px 4px;"
            "color:#1F2937;font-size:14px;'>"
            "<span style='color:#FF6B35;font-weight:600;border-bottom:2px solid #FF6B35;"
            "padding-bottom:4px;'>综合</span>"
            "<span>销量</span>"
            "<span>价格 ↑</span>"
            "<span>筛选 ▽</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    return q, provider, run_clicked


def render_pipeline_steps(result: Dict[str, Any], key: str = "pipeline_steps") -> None:
    """用 sac.steps 把 V5 主链路走过的 4 个阶段可视化。"""
    intent_ok = bool(_safe_get(result, "intent_enricher", "intent_summary", default=""))
    n_evidence = len(_safe_get(result, "evidence_selector", "selected_evidence_items", default=[]) or [])
    n_cands = len(_safe_get(result, "copy_ranker", "ranked_candidates", default=[]) or [])
    final_ok = bool(_safe_get(result, "final_ad_copy", "final_ad_copy", default=""))

    if _HAS_SAC:
        # index 表示当前活动 step 的 0-based 序号；4 个 step 合法范围 0-3。
        if final_ok:
            cur_idx = 3
        elif n_cands:
            cur_idx = 2
        elif n_evidence:
            cur_idx = 1
        else:
            cur_idx = 0
        sac.steps(
            items=[
                sac.StepsItem(title="意图增强", subtitle="intent" + (" ✓" if intent_ok else "")),
                sac.StepsItem(title="证据选择", subtitle=f"召回 {n_evidence} 条"),
                sac.StepsItem(title="候选生成 / 排序", subtitle=f"{n_cands} 条候选"),
                sac.StepsItem(title="最终文案", subtitle="final" + (" ✓" if final_ok else "")),
            ],
            index=cur_idx,
            return_index=False,
            key=key,
        )
    else:
        cols = st.columns(4)
        cols[0].metric("意图", "✓" if intent_ok else "—")
        cols[1].metric("召回商品", n_evidence)
        cols[2].metric("候选文案", n_cands)
        cols[3].metric("最终文案", "✓" if final_ok else "—")


def render_badcase_form(query: str, provider: str, result: Dict[str, Any], key_prefix: str) -> None:
    st.markdown("### 🐞 标记为 Badcase")
    # 与离线评测指标对齐的标签清单：
    #   BERTScore-F1 / PPL / Distinct-2 / Fallback 率
    # 再叠加几个常见的人工 badcase 模式（评测指标覆盖不到但人工一眼能看出来）。
    tag_options = [
        "语义不匹配（BERTScore 低）",
        "表达不通顺（PPL 高）",
        "表达模板化（Distinct-2 低）",
        "触发 Fallback（证据不足）",
        "复述商品标题",
        "错用品牌词",
        "风险词 / 合规问题",
        "信息编造（与证据不符）",
        "无亮点 / 缺乏行动召唤",
    ]
    with st.form(key=f"{key_prefix}_badcase_form", clear_on_submit=True):
        reason = st.text_area(
            "原因 / 描述（可选）",
            placeholder="可空。也可以只勾标签不写文字 —— 但理由和标签至少要有一项。",
            key=f"{key_prefix}_reason",
        )
        col_a, col_b = st.columns([1, 2])
        with col_a:
            severity = st.selectbox("严重程度", ["low", "medium", "high"], index=1, key=f"{key_prefix}_sev")
        with col_b:
            tags_selected = st.multiselect(
                "标签（对齐评测指标，可多选）",
                options=tag_options,
                key=f"{key_prefix}_tags_sel",
                help=(
                    "前 4 项对齐离线自动评测指标："
                    "BERTScore-F1（语义匹配）/ PPL（通顺度）/ Distinct-2（多样性）/ Fallback 率。"
                    "后面是常见人工 badcase 模式。"
                ),
            )
        tags_extra = st.text_input(
            "其他标签（逗号分隔，可空）",
            placeholder="例如：宿舍场景, 大促节点",
            key=f"{key_prefix}_tags_extra",
        )
        submitted = st.form_submit_button("加入 badcase 列表")
        if submitted:
            extra_tags = [t.strip() for t in tags_extra.split(",") if t.strip()]
            tags = list(tags_selected) + extra_tags
            # 去重保序
            seen = set()
            tags = [t for t in tags if not (t in seen or seen.add(t))]
            if not reason.strip() and not tags:
                st.warning("⚠️ 请至少填写「原因」或勾选/填写一个「标签」")
                return
            entry = {
                "marked_at": datetime.now().isoformat(timespec="seconds"),
                "query": query,
                "provider": provider,
                "severity": severity,
                "reason": reason.strip(),
                "tags": tags,
                "final_ad_copy": _safe_get(result, "final_ad_copy", "final_ad_copy", default=""),
                "ranked_top3": [
                    {
                        "candidate_id": c.get("candidate_id"),
                        "strategy": c.get("strategy"),
                        "rank_score": c.get("rank_score"),
                        "text": c.get("text"),
                    }
                    for c in (_safe_get(result, "copy_ranker", "ranked_candidates", default=[]) or [])[:3]
                ],
                "intent": result.get("intent_enricher") or {},
                "validator_summary": _safe_get(result, "copy_validator", "validator_summary", default={}),
                "llm_status": result.get("llm_status"),
                "fallback_used": _safe_get(result, "llm_provider", "fallback_used", default=False),
                "fallback_reason": _safe_get(result, "llm_provider", "fallback_reason", default=""),
            }
            OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            badcase_path = OUTPUTS_DIR / "badcases.jsonl"
            with badcase_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            st.success(f"✅ 已写入 {badcase_path}")


def render_result(query: str, provider: str, result: Dict[str, Any], key_prefix: str) -> None:
    mode = st.session_state.get("view_mode", "调试模式")
    is_demo = mode == "演示模式"

    # 链路进度（两种模式都显示）
    render_pipeline_steps(result, key=f"steps_{key_prefix}")

    if is_demo:
        # ========== 🛒 演示模式：只展示 C 端商品列表 + Badcase 表单 ==========
        render_consumer_view(query, result, key_prefix=f"{key_prefix}_consumer")
        st.divider()
        render_badcase_form(query, provider, result, key_prefix)
    else:
        # ========== 🔧 调试模式：完整诊断信息 ==========
        # 顶部：商品 ↔ 文案 配对表（系统真实产物）
        render_pair_table(result)
        render_status_strip(result)
        st.divider()
        # C 端预览（默认折叠）
        with st.expander("🛒 C 端用户视角预览（点击展开）", expanded=False):
            render_consumer_view(query, result, key_prefix=f"{key_prefix}_consumer")
        st.divider()
        with st.expander("🛍️ 全部候选文案诊断（每条候选 ↔ 引用商品）", expanded=False):
            render_copy_product_pairs(result)
        st.divider()
        col1, col2 = st.columns([1, 1])
        with col1:
            render_intent(result)
        with col2:
            render_evidence(result)
        st.divider()
        render_candidates(result)
        render_validator(result)
        render_advanced(result)
        render_full_json(result, key_prefix=key_prefix)
        st.divider()
        render_badcase_form(query, provider, result, key_prefix)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 看板设置")
    provider = st.selectbox(
        "LLM Provider",
        ["local_fake", "deepseek_chat", "sft_local"],
        help=(
            "local_fake：本地 mock，无需 API；"
            "deepseek_chat：调真实 DeepSeek API（需 DEEPSEEK_API_KEY 环境变量）；"
            "sft_local：本地 SFT/LoRA 入口（需 SFT_MODEL_PATH）。"
        ),
    )
    top_k = st.slider("evidence top_k", 1, 10, 3)
    candidate_count = st.slider("candidate_count", 1, 10, 5)

    st.markdown("---")
    st.caption(f"项目根：`{PROJECT_ROOT}`")
    st.caption(f"输出目录：`{OUTPUTS_DIR}`")

    if not AD_GEN_DIR.exists():
        st.error(f"❌ 找不到 ad_generation 目录：{AD_GEN_DIR}")
        st.caption("可设置环境变量 KS_PROJECT_ROOT 指向 V5 main project 的实际位置。")

    if provider == "deepseek_chat" and not os.environ.get("DEEPSEEK_API_KEY"):
        st.warning("未检测到 DEEPSEEK_API_KEY，将自动 fallback 到 local_fake。")
    if provider == "sft_local" and not os.environ.get("SFT_MODEL_PATH"):
        st.info("未配置 SFT_MODEL_PATH，sft_local 会优雅 fallback。")

    st.markdown("---")
    use_cache = st.toggle(
        "结果缓存（同 query 秒回）",
        value=True,
        help="按 (query, provider, top_k, candidate_count) 维度缓存。调试反复跑同一条 query 时极快。",
        key="sb_use_cache",
    )
    cache_size = len(st.session_state.get("_run_cache", {}))
    st.caption(f"已缓存 {cache_size} 条结果")
    cc1, cc2 = st.columns(2)
    if cc1.button("🧹 清结果缓存", width="stretch"):
        st.session_state["_run_cache"] = {}
        st.success("结果缓存已清空。")
    if cc2.button("🔄 清 runtime", width="stretch"):
        load_runtime.clear()
        st.success("runtime cache 已清，下次请求会重建。")
    parallel_max = st.slider(
        "多 query 并发数",
        min_value=1,
        max_value=8,
        value=4,
        help="多样性增强时并发跑 sub-query 的线程数；deepseek_chat 主要瓶颈是网络等待，并发能显著缩短墙钟时间。",
        key="sb_parallel_max",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
# 顶部 header：标题 + 全局模式切换
header_l, header_r = st.columns([3, 2])
with header_l:
    st.markdown(
        """
<div class="brand-header">
  <div class="brand-title">🎯 Ecommerce CopyGen <span class="accent">Studio</span></div>
  <div class="brand-sub">LLM 驱动的电商广告生成与诊断平台</div>
</div>
""",
        unsafe_allow_html=True,
    )
with header_r:
    st.markdown("<br>", unsafe_allow_html=True)
    if _HAS_SAC:
        mode = sac.segmented(
            items=[
                sac.SegmentedItem(label="🔧 调试模式", icon=None),
                sac.SegmentedItem(label="🛒 演示模式", icon=None),
            ],
            index=0 if st.session_state.get("view_mode", "调试模式") == "调试模式" else 1,
            return_index=False,
            color="yellow",
            bg_color="transparent",
            key="view_mode_segmented",
        )
        # sac.segmented 返回 label 字符串
        st.session_state["view_mode"] = "调试模式" if "调试" in str(mode) else "演示模式"
    else:
        mode = st.radio(
            "模式",
            ["🔧 调试模式", "🛒 演示模式"],
            index=0 if st.session_state.get("view_mode", "调试模式") == "调试模式" else 1,
            horizontal=True,
            label_visibility="collapsed",
            key="view_mode_radio",
        )
        st.session_state["view_mode"] = "调试模式" if "调试" in mode else "演示模式"

is_demo = st.session_state.get("view_mode", "调试模式") == "演示模式"

if not is_demo:
    st.caption(
        "Query → 意图增强 → 证据选择 → 候选生成（多 provider）→ 校验/排序/重写 → 最终广告文案"
    )

tab_live, tab_offline, tab_badcase = st.tabs(["● 实时调用", "📂 离线浏览（JSONL）", "🐞 Badcase 列表"])

# --- Tab: 实时调用 ---
with tab_live:
    if is_demo:
        # 🛒 演示模式：Taobao 风格搜索框作为 query 入口（替代调试风的"输入 Query + ▶ 运行"）
        # 隐藏 sidebar 让画面更沉浸
        st.markdown(
            "<style>section[data-testid='stSidebar']{display:none !important;}</style>",
            unsafe_allow_html=True,
        )
        default_q = st.session_state.get("shared_query", "运动手环")
        query, demo_provider, run_btn = render_taobao_search_bar(default_q, key_prefix="demo")
        # 用演示模式的 provider 覆盖 sidebar 的 provider
        provider = demo_provider
        # 🔥 热门搜索 标签（替代调试模式 5 个示例按钮）
        st.markdown(
            "<div style='font-size:13px;color:#6B7280;padding:8px 0 4px 0;'>"
            "🔥 热门搜索</div>",
            unsafe_allow_html=True,
        )
        sample_cols = st.columns(5)
        samples = ["运动手环", "宿舍吹风机", "通勤双肩包", "敏感肌面霜", "手机壳"]
        for col, sample in zip(sample_cols, samples):
            if col.button(sample, key=f"hot_{sample}", width="stretch"):
                st.session_state["pending_query"] = sample
                st.session_state["pending_run"] = True
                st.session_state["demo_query"] = sample  # 同步到搜索框
        # 演示模式下不暴露多样性增强（属于调试功能）
        diversity_on = False
        sub_queries_raw = ""
    else:
        # 🔧 调试模式：原始的"输入 Query + ▶ 运行"
        col_q, col_btn = st.columns([4, 1])
        with col_q:
            query = st.text_input(
                "输入 Query",
                value=st.session_state.get("shared_query", "运动手环"),
                placeholder="例如：运动手环 / 宿舍吹风机 / 通勤双肩包 / 敏感肌面霜",
                key="debug_query",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("▶ 运行", type="primary", width="stretch")

        sample_cols = st.columns(5)
        samples = ["运动手环", "宿舍吹风机", "通勤双肩包", "敏感肌面霜", "手机壳"]
        for col, sample in zip(sample_cols, samples):
            if col.button(sample, key=f"sample_{sample}", width="stretch"):
                st.session_state["pending_query"] = sample
                st.session_state["pending_run"] = True

        # --- Query 多样性增强（前端 workaround：多 query 各跑一次，合并展示） ---
        with st.expander("🧠 Query 多样性增强（多 query 头脑风暴）", expanded=False):
            st.caption(
                "V5 召回器对单 query 倾向于聚焦同一品类。"
                "若想让候选文案覆盖更多角度/品类，可在下面写多条 sub-query（每行一条），"
                "看板会并发调用 V5 并合并所有候选文案后统一展示。"
            )
            diversity_on = st.toggle("启用多样性增强", value=False, key="live_diversity_on")
            sub_queries_raw = st.text_area(
                "Sub-queries（每行一条，留空则只用主 query）",
                height=110,
                placeholder="例：\n端午节送朋友的茶礼盒\n端午节送朋友的咖啡礼盒\n端午节送朋友的香薰\n端午节送朋友的零食礼包",
                key="live_sub_queries",
                disabled=not diversity_on,
            )

    if st.session_state.get("pending_run"):
        query = st.session_state.pop("pending_query", query)
        run_btn = True
        st.session_state["pending_run"] = False

    # 同步 query 到两个模式共用的 session key
    if (query or "").strip():
        st.session_state["shared_query"] = query.strip()
    if run_btn and (query or "").strip():
        # 组装 query 列表
        qlist: List[str] = [query.strip()]
        if diversity_on:
            for ln in (sub_queries_raw or "").splitlines():
                s = ln.strip()
                if s and s not in qlist:
                    qlist.append(s)

        if len(qlist) == 1:
            with st.spinner(f"正在用 `{provider}` 跑 V5 链路..."):
                try:
                    result, elapsed, hit = run_query_timed(
                        qlist[0], provider, top_k=top_k, candidate_count=candidate_count, use_cache=use_cache
                    )
                    st.session_state["last_result"] = (qlist[0], provider, result)
                    st.session_state["last_timing"] = {"elapsed": elapsed, "hit": hit}
                    st.session_state.pop("last_diversity", None)
                except Exception as e:  # noqa: BLE001
                    st.error(f"运行失败：{type(e).__name__}: {e}")
                    st.exception(e)
        else:
            sub_results: List[Dict[str, Any]] = []
            errors: List[str] = []
            timings: Dict[str, Tuple[float, bool]] = {}
            wall_t0 = time.perf_counter()
            progress = st.progress(0.0, text=f"多 query 并发调用（max={parallel_max}）：0/{len(qlist)}")
            with ThreadPoolExecutor(max_workers=max(1, min(parallel_max, len(qlist)))) as ex:
                fut_to_q = {
                    ex.submit(
                        run_query_timed, sq, provider, top_k, candidate_count, use_cache
                    ): sq
                    for sq in qlist
                }
                done = 0
                for fut in as_completed(fut_to_q):
                    sq = fut_to_q[fut]
                    try:
                        r, el, hit = fut.result()
                        sub_results.append({"query": sq, "result": r})
                        timings[sq] = (el, hit)
                    except Exception as e:  # noqa: BLE001
                        errors.append(f"`{sq}`：{type(e).__name__}: {e}")
                    done += 1
                    progress.progress(
                        done / len(qlist),
                        text=f"多 query 并发调用（max={parallel_max}）：{done}/{len(qlist)}",
                    )
            progress.empty()
            wall_elapsed = time.perf_counter() - wall_t0
            for err in errors:
                st.error(err)
            if sub_results:
                order = {q: i for i, q in enumerate(qlist)}
                sub_results.sort(key=lambda x: order.get(x["query"], 9999))
                st.session_state["last_diversity"] = {
                    "queries": qlist,
                    "sub_results": sub_results,
                    "provider": provider,
                    "timings": timings,
                    "wall_elapsed": wall_elapsed,
                }
                st.session_state["last_result"] = (qlist[0], provider, sub_results[0]["result"])
                st.session_state["last_timing"] = {
                    "elapsed": timings.get(qlist[0], (0.0, False))[0],
                    "hit": timings.get(qlist[0], (0.0, False))[1],
                }

    # --- 结果展示 ---
    diversity_state = st.session_state.get("last_diversity")
    if diversity_state and diversity_state.get("provider") == provider:
        wall = diversity_state.get("wall_elapsed", 0.0)
        timings = diversity_state.get("timings", {})
        sum_solo = sum(t[0] for t in timings.values()) or 0.0
        speedup = (sum_solo / wall) if wall > 0 else 0.0
        hit_n = sum(1 for t in timings.values() if t[1])
        st.success(
            f"⏱ 多 query 总墙钟 **{wall:.2f}s**  ·  串行估算 {sum_solo:.2f}s  ·  "
            f"并发加速 ≈ **{speedup:.1f}×**  ·  cache 命中 {hit_n}/{len(timings)}"
        )
        with st.expander("各 sub-query 耗时", expanded=False):
            timing_rows = [
                {
                    "query": q,
                    "elapsed_s": round(timings.get(q, (0.0, False))[0], 2),
                    "cache": "hit" if timings.get(q, (0.0, False))[1] else "miss",
                }
                for q in diversity_state.get("queries", [])
            ]
            st.dataframe(timing_rows, width="stretch", hide_index=True)
        st.markdown(f"#### 多 query 合并结果（{len(diversity_state['sub_results'])} 条 sub-query）")

        merged: Dict[str, Dict[str, Any]] = {}
        all_evidence_cats: Dict[str, int] = {}
        for sr in diversity_state["sub_results"]:
            r = sr["result"]
            cands = _safe_get(r, "copy_ranker", "ranked_candidates", default=[]) or []
            for c in cands:
                if not isinstance(c, dict):
                    continue
                txt = (c.get("text") or "").strip()
                if not txt:
                    continue
                score = c.get("rank_score") or 0
                prev = merged.get(txt)
                if prev is None or (score or 0) > (prev.get("rank_score") or 0):
                    cc = dict(c)
                    cc["_from_query"] = sr["query"]
                    merged[txt] = cc
            for it in (_safe_get(r, "evidence_selector", "selected_evidence_items", default=[]) or []):
                if not isinstance(it, dict):
                    continue
                leaf = _leaf_category(it.get("category_path")) or "(unknown)"
                all_evidence_cats[leaf] = all_evidence_cats.get(leaf, 0) + 1

        merged_list = sorted(
            merged.values(), key=lambda c: (c.get("rank_score") or 0), reverse=True
        )
        st.caption(
            f"合并候选 {len(merged_list)} 条（去重后） · "
            f"覆盖 {len(all_evidence_cats)} 个叶子类目"
        )
        if all_evidence_cats:
            st.bar_chart(all_evidence_cats, height=140, color="#B8786E")

        st.markdown("##### 合并后 Top 候选")
        for i, c in enumerate(merged_list[:10], 1):
            score = c.get("rank_score")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            is_valid = c.get("is_valid")
            if is_valid:
                valid_badge = "<span class='cg-valid'>✓</span>"
            elif is_valid is False:
                valid_badge = "<span class='cg-invalid'>✗</span>"
            else:
                valid_badge = "·"
            st.markdown(
                f"**#{i}** {valid_badge}  ·  rank_score = `{score_str}`  ·  "
                f"strategy=`{c.get('strategy')}`  ·  来自 query=`{c.get('_from_query')}`",
                unsafe_allow_html=True,
            )
            st.write(c.get("text", ""))
            issues = c.get("issues") or []
            if issues:
                st.caption("issues：" + " · ".join(f"`{x}`" for x in issues))
            st.divider()

        with st.expander("逐条 sub-query 详情", expanded=False):
            for sr in diversity_state["sub_results"]:
                st.markdown(f"---\n##### Sub-query：`{sr['query']}`")
                render_result(sr["query"], provider, sr["result"], key_prefix=f"div_{sr['query']}")
    elif "last_result" in st.session_state:
        q, p, r = st.session_state["last_result"]
        timing = st.session_state.get("last_timing") or {}
        if timing:
            if timing.get("hit"):
                badge = "<span style='color:#B8985C;font-weight:700;'>● cache hit</span>"
            else:
                badge = f"⏱ {timing.get('elapsed', 0):.2f}s"
            st.markdown(
                f"<div class='cg-result-strip'>结果：<code>{q}</code> · provider=<code>{p}</code> · {badge}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"#### 结果：`{q}` · provider=`{p}`")
        render_result(q, p, r, key_prefix="live")
    else:
        st.info("点击右上『▶ 运行』或下面的示例 query，开始第一次调用。")


# --- Tab: 离线浏览 ---
with tab_offline:
    default_jsonl = OUTPUTS_DIR / "demo_v5_outputs_sample.jsonl"
    paths_str = st.text_area(
        "JSONL 文件路径（每行一个，可同时加载多个）",
        value=str(default_jsonl),
        height=80,
        help="默认指向 demo_v5_pipeline.py 的输出。也支持 real_sft_ad_copy.jsonl 等其他每行 {query, result} 结构的 JSONL。",
    )
    paths = [Path(p.strip()) for p in paths_str.splitlines() if p.strip()]

    records: List[Dict[str, Any]] = []
    load_errs: List[str] = []
    for p in paths:
        if not p.exists():
            load_errs.append(f"找不到 `{p}`")
            continue
        try:
            with p.open("r", encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        rec = json.loads(ln)
                        rec["_source"] = p.name
                        records.append(rec)
        except Exception as e:  # noqa: BLE001
            load_errs.append(f"读取 `{p}` 失败：{e}")
    for err in load_errs:
        st.warning(err)

    if not records:
        st.info(
            "暂无可浏览的记录。先运行 "
            "`python -X utf8 ad_generation/demo_v5_pipeline.py --limit 5 --provider local_fake` "
            "生成 JSONL，或修改上方路径。"
        )
    else:
        summary_rows: List[Dict[str, Any]] = []
        for i, rec in enumerate(records):
            r = rec.get("result") or {}
            ranked = _safe_get(r, "copy_ranker", "ranked_candidates", default=[]) or []
            top_score = ranked[0].get("rank_score") if ranked and isinstance(ranked[0], dict) else None
            v_summary = _safe_get(r, "copy_validator", "validator_summary", default={}) or {}
            evidence = _safe_get(r, "evidence_selector", "selected_evidence_items", default=[]) or []
            cats = {_leaf_category(it.get("category_path")) for it in evidence if isinstance(it, dict)}
            cats.discard(None)
            summary_rows.append(
                {
                    "#": i,
                    "source": rec.get("_source"),
                    "query": rec.get("query"),
                    "provider": r.get("provider") or "-",
                    "fallback": "是" if _safe_get(r, "llm_provider", "fallback_used", default=False) else "否",
                    "invalid": int(v_summary.get("invalid_count") or 0),
                    "n_cats": len(cats),
                    "top_score": round(top_score, 4) if isinstance(top_score, (int, float)) else None,
                    "final_ad_copy": _safe_get(r, "final_ad_copy", "final_ad_copy", default=""),
                }
            )

        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
        kw = c1.text_input("关键词（query / final_ad_copy）", key="off_kw")
        provider_options = sorted({row["provider"] for row in summary_rows if row["provider"]})
        provider_filter = c2.multiselect("provider", provider_options, key="off_prov")
        fallback_filter = c3.selectbox("fallback", ["全部", "仅是", "仅否"], key="off_fb")
        issue_filter = c4.selectbox("issues", ["全部", "仅有 invalid"], key="off_iss")

        def _match(row: Dict[str, Any]) -> bool:
            if kw:
                hay = f"{row.get('query') or ''} {row.get('final_ad_copy') or ''}"
                if kw.lower() not in hay.lower():
                    return False
            if provider_filter and row.get("provider") not in provider_filter:
                return False
            if fallback_filter == "仅是" and row.get("fallback") != "是":
                return False
            if fallback_filter == "仅否" and row.get("fallback") != "否":
                return False
            if issue_filter == "仅有 invalid" and not row.get("invalid"):
                return False
            return True

        filtered = [row for row in summary_rows if _match(row)]
        st.caption(f"共 {len(records)} 条 · 筛选后 {len(filtered)} 条")
        st.dataframe(filtered, width="stretch", hide_index=True)

        if filtered:
            indices = [row["#"] for row in filtered]
            if "off_current_idx" not in st.session_state or st.session_state["off_current_idx"] not in indices:
                st.session_state["off_current_idx"] = indices[0]
            current = st.session_state["off_current_idx"]

            nav1, nav2, nav3 = st.columns([1, 4, 1])
            if nav1.button("◀ 上一条", width="stretch", disabled=indices.index(current) == 0):
                st.session_state["off_current_idx"] = indices[max(0, indices.index(current) - 1)]
                st.rerun()
            choice = nav2.selectbox(
                "跳转到",
                options=indices,
                index=indices.index(current),
                format_func=lambda i: f"#{i}  {records[i].get('query')}",
                key="off_jump",
            )
            if choice != current:
                st.session_state["off_current_idx"] = choice
                st.rerun()
            if nav3.button("下一条 ▶", width="stretch", disabled=indices.index(current) == len(indices) - 1):
                st.session_state["off_current_idx"] = indices[min(len(indices) - 1, indices.index(current) + 1)]
                st.rerun()

            current = st.session_state["off_current_idx"]
            rec = records[current]
            r = rec.get("result") or {}
            q = rec.get("query") or ""
            p = r.get("provider") or "-"
            st.markdown(f"#### #{current} · `{q}` · provider=`{p}` · 来源 `{rec.get('_source')}`")
            render_result(q, p, r, key_prefix=f"off_{current}")
        else:
            st.info("筛选后没有匹配的记录。")


# --- Tab: Badcase 列表 ---
with tab_badcase:
    bc_path = OUTPUTS_DIR / "badcases.jsonl"
    st.caption(f"badcase 文件路径：`{bc_path}`")
    if not bc_path.exists():
        st.info("暂无 badcase 记录。在『实时调用』或『离线浏览』tab 里点『标记为 badcase』即可写入。")
    else:
        try:
            bc_rows: List[Dict[str, Any]] = []
            with bc_path.open("r", encoding="utf-8") as f:
                for ln in f:
                    if ln.strip():
                        bc_rows.append(json.loads(ln))
        except Exception as e:  # noqa: BLE001
            st.error(f"读取 badcase 文件失败：{e}")
            bc_rows = []

        if not bc_rows:
            st.info("badcase 文件为空。")
        else:
            st.caption(f"共 {len(bc_rows)} 条 badcase")
            preview = [
                {
                    "marked_at": b.get("marked_at"),
                    "query": b.get("query"),
                    "provider": b.get("provider"),
                    "severity": b.get("severity"),
                    "tags": ", ".join(b.get("tags") or []),
                    "reason": b.get("reason"),
                    "final_ad_copy": b.get("final_ad_copy"),
                }
                for b in bc_rows
            ]
            st.dataframe(preview, width="stretch", hide_index=True)
            st.download_button(
                "下载 badcases.jsonl",
                data=bc_path.read_bytes(),
                file_name="badcases.jsonl",
                mime="application/json",
                key="dl_badcases",
            )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    "<div class='app-footer'>Ecommerce CopyGen Studio · v0.3 · 2026 · powered by V5 LLM pipeline</div>",
    unsafe_allow_html=True,
)
