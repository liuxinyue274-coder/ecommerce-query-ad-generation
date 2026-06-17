# Ecommerce CopyGen Studio · 交接文档

> LLM 驱动的电商广告生成与诊断平台 · Streamlit 可视化前端
> 最后更新：2026-05-24

---

## 1. 项目定位

本目录 `.\` 是一个**纯前端可视化看板**，用于：

1. 实时调用 V5 LLM 广告生成主链路（query → 意图增强 → 证据召回 → 候选生成/排序 → 最终文案），把每一步的中间产物可视化，便于调试。
2. 离线浏览历史 JSONL 输出，快速翻看不同 query 的 V5 跑批结果。
3. 给 PM/设计/老板看的"演示模式"——把后端结果包装成 C 端搜索结果页，用于产品演示。
4. 一键标记 badcase 并落盘到 `outputs/badcases.jsonl`，供日后 SFT 数据筛选。

**重要边界**：本目录**不修改任何 V5 主项目代码**。V5 主代码位于 `<V5_PROJECT_ROOT>\`，看板只通过 import + 调用的方式使用它，前后端职责清晰。

---

## 2. 目录结构

```
.\
├── app.py                    # 主应用,~1780 行,所有 UI 与业务逻辑
├── run.bat                   # Windows 一键启动脚本
├── requirements.txt          # Python 依赖清单
├── README.md                 # 用户向 README
├── HANDOFF.md                # 本文件
├── .streamlit\
│   └── config.toml           # Streamlit 主题配置
├── outputs\
│   └── badcases.jsonl        # 用户标记的 badcase 落盘文件 (本地生成)
└── docs\
```

---

## 3. 启动方式

### 前置条件

- Python 3.9+
- V5 主项目已就位：`<V5_PROJECT_ROOT>\`（路径可通过环境变量 `KS_PROJECT_ROOT` 覆盖）
- 如要用 DeepSeek provider：环境变量 `DEEPSEEK_API_KEY=sk-...`（**绝不要 commit 到代码或日志**）
- 如要用 SFT 本地模型：环境变量 `SFT_MODEL_PATH` 指向模型目录
- 不配置任何 key 时默认走 `local_fake` provider，纯本地 mock，可以直接跑

### 安装与运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

或在 Windows 下双击 `run.bat`（启动前请编辑里面的 `DEEPSEEK_API_KEY`）。

默认浏览器打开 http://localhost:8501。

---

## 4. 架构与数据流

```
用户在浏览器输入 query
        │
        ▼
┌─────────────────────────────────────────┐
│ app.py · run_query(query, provider)     │
│  ↓ import v5_runtime                    │
│  ↓ 复用 @st.cache_resource 加载的全量    │
│    runtime context (语料 + 排序 + 用户) │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ V5 主链路 (V5 主项目 内)          │
│  intent_enricher                        │
│   → evidence_selector                   │
│    → copy_ranker (LLM 生成候选 + 排序)  │
│     → final_ad_copy                     │
│  返回 result dict (含每一步中间产物)    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│ 看板渲染 (按模式分流)                   │
│  调试模式 → render_result (B 端全展开)  │
│  演示模式 → render_consumer_view        │
│              (C 端搜索结果页 3 卡片)    │
└─────────────────────────────────────────┘
```

**Result 关键字段** (V5 主链路返回的 dict)：

- `intent_enricher` — 意图结构（scenario / audience / purchase_focus）
- `evidence_selector` — 召回的广告商品 list (`recalled_items`) + 公共卖点
- `copy_ranker` — LLM 生成的候选文案 list（含 `rank_score`、`is_valid`、`used_evidence`、`strategy`）
- `final_ad_copy` — 最终选定的文案
- `copy_validator` — 校验通过/失败的候选明细
- `llm_provider` / `runtime_cache` — 元信息（provider、是否 fallback、cache hit）

---

## 5. app.py 关键函数索引

| 函数 / 区块 | 行号附近 | 职责 |
| --- | --- | --- |
| `load_runtime` | ~330 | `@st.cache_resource` 缓存的 V5 runtime 上下文（首次启动 5–30 秒）|
| `run_query` | ~360 | 调用 V5 主链路的入口，单 query 触发整条管线 |
| `_safe_get` | helper | 安全多层 dict 取值，防 None 链路炸 |
| `_build_item_copy_pairs` | ~470 | **核心逻辑**：把 n 个商品和 m 条候选文案做配对，三档优先级：直接引用 → 按 rank 分配（轮流）→ 兜底重复 |
| `render_status_strip` | ~588 | 顶部 5 列 metric（Provider / Model / LLM Status / Fallback / Cache）|
| `render_intent` | ~602 | 意图结构展示 |
| `render_evidence` | ~620 | 召回商品 + 叶子类目分布 bar chart |
| `render_pair_table` | ~430 | 商品-文案配对表 |
| `render_validator` | ~786 | copy_validator 校验摘要 |
| `render_consumer_view` | ~953 | **演示模式**：C 端搜索结果页 + 商品卡 + 立即购买 |
| `render_result` | ~1163 | 调试模式总入口，按顺序串起所有 render_xxx |
| Badcase 表单 | ~1180 | 一键标记当前 query 为 badcase，写 `outputs/badcases.jsonl` |
| `tab_live / tab_offline / tab_badcase` | ~1363 | 三大主 tab：实时调用 / 离线浏览 / Badcase 列表 |

---

## 6. 商品 ↔ 文案配对规则（`_build_item_copy_pairs`）

V5 主链路返回 n 个召回商品和 m 条 LLM 候选文案。看板需要把它们配对成 n 个"商品 + 文案"展示卡。规则（按优先级）：

1. **linked**（直接引用）— 如果某条候选文案 `used_evidence` 列表里包含该商品 id，且该候选 `rank_score` 在引用集合里最高，直接绑给它
2. **distributed**（按 rank 分配）— 商品没被任何候选直接引用时，从尚未被使用的候选里按 rank_score 顺序轮流抓一条，保证 m ≥ n 时 n 个商品拿到 n 条**不同**文案
3. **fallback**（兜底重复）— 候选不够 n 条时，剩余商品全部用 `ranked[0]` 兜底
4. **none**（无候选）— 一条候选都没有时，text 为空

这套规则解决了"3 张卡显示同一句文案"的早期 bug。

---

## 7. 缓存与性能

- `@st.cache_resource load_runtime` — V5 runtime 上下文（语料/排序/用户/配对）只加载一次，所有后续 query 复用
- `@st.cache_data run_query` — 同 (query, provider, top_k, candidate_count) 命中缓存，秒回（结果条显示 `cache hit`）
- 多 query 并发 — `ThreadPoolExecutor` 跑批多个 query

---

## 8. Badcase 收集

调试模式下每次结果旁有"标记为 badcase"表单，提交后追加写到 `outputs/badcases.jsonl`，每行一个 JSON：

```json
{
  "marked_at": "2026-05-24T...",
  "query": "...",
  "provider": "...",
  "severity": "P0/P1/P2",
  "tags": ["..."],
  "reason": "...",
  "final_ad_copy": "...",
  "intent": {...},
  "validator_summary": {...}
}
```

可在 `tab_badcase` 中浏览、下载完整 jsonl。

---

## 9. 安全 & 红线

- **API key 绝不入库**：DeepSeek key 只能放环境变量或本地未提交的 `run.bat`，不写代码、不写日志、不入 badcases.jsonl。当前 `run.bat` 已做脱敏。**如真实 key 曾被提交，必须在 DeepSeek 控制台立即吊销并换新**。
- **不动 V5 主代码**：`<V5_PROJECT_ROOT>\ad_generation\` 内任何文件均不修改。需要新能力时在看板侧（`app.py`）加封装层。
- **不分叉 V6**：本仓库只是 V5 的"显示器"，业务逻辑改动一律在 V5 主项目讨论。

---

## 10. 后续可做（未排期）

- [ ] 移除 `local_fake` provider 限定，接入更多线上 LLM（Claude / Qwen / GLM）
- [ ] Badcase 表单加"已修复"状态字段，跟踪修复闭环
- [ ] 离线浏览支持多 JSONL 合并 diff（同 query 不同模型对比）
- [ ] 把 `_build_item_copy_pairs` 写成独立 module + 单测
- [ ] V5 链路如果开始返回流式响应，需要看板支持 `st.write_stream`

---

## 11. 联系上下文

- V5 主项目交接文档：`<V5_PROJECT_ROOT>\V5_HANDOFF_README.md（私有,不公开）`
- 主入口文件：`app.py` （单文件，所有逻辑都在里面）
