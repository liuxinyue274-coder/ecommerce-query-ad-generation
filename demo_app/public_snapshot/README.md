# V5 广告生成可视化看板

基于 Streamlit 的 V5 动态广告创意生成系统调试看板。

输入一个 Query，实时展示完整链路的中间结果：

```
Query → 意图结构 → 召回的广告商品 → 候选广告文案 → 最终广告文案
```

附带：provider/fallback 状态、validator 风险提示、badcase 一键标记落盘。

---

## 功能

- **实时调用**：输入任意 Query，选择 provider（`local_fake` / `deepseek_chat` / `sft_local`），一键运行 V5 主链路。
- **离线浏览**：读取 `outputs/demo_v5_outputs.jsonl` 中已生成的样本，逐条查看。
- **Badcase 收集**：每条结果旁有"标记为 badcase"按钮，可填原因 / 严重程度 / 标签，追加写入 `outputs/badcases.jsonl`。
- **完整 result 下载**：随时下载某次调用的完整 JSON 结果，方便贴评测、提 issue。

## 目录结构

```
.\
├── app.py              # Streamlit 主程序
├── requirements.txt    # Python 依赖
├── run.bat             # Windows 一键启动
├── README.md           # 本文档
└── outputs\            # 看板自身的输出目录
    └── badcases.jsonl  # 标记的 badcase 持续追加
```

看板**不修改**项目主代码，只通过 `sys.path` 引入 `<V5_PROJECT_ROOT>\ad_generation` 下的模块：

- `v5_dynamic_creative.v5_llm_dynamic_creative`：V5 主链路入口
- `v5_runtime.build_v5_runtime_context`：构建 runtime cache（语料 / 排序 / 用户 / 配对）

## 安装与启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> 看板本身只额外依赖 `streamlit`。其余 V5 链路需要的依赖（如 numpy/torch、`requests` 等）应该已经在 `V5 主项目` 那边装好。如果直接 import 报 `ModuleNotFoundError`，按报错信息补装即可。

### 2. 启动

**方式一：双击 `run.bat`**（推荐 Windows 用户）

如果 V5 主项目 不在默认的 `<V5_PROJECT_ROOT>`，先用记事本打开 `run.bat` 改 `KS_PROJECT_ROOT`。

**方式二：命令行**

```bash
cd "."
streamlit run app.py
```

启动后浏览器自动打开 `http://localhost:8501`。

## 配置

通过环境变量配置（在 `run.bat` 里也可以设）：

| 变量 | 说明 | 默认 |
| --- | --- | --- |
| `KS_PROJECT_ROOT` | V5 主项目 的根路径 | `<V5_PROJECT_ROOT>` |
| `DEEPSEEK_API_KEY` | DeepSeek API key（仅 `deepseek_chat` provider 需要） | 无；缺省时会自动 fallback 到 `local_fake` |
| `DEEPSEEK_BASE_URL` | DeepSeek API base | `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | DeepSeek 模型名 | `deepseek-chat` |
| `SFT_MODEL_PATH` | 本地 SFT 模型目录（仅 `sft_local` 需要） | 无；缺省时优雅 fallback |

## 使用提示

- **首次运行慢**：第一次 query 会触发 `build_v5_runtime_context` 加载 `items_lite / rank_lite / users_lite / train_pairs`，需要 5–30 秒；后续请求复用 cache，秒级返回。
- **provider 选择**：日常调试用 `local_fake`（无 API、可复现）；想看真实模型效果用 `deepseek_chat`（需 API key）。
- **看板字段位置**全部对齐 `V5_HANDOFF_README.md（私有,不公开）` 第 8 节的字段路径，包括 `result.intent_enricher` / `result.evidence_selector.selected_evidence_items` / `result.copy_ranker.ranked_candidates` / `result.final_ad_copy.final_ad_copy` 等。

## Badcase 输出格式

`outputs/badcases.jsonl` 每行一条 JSON，字段：

```json
{
  "marked_at": "2026-05-24T12:34:56",
  "query": "运动手环",
  "provider": "local_fake",
  "severity": "medium",
  "reason": "标题复述商品名，不像广告文案",
  "tags": ["相关性", "自然度"],
  "final_ad_copy": "...",
  "ranked_top3": [{"candidate_id": "...", "strategy": "...", "rank_score": 0.87, "text": "..."}],
  "intent": {...},
  "validator_summary": {...},
  "llm_status": "...",
  "fallback_used": false,
  "fallback_reason": ""
}
```

后续要做 SFT 数据筛选、prompt 优化、validator 补规则时，直接读这个文件即可。

## 故障排查

| 现象 | 处理 |
| --- | --- |
| `ModuleNotFoundError: No module named 'v5_dynamic_creative'` | 检查 `KS_PROJECT_ROOT` 是否指向正确目录；该目录下需有 `ad_generation/v5_dynamic_creative.py` |
| 首次运行卡很久 | 正常，runtime cache 在加载语料；之后 sidebar 也可以「清空运行时缓存」手动重置 |
| `deepseek_chat` 实际跑成了 local_fake | 看顶部状态条，`Fallback=是` 说明触发了 fallback；通常是没设 `DEEPSEEK_API_KEY` 或网络/余额问题 |
| 离线浏览找不到 JSONL | 先在 V5 主项目 跑一次 `python -X utf8 ad_generation/demo_v5_pipeline.py --limit 5 --provider local_fake` 生成 |
