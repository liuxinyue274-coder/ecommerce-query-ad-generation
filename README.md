# KuaiSearch LLM Ad Creative System

基于大语言模型的电商搜索广告 Query 意图增强与动态创意生成系统。

本项目基于 KuaiSearch 电商搜索场景，扩展了广告创意生成链路：从用户 Query 出发，结合商品证据、用户行为和排序信号，构建可解释、可校验、可回退的 LLM 广告文案生成流程，并提供 Streamlit 可视化展示。

## System Flow

```text
Query
  -> Intent Enhancement
  -> Evidence Selection
  -> Prompt Construction
  -> LLM Provider
  -> Parser
  -> Validator
  -> Ranker
  -> Rewriter
  -> Final Ad Copy
```

## Current Status

- V5 链路已完成：`ad_generation/v5_dynamic_creative.py`、`v5_runtime.py`、`v5_infer_adapter.py`
- Provider 支持：`local_fake`、`deepseek_chat`
- `sft_local` 已预留入口，用于后续本地 SFT 模型接入
- 已生成 demo 输出，并提供可视化浏览页面
- `demo_app/` 中保留小样例文件，适合 GitHub 与 Streamlit 展示
- 不包含完整 KuaiSearch 数据、API Key 或模型权重

## Repository Layout

```text
.
├── ad_generation/          # V5 广告生成、LLM provider、评测与 SFT 数据构建
├── demo/                   # 小规模原始 demo 样例
├── demo_app/               # Streamlit 展示应用
│   ├── app.py
│   ├── outputs/
│   │   └── demo_v5_outputs_sample.jsonl
│   ├── docs/
│   ├── public_snapshot/
│   ├── requirements.txt
│   └── .streamlit/
├── docs/
│   └── DEPLOYMENT.md
├── outputs/                # 本地生成输出，默认不提交大文件
├── scripts/                # 原 KuaiSearch 训练/处理脚本
├── requirements.txt        # 主链路依赖
└── README.md
```

完整数据目录如 `items_lite/`、`users_lite/`、`rank_lite/`、`recall_lite/` 默认被 `.gitignore` 排除。需要运行完整实时链路时，请在本地准备这些数据。

## Quick Start

安装主项目依赖：

```bash
python -m pip install -r requirements.txt
```

运行 V5 demo pipeline：

```bash
python -X utf8 ad_generation/demo_v5_pipeline.py --limit 20 --provider local_fake
```

如需使用 DeepSeek：

```bash
set DEEPSEEK_API_KEY=YOUR_API_KEY_HERE
python -X utf8 ad_generation/demo_v5_pipeline.py --limit 20 --provider deepseek_chat
```

不要把 API Key 写入代码、日志或提交记录。

## Demo App

安装展示应用依赖：

```bash
python -m pip install -r demo_app/requirements.txt
```

启动展示页面：

```bash
streamlit run demo_app/app.py
```

展示应用默认读取：

```text
demo_app/outputs/demo_v5_outputs_sample.jsonl
```

实时调用模式默认以仓库根目录作为 `KS_PROJECT_ROOT`。如果主项目在其他路径，可设置环境变量：

```bash
set KS_PROJECT_ROOT=D:\path\to\KuaiSearch-main
```

## Output Fields

主要输出字段：

| Field | Description |
|---|---|
| `query` | 用户搜索词 |
| `final_ad_copy` | 最终广告文案 |
| `ranked_candidates` | 候选文案及排序分数 |
| `provider` | 实际使用的 LLM provider |
| `fallback` | 是否发生回退 |
| `validator` | 文案校验结果与问题摘要 |
| `evidence` | 被选中的商品、类目、行为与排序证据 |

## Evaluation Plan

当前评测方案以自动指标为主，后续可加入人工标注：

- BERTScore：衡量生成文案与参考文案的语义一致性
- PPL：衡量语言流畅度
- 证据覆盖率：文案是否覆盖被选商品、品牌、类目等证据
- 多样性：候选文案之间的表达差异和品类覆盖
- fallback rate：不同 provider 的失败与回退比例
- provider 对比：`local_fake`、`deepseek_chat`、后续 `sft_local` 的质量、稳定性和延迟对比

## Notes

- 本仓库不包含完整 KuaiSearch 原始数据。
- 本仓库不包含 API Key。
- 本仓库不包含模型权重、checkpoint 或大体积缓存。
- 大文件与本地生成结果默认由 `.gitignore` 排除，只保留展示所需的小样例文件。

## Deployment

部署说明见 [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)。
