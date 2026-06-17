# demo_v5_eval_min120_deepseek 抽样明细说明

## 1. 文件对应关系

- 评测主文件：`D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_deepseek.jsonl`
- 统计文件：`D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_deepseek_stats.json`
- 抽样说明：`D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_deepseek_sampling_notes.md`
- 抽样脚本：`D:\hllm_env\KuaiSearch-main\ad_generation\eval_v5\build_v5_eval_set.py`

## 2. 抽样来源

- 候选池文件：`D:\hllm_env\KuaiSearch-main\ad_generation\data\train_pairs.jsonl`
- 去重口径：按 `normalized_query` 去重，同一个标准化 query 只保留第一次出现的样本
- top-k 口径：`top_k=3`
- 候选数口径：`candidate_count=5`
- 随机种子：`seed=42`

## 3. 分层规则

### 3.1 Query 长度桶

- `short`：`len(normalized_query) <= 6`
- `medium`：`7 <= len(normalized_query) <= 15`
- `long`：`len(normalized_query) >= 16`

### 3.2 Evidence 强度桶

- 命中字段：`item_title / category_path / brand_name(非占位) / seller_name / ranking_signal / relevance_signal`
- 分数公式：`evidence_score = 命中字段总数 / (top_k * 6)`
- `high`：`evidence_score >= 0.75`
- `low`：`evidence_score < 0.75`

## 4. 目标配额

| 分层格子 | 配额 |
| --- | ---: |
| short-high | 20 |
| short-low | 20 |
| medium-high | 30 |
| medium-low | 30 |
| long-high | 10 |
| long-low | 10 |

## 5. 候选池覆盖情况

| 分层格子 | 候选量 |
| --- | ---: |
| short-high | 29136 |
| short-low | 44809 |
| medium-high | 33454 |
| medium-low | 56813 |
| long-high | 918 |
| long-low | 1941 |

## 6. 本次实际抽样结果

- 总样本数：`120`
- 请求 provider：`deepseek_chat`
- 实际 provider 分布：`{'deepseek_chat': 120}`
- model 分布：`{'deepseek-chat': 120}`
- llm_status 分布：`{'provider_ok': 120}`

| 分层格子 | 实际条数 |
| --- | ---: |
| short-high | 20 |
| short-low | 20 |
| medium-high | 30 |
| medium-low | 30 |
| long-high | 10 |
| long-low | 10 |

长度分布：
- `short`：40
- `medium`：60
- `long`：20

证据强度分布：
- `high`：60
- `low`：60

抽中样本类目分布：
- `digital`：10
- `apparel`：52
- `food`：16
- `health`：13
- `misc`：29

## 7. 环境与配置说明

- `llm_config_path`：`D:\hllm_env\KuaiSearch-main\ad_generation\llm_config.deepseek_chat.json`
- `DEEPSEEK_API_KEY` 当前进程可见：`True`
- `SFT_MODEL_PATH` 当前进程可见：`False`

## 8. 复现命令

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -X utf8 D:\hllm_env\KuaiSearch-main\ad_generation\eval_v5\build_v5_eval_set.py --provider deepseek_chat --output_path D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_deepseek.jsonl --stats_path D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_deepseek_stats.json --notes_path D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_deepseek_sampling_notes.md
```
