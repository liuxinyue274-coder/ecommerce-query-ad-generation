# demo_v5_eval_min120 抽样明细说明

## 1. 文件对应关系

- 评测主文件：`outputs/demo_v5_eval_min120.jsonl`
- 统计文件：`outputs/demo_v5_eval_min120_stats.json`
- 抽样脚本：`ad_generation/eval_v5/build_v5_eval_set.py`

本说明对应 2026-05-23 这次实际生成的 120 条 V5 评测集。

## 2. 抽样来源

样本候选池来自：

- `ad_generation/data/train_pairs.jsonl`

抽样时不是直接按原始 `query` 生搬硬套，而是先做一次轻清洗：

- 对每条记录读取原始 `query`
- 调用现有项目里的 `normalize_query(...)`
- 用 `normalized_query` 作为后续分层和去重依据

去重规则：

- 同一个 `normalized_query` 只保留第一次出现的样本

本次候选池规模：

- 去重后可用 `normalized_query` 总数：`167071`

## 3. 分层规则

### 3.1 Query 长度桶

按 `normalized_query` 的字符长度分桶：

- `short`：`len(normalized_query) <= 6`
- `medium`：`7 <= len(normalized_query) <= 15`
- `long`：`len(normalized_query) >= 16`

### 3.2 Evidence 强度桶

先取每条候选样本的前 `top_k=3` 个证据商品，逐个统计下面 6 个字段是否可用：

- `item_title`
- `category_path`
- `brand_name`，且不能是占位值，如 `无品牌/其他/unknown/null`
- `seller_name`
- `ranking_signal`
- `relevance_signal`

单条样本的证据分数计算方式：

```text
evidence_score = 命中字段总数 / (top_k * 6)
               = 命中字段总数 / 18
```

强度分桶规则：

- `high`：`evidence_score >= 0.75`
- `low`：`evidence_score < 0.75`

## 4. 配额设计

目标总样本量：`120`

配额如下：

| 分层格子 | 配额 |
| --- | ---: |
| short-high | 20 |
| short-low | 20 |
| medium-high | 30 |
| medium-low | 30 |
| long-high | 10 |
| long-low | 10 |

## 5. 候选池覆盖情况

脚本在正式抽样前，先统计每个分层格子的候选量。结果如下：

| 分层格子 | 候选量 |
| --- | ---: |
| short-high | 29136 |
| short-low | 44809 |
| medium-high | 33454 |
| medium-low | 56813 |
| long-high | 918 |
| long-low | 1941 |

结论：

- 6 个格子都远大于目标配额
- `long-high` 和 `long-low` 也都足够，不存在“长 query 抽不满”的问题

## 6. 实际抽样过程

脚本里的实际抽样逻辑是：

1. 先按 `query_bucket + evidence_strength` 把候选池分成 6 个桶。
2. 对每个桶分别使用固定随机种子 `seed=42` 做 `shuffle`。
3. 从每个桶里按配额取前 `N` 条。
4. 取出的样本再按 `(query_category, normalized_query)` 排序，便于结果更稳定、可读。
5. 将 6 个桶按固定顺序拼接输出：
   `short-high -> short-low -> medium-high -> medium-low -> long-high -> long-low`

说明：

- `query_category` 不是抽样配额条件，只是抽中后保留的分析字段
- 本次没有对 `digital/apparel/food/health/misc` 再做强制平衡

## 7. 本次实际抽样结果

抽样完成后，120 条样本的分层结果与目标完全一致：

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

抽中样本的 `query_category` 分布：

- `apparel`：52
- `digital`：10
- `food`：16
- `health`：13
- `misc`：29

这里的类目分布说明：

- 这套评测集是“长度 + 证据强度”的轻分层方案
- 类目只是保留为分析字段，所以最终分布仍然贴近原始数据，而不是人为强行均衡

## 8. 生成阶段说明

抽样完成后，脚本会把这 120 条 `query` 再送入现有 V5 主链路，补齐：

- `intent`
- `selected_evidence`
- `raw_llm_output`
- `parsed_candidates`
- `validated_candidates`
- `ranked_candidates`
- `rewritten_candidates`
- `best_copy`
- `final_ad_copy`
- `fallback_triggered`
- `fallback_reason`
- `cache_status`

本次运行环境说明：

- `provider` 请求值：`local_fake`
- `DEEPSEEK_API_KEY`：缺失
- `SFT_MODEL_PATH`：缺失

因此，本次评测集中实际生成字段全部来自：

- `provider = local_fake`
- `model = mock-dynamic-creative`

## 9. 复现命令

本次评测集对应的生成命令如下：

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -X utf8 D:\hllm_env\KuaiSearch-main\ad_generation\eval_v5\build_v5_eval_set.py `
  --provider local_fake `
  --output_path D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120.jsonl `
  --stats_path D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_stats.json
```

如果后续环境补齐了 `DEEPSEEK_API_KEY`，可以只改 `--provider deepseek_chat`，沿用同一套分层抽样脚本再跑一版可比结果。
