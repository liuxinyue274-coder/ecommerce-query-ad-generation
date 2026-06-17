# demo_v5_eval_min120_baselines 说明

## 1. 文件对应关系

- deepseek 参考批次：`D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_deepseek.jsonl`
- baseline 主文件：`D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_baselines.jsonl`
- baseline 统计文件：`D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_baselines_stats.json`
- baseline 说明文件：`D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_baselines_notes.md`

## 2. 运行目标

- 使用 `demo_v5_eval_min120_deepseek.jsonl` 中已经固定好的同一批 120 条 query。
- 在不改变 query 批次和样本顺序的前提下，回放其他 baseline，补充对照结果。
- 本文件对应的 baseline 包括：
  - `baseline_template_top1`
  - `baseline_template_personalized`
  - `baseline_summary_topk`
  - `baseline_llm_topk_local_fake`
  - `baseline_v5_local_fake`

## 3. baseline 口径

- `baseline_template_top1`：旧版 `template` 模式，`top_k=1`。
- `baseline_template_personalized`：旧版 `template` 模式，优先使用 train_pairs 中同 query 的 `user_id/session_id`。
- `baseline_summary_topk`：旧版 `summary` 模式，`top_k=3`。
- `baseline_llm_topk_local_fake`：旧版 `llm` 模式，配置文件为 `D:\hllm_env\KuaiSearch-main\ad_generation\llm_config.mock.json`，实际 provider 是 mock/local。
- `baseline_v5_local_fake`：V5 全链路，provider 固定为 `local_fake`。
- `baseline_v5_sft_local`：当前未执行，因为 `SFT_MODEL_PATH` 不可见。

## 4. 本次运行摘要

- 样本数：`120`
- `SFT_MODEL_PATH` 当前进程可见：`False`

| baseline | fallback_ratio | 备注 |
| --- | ---: | --- |
| baseline_template_top1 | 0.0 |  |
| baseline_template_personalized | 0.0 |  |
| baseline_summary_topk | 0.3583 |  |
| baseline_llm_topk_local_fake | 0.9917 | llm_provider={'mock': 120} |
| baseline_v5_local_fake | 1.0 | provider={'local_fake': 120} |

## 5. 复现命令

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -X utf8 D:\hllm_env\KuaiSearch-main\ad_generation\eval_v5\build_eval_baselines_from_deepseek.py --source_eval_path D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_deepseek.jsonl --output_path D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_baselines.jsonl --stats_path D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_baselines_stats.json --notes_path D:\hllm_env\KuaiSearch-main\outputs\demo_v5_eval_min120_baselines_notes.md
```
