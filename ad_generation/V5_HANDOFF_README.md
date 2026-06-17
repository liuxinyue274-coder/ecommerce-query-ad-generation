# V5 广告生成交接 README

这份文档面向后续负责评测和前端系统搭建的同学，不是普通用户手册。重点是交代当前 V5 动态广告创意生成工作的完成状态、代码入口、运行方式、输出结构、已知限制和后续可接续的工作方向。

## 1. 当前完成状态总览

当前已经完成的内容：

- V5 动态广告创意生成框架
- `local_fake` 本地 mock provider
- `deepseek_chat` 真实 LLM API provider
- `sft_local` 本地 SFT / LoRA provider 预留入口
- DeepSeek 真实 API 生成链路验证
- 真实 SFT 数据构造脚本
- SFT dry-run 训练链路验证
- runtime cache，避免 demo 重复扫描数据
- `parser -> validator -> ranker -> rewriter -> final_ad_copy` 后处理链路

需要明确说明的几点：

- 当前阶段不强制训练 LoRA。
- `sft_local` 当前是预留的本地模型入口；如果没有配置 `SFT_MODEL_PATH`，会优雅 fallback，不会中断 V5。
- 真正调用 DeepSeek API 的 provider 只有 `deepseek_chat`。
- 当前已经把“模板/摘要规则基线 + LLM provider + 多候选后处理”串起来，形成一个可运行、可观察、可扩展的动态创意生成原型。

## 2. V5 总体链路

总体链路如下：

```text
Query
→ query_normalization
→ intent_enricher
→ evidence_selector
→ user_profile_builder
→ style_retriever
→ prompt_builder
→ llm_provider
   ├─ local_fake
   ├─ deepseek_chat
   └─ sft_local
→ output_parser
→ copy_validator
→ copy_ranker
→ copy_rewriter
→ final_ad_copy
```

各步骤作用：

- `query_normalization`
  统一 query 文本，抽取基础 token、长度、是否含场景词/型号词等弱信号。
- `intent_enricher`
  基于 query 和检索结果补充意图摘要，例如品类、场景、人群、购买关注点、属性提示。
- `evidence_selector`
  从检索候选中选出更适合广告创意生成的证据，形成 anchor item、selected evidence、common selling points、fact block。
- `user_profile_builder`
  从 `users_lite` 和最近行为中拼弱画像，用于轻量个性化，不追求强个性化推荐。
- `style_retriever`
  给出风格约束、负向规则和样例文案，帮助 prompt 更稳定地产生自然、短促、非标题复述的广告表达。
- `prompt_builder`
  把 query、intent、evidence、user profile、style examples 组装成统一 prompt。
- `llm_provider`
  选择具体生成 provider：
  - `local_fake`：本地 mock 生成，便于无 API 环境演示。
  - `deepseek_chat`：真实 DeepSeek API。
  - `sft_local`：预留给未来本地 SFT / LoRA 模型。
- `output_parser`
  解析 provider 返回内容，转成统一候选结构。
- `copy_validator`
  对候选做基础合法性、相关性、长度、风险点检查。
- `copy_ranker`
  基于有效性、自然度、相关性、简洁度等子分数对候选排序。
- `copy_rewriter`
  对排序结果做必要改写，或在候选不可用时触发 summary/template fallback。
- `final_ad_copy`
  输出最终广告文案，供评测、前端展示和后续训练数据构造使用。

## 3. 与旧版本 template / summary 的区别

| 方案 | 主要机制 | 优点 | 局限 |
| --- | --- | --- | --- |
| `template` | top-1 商品证据 + 模板文案 | 稳定、成本低、容易复现 | 容易像商品标题复述，表达僵硬 |
| `summary` | top-k 商品摘要后生成方向型文案 | 比 template 更有方向感 | 证据稀疏时 fallback 多，创意和自然度有限 |
| `V5` | 意图增强 + 证据选择 + 生成模型 + 多候选 + 校验排序重写 | 更适合动态创意生成，可接真实 API 或本地模型 | 链路更复杂，需要关注 provider 状态、fallback 和后处理结果 |

## 4. 主要代码文件说明

以下以当前项目实际文件路径为准：

- [ad_generation/v5_dynamic_creative.py](D:\hllm_env\KuaiSearch-main\ad_generation\v5_dynamic_creative.py)
  V5 主链路，串联 `intent / evidence / user / style / provider / parser / validator / ranker / rewriter`。

- [ad_generation/v5_runtime.py](D:\hllm_env\KuaiSearch-main\ad_generation\v5_runtime.py)
  runtime cache 相关实现，缓存 evidence bundle，避免 demo 每条 query 重复扫描 `items_lite / rank_lite / train_pairs`。

- [ad_generation/llm_generator.py](D:\hllm_env\KuaiSearch-main\ad_generation\llm_generator.py)
  provider 统一入口，当前支持 `local_fake`、`deepseek_chat`、`sft_local`。

- [ad_generation/demo_v5_pipeline.py](D:\hllm_env\KuaiSearch-main\ad_generation\demo_v5_pipeline.py)
  V5 demo 脚本。适合前端同学快速联调输出，也适合评测同学先观察完整结构。

- [ad_generation/build_real_sft_data.py](D:\hllm_env\KuaiSearch-main\ad_generation\build_real_sft_data.py)
  基于 V5 上下文和 `deepseek_chat / local_fake` 构造真实 SFT 数据，支持进度日志、断点续跑、超时、重试、即时落盘。

- [ad_generation/train_sft_ad_copy.py](D:\hllm_env\KuaiSearch-main\ad_generation\train_sft_ad_copy.py)
  SFT dry-run / 训练链路验证脚本。当前默认先做数据格式和训练文本拼接验证，不强制启动真实训练。

- `ad_generation/data` 与 `outputs`
  - `ad_generation/data/train_pairs.jsonl`：V5 主要配对数据之一。
  - `outputs/`：demo、SFT 数据、preview、train config 等输出目录。

- [ad_generation/README.md](D:\hllm_env\KuaiSearch-main\ad_generation\README.md)
  旧 README 与补充说明，保留整体背景和一些脚本说明。

## 5. Provider 说明

### `local_fake`

- 不依赖外部 API。
- 用于本地可复现 demo。
- 当前默认 fallback provider。

运行：

```bash
python -X utf8 ad_generation/demo_v5_pipeline.py --limit 5 --provider local_fake
```

### `deepseek_chat`

- 调用 DeepSeek API。
- 用于真实 LLM 效果验证。
- 用于构造真实 SFT 数据。
- 需要环境变量：
  - `DEEPSEEK_API_KEY`
  - `DEEPSEEK_BASE_URL`，默认 `https://api.deepseek.com`
  - `DEEPSEEK_MODEL`，默认 `deepseek-chat`

运行：

```bash
python -X utf8 ad_generation/demo_v5_pipeline.py --limit 5 --provider deepseek_chat
```

说明：

- 如果没有 `DEEPSEEK_API_KEY`，会自动 fallback 到 `local_fake`。
- fallback 信息会记录在 `status / metadata / provider bundle` 中，评测时要一并记录。

### `sft_local`

- 用于后续加载本地 SFT / LoRA 模型。
- 不调用 DeepSeek API。
- 需要：
  - `SFT_MODEL_PATH`
  - `SFT_ADAPTER_PATH`，可选

运行：

```bash
python -X utf8 ad_generation/demo_v5_pipeline.py --limit 5 --provider sft_local
```

说明：

- 如果没有配置 `SFT_MODEL_PATH`，会返回 `skipped_no_sft_model` 并 fallback。
- 当前阶段它是一个预留入口，不等于已经有可用 LoRA 模型。

## 6. 常用运行命令

1. 跑 V5 local demo：

```bash
python -X utf8 ad_generation/demo_v5_pipeline.py --limit 5 --provider local_fake
```

2. 跑 DeepSeek demo：

```bash
python -X utf8 ad_generation/demo_v5_pipeline.py --limit 5 --provider deepseek_chat
```

3. 构造真实 SFT 数据，小批量：

```bash
python -X utf8 ad_generation/build_real_sft_data.py --target_count 20 --max_source 80 --provider deepseek_chat --timeout 30 --max_retries 1 --resume
```

4. 构造真实 SFT 数据，使用 `local_fake`：

```bash
python -X utf8 ad_generation/build_real_sft_data.py --target_count 20 --max_source 80 --provider local_fake --resume
```

5. SFT dry-run：

```bash
python -X utf8 ad_generation/train_sft_ad_copy.py --data outputs/real_sft_ad_copy.jsonl --dry_run
```

6. `sft_local` 验证：

```bash
python -X utf8 ad_generation/demo_v5_pipeline.py --limit 2 --provider sft_local
```

## 7. 输出文件说明

- `outputs/demo_v5_outputs.jsonl`
  V5 demo 输出，包含 `query`、完整 `result`、候选文案、排序、最终文案等。

- `outputs/real_sft_ad_copy.jsonl`
  由 `deepseek_chat / local_fake` 生成并经过 `validator / ranker / rewriter` 过滤后的 SFT 数据。

- `outputs/real_sft_ad_copy_preview.md`
  SFT 数据预览文件，适合人工快速查看样本质量。

- `outputs/sft_train_config.json`
  SFT dry-run 或训练配置输出。

- `outputs/sft_train_preview.md`
  训练文本预览，帮助检查消息格式和训练样本拼接结果。

## 8. V5 输出字段说明

后续评测和前端同学最应该关注以下字段。注意：有些是“概念字段”，在当前实现中是嵌套字段，不一定是顶层同名字段。

### 核心输入与上下文字段

- `query`
  demo 输出 JSONL 顶层字段。

- `normalized_query`
  当前位于 `result.query_normalization.normalized_query`。

- `intent`
  当前建议读取 `result.intent_enricher`，其中重点看：
  - `intent_summary`
  - `scenario`
  - `audience`
  - `purchase_focus`

- `selected_evidence`
  当前位于 `result.evidence_selector.selected_evidence_items`。

- `user_profile`
  当前位于 `result.user_profile_builder`，其中重点看 `persona_summary` 和 `recent_behavior_titles`。

- `retrieved_style_examples`
  当前位于 `result.style_retriever.style_examples`。

### Provider 与状态字段

- `provider`
  当前位于 `result.provider`，表示实际生效 provider。

- `model`
  当前位于 `result.model`。

- `llm_status`
  当前位于 `result.llm_status`，用于区分 `provider_ok / local_fake_json_ok / skipped_no_sft_model / skipped_no_api_key` 等状态。

- `fallback_triggered`
  当前没有单独顶层字段，建议读取 `result.llm_provider.fallback_used`。

- `fallback_reason`
  当前位于 `result.llm_provider.fallback_reason`。

- `cache_status`
  当前位于 `result.runtime_cache.cache_status`。

### 生成与后处理字段

- `raw_llm_output`
  当前位于 `result.llm_provider.raw_generations`。

- `parsed_candidates`
  当前位于 `result.llm_output_parser.parsed_candidates`。

- `validated_candidates`
  当前位于 `result.copy_validator.validated_candidates`。

- `validator_result`
  当前建议读取：
  - `result.copy_validator.validator_summary`
  - 每条 candidate 内的 `is_valid / issues`

- `ranked_candidates`
  当前位于 `result.copy_ranker.ranked_candidates`。

- `score`
  当前建议读取 candidate 级别的 `rank_score`。

- `rewritten_candidates`
  当前没有同名字段。重写结果主要体现在：
  - `result.copy_rewriter.rewritten`
  - `result.copy_rewriter.rewrite_reason`
  - `result.copy_rewriter.final_candidate`

- `best_copy`
  当前没有单独顶层字段。概念上可对应：
  - 排序阶段的 `result.copy_ranker.top_candidate`
  - 最终输出阶段的 `result.final_ad_copy.final_ad_copy`

- `final_ad_copy`
  当前位于 `result.final_ad_copy.final_ad_copy`，前端与评测都应优先使用。

给前端的建议：

- 前端优先展示 `final_ad_copy` 和 `ranked_candidates`。

给评测的建议：

- 评测优先使用 `final_ad_copy`、`ranked_candidates`、`validator_result`、`score`、`provider`、`fallback_triggered`。

## 9. SFT 数据说明

- 旧 `sft_sample.jsonl` 中的 assistant 多为模板伪标签，不适合作为最终高质量监督数据。
- 当前新增 [build_real_sft_data.py](D:\hllm_env\KuaiSearch-main\ad_generation\build_real_sft_data.py)，使用 `deepseek_chat` 优先生成候选，再经过 `parser / validator / ranker / rewriter` 过滤。
- `outputs/real_sft_ad_copy.jsonl` 属于高质量伪标注数据，不是人工金标。
- 当前已完成 SFT dry-run，证明训练格式和训练文本构造链路可用。
- LoRA 真实训练不是本阶段必须项；后续如果需要，可以基于 [train_sft_ad_copy.py](D:\hllm_env\KuaiSearch-main\ad_generation\train_sft_ad_copy.py) 继续往下做。

## 10. 交给评测同学的建议

评测同学可以从两种方式拿结果：

- 直接读取 `outputs/demo_v5_outputs.jsonl`
- 直接调用 [demo_v5_pipeline.py](D:\hllm_env\KuaiSearch-main\ad_generation\demo_v5_pipeline.py) 获取最新结果

建议评测维度：

- Query 相关性
- 证据支持度
- 个性化显性程度
- 自然度
- 创意多样性
- 合规风险
- fallback 比例
- provider 对比：`local_fake vs deepseek_chat vs sft_local`

建议对比 baseline：

- `template top-1`
- `summary top-k`
- `V5 local_fake`
- `V5 deepseek_chat`
- `V5 sft_local`，如果后续有本地模型

建议记录：

- provider
- llm_status
- fallback_used / fallback_reason
- rank_score
- final_ad_copy

否则会把“真实模型效果”和“fallback 效果”混在一起。

## 11. 交给前端同学的建议

前端可以先对接 [demo_v5_pipeline.py](D:\hllm_env\KuaiSearch-main\ad_generation\demo_v5_pipeline.py) 的输出 JSONL，或者在此基础上封装一个简单接口。

建议前端展示字段：

- Query
- `final_ad_copy`
- `ranked_candidates` top3
- intent 摘要
- selected evidence 摘要
- provider 状态
- fallback 状态
- 风险提示 / validator 结果

建议交互：

- 输入 Query
- 选择 provider：`local_fake / deepseek_chat / sft_local`
- 展示候选文案
- 展示最终文案
- 展示证据来源和安全提示

前端第一阶段不需要把所有中间模块都可视化，只要把“最终文案 + top3 候选 + provider/fallback 状态 + 证据摘要”串起来即可。

## 12. 已知限制

- `local_fake` 是 mock，不代表真实模型能力。
- `deepseek_chat` 调用依赖 API key 和余额。
- SFT 数据是模型生成的高质量伪标注，不是人工金标。
- 当前 `user_profile` 仍是弱画像，不是完整用户建模。
- 当前没有完成完整离线评测。
- 当前没有真正训练 LoRA。
- `sft_local` 当前主要是预留入口。

## 13. 后续工作建议

- 扩大 DeepSeek 生成 SFT 数据到 50-100 条或更多。
- 人工审核一部分 SFT 数据，提高监督质量。
- 做 V5 与 `template / summary` 的离线对比。
- 接入前端 demo。
- 有资源时做 LoRA 小步训练。
- 优化 `prompt_builder`。
- 加入人工评审或 LLM judge。

## 14. 一条完整样例

示例 query：

```text
运动手环
```

示例 intent 摘要：

```text
中性品牌 / 智能手环设备 / sports
```

示例 selected evidence 摘要：

- 新款智能手表/运动手环，可充电，包含计步相关信息
- 候选证据里能看到“心率 / 计步 / 运动”这类共同卖点

示例 provider：

```text
local_fake
```

示例 ranked candidates top3：

1. 跑步通勤都能戴，心率睡眠随手看。
2. 日常佩戴不累赘，运动睡眠都能记。
3. 通勤运动都适合，健康数据随手看。

示例 final_ad_copy：

```text
跑步通勤都能戴，心率睡眠随手看。
```

## 15. 注意事项

- 不要把 `sft_local` 理解成 DeepSeek API，它是本地微调模型入口。
- DeepSeek API 只走 `deepseek_chat` provider。
- 不要在代码或文档里泄露 API key。
- 如果只是做前端展示，优先使用 `local_fake` 或 `deepseek_chat`。
- 如果做评测，需要记录 provider 和 fallback 状态。

## 附：建议交接顺序

如果后续是多人并行接手，建议顺序如下：

1. 评测同学先用 `demo_v5_pipeline.py` 和 `outputs/demo_v5_outputs.jsonl` 跑一轮样本观察。
2. 前端同学先只对接 `final_ad_copy + ranked_candidates + provider/fallback 状态`。
3. 若需要模型化推进，再扩充 `real_sft_ad_copy.jsonl`，考虑小规模 LoRA 训练。
4. 训练完成后，再把本地模型挂到 `sft_local`，不需要重写 V5 主链路。
