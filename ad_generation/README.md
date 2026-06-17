# ad_generation

## 模块说明

`ad_generation/` 是 KuaiSearch 上的最小广告生成原型，目前已经支持第三版摘要式广告文案 infer。

目标闭环：

```text
query -> 候选商品证据 -> 一条中文广告文案
```

实现原则：

- 不修改现有 `recall/`、`relevance/`、`ranking/` 代码
- 不调用外部 API
- 第一阶段先用模板文案跑通
- 第二阶段在 infer 中加入 `user_id / session_id` 弱个性化命中逻辑

## 文件说明

- `build_train_pairs.py`
  - 从 `corpus.jsonl`、`rank.jsonl`、可选 `users.jsonl` 中构造训练对
  - 输出 `ad_generation/data/train_pairs.jsonl`
- `prompt_builder.py`
  - 负责用户上下文、证据块、prompt、模板文案
  - 第二版会避免品牌重复，如 `iQOOiQOO 13...`
  - 第三版新增 `render_summary_copy(...)`，可结合 top-k 证据生成更自然的摘要式文案
- `llm_generator.py`
  - 负责 LLM 生成接口适配
  - 当前支持 `mock / local / api_openai_compatible / deepseek_chat / sft_local` provider 结构
  - 第一版默认用 `mock` 跑通链路，不写入任何真实 key
- `infer.py`
  - 负责 `query -> evidence -> prompt -> ad copy`
  - 第二版支持 `user_id / session_id` 和 query normalization
  - 第三版支持 `--copy_style summary|template`
  - 当前支持 `--copy_style llm`，失败时自动 fallback 到 `summary -> template`

## 输入文件说明

默认输入路径：

- `data/corpus.jsonl`
- `data/rank.jsonl`
- `data/users.jsonl`

如果本地先用 Lite 或 demo 数据验证，也可以传：

- `items_lite/train.jsonl`
- `rank_lite/train.jsonl`
- `users_lite/train.jsonl`
- `demo/items.jsonl`
- `demo/rank.jsonl`
- `demo/users.jsonl`

兼容字段读取包括但不限于：

- `title` / `item_title`
- `brand` / `brand_name`
- `seller` / `seller_name`
- `age` / `age_bucket`

## 输出文件说明

运行 `build_train_pairs.py` 后会生成：

- `ad_generation/data/train_pairs.jsonl`

每条记录至少包含：

- `query`
- `user_id`
- `session_id`
- `user_profile`
- `recent_behavior_titles`
- `evidence_items`
- `target_item`
- `target_copy`

## Infer 第二版参数

`infer.py` 支持以下参数：

- `--query`
- `--user_id`
- `--session_id`
- `--corpus_path`
- `--rank_path`
- `--users_path`
- `--pairs_path`
- `--top_k`
- `--copy_style`
- `--llm_config`
- `--mode`

其中：

- `user_id` 用来优先命中同用户样本，做弱个性化证据选择
- `session_id` 用来优先命中同 session 样本，做更细粒度的弱个性化证据选择
- `copy_style` 用来控制文案生成方式：
  - `summary`：默认值，优先结合 top-k 证据做摘要式文案
  - `template`：强制使用 top-1 模板文案
  - `llm`：先构造 LLM prompt，再走 `llm_generator.py`；若失败则回退到 `summary -> template`
- `llm_config` 用来指定 LLM provider 配置文件路径

## Query Normalization

第二版 infer 会同时保留：

- `raw_query`
- `normalized_query`

`normalized_query` 会做轻量清洗：

- 去掉首尾多余引号、单引号、井号和异常括号残片
- 合并多余空白
- 保留中文、英文、数字主体

这样可以让像 `'iqoo13`、`"氨糖软骨素钙片` 这类 query 更稳定地命中已有证据。

## 推理优先级

`infer.py` 的证据检索顺序如下：

1. 若传入 `user_id`，优先查 `train_pairs.jsonl` 中 query 匹配且 `user_id` 匹配的样本
2. 若传入 `session_id`，其次查 `train_pairs.jsonl` 中 query 匹配且 `session_id` 匹配的样本
3. 再回退到 query 级 `train_pairs.jsonl`
4. 再扫描 `rank.jsonl` 的正样本，优先 user/session 精确命中
5. 仍未命中时，优先尝试对接 `recall/BM25/bm25.py`
6. 如果旧 BM25 接口或依赖不适合直接调用，则回退到本地 lexical overlap

终端里会打印命中层级，例如：

- `train_pairs_user_exact`
- `train_pairs_session_exact`
- `train_pairs_query_exact`
- `rank_user_exact`
- `rank_session_exact`
- `rank_query_exact`
- `legacy_bm25`
- `lexical_overlap`

## 运行命令示例

### 1. 只传 query，使用第三版 summary 文案

```bash
python ad_generation/infer.py ^
  --query "'iqoo13" ^
  --corpus_path items_lite/train.jsonl ^
  --rank_path rank_lite/train.jsonl ^
  --users_path users_lite/train.jsonl ^
  --pairs_path ad_generation/data/train_pairs.jsonl ^
  --top_k 3 ^
  --copy_style summary
```

### 2. 传 query + user_id，使用第三版 summary 文案

```bash
python ad_generation/infer.py ^
  --query "'iqoo13" ^
  --user_id 1 ^
  --corpus_path items_lite/train.jsonl ^
  --rank_path rank_lite/train.jsonl ^
  --users_path users_lite/train.jsonl ^
  --pairs_path ad_generation/data/train_pairs.jsonl ^
  --top_k 3 ^
  --copy_style summary
```

### 3. 强制使用旧版 template 文案

```bash
python ad_generation/infer.py ^
  --query "'iqoo13" ^
  --corpus_path items_lite/train.jsonl ^
  --rank_path rank_lite/train.jsonl ^
  --users_path users_lite/train.jsonl ^
  --pairs_path ad_generation/data/train_pairs.jsonl ^
  --top_k 1 ^
  --copy_style template
```

### 4. 使用 mock provider 跑通 llm 模式

```bash
python ad_generation/infer.py ^
  --query "华为手环" ^
  --corpus_path items_lite/train.jsonl ^
  --rank_path rank_lite/train.jsonl ^
  --users_path users_lite/train.jsonl ^
  --pairs_path ad_generation/data/train_pairs.jsonl ^
  --top_k 3 ^
  --copy_style llm ^
  --llm_config ad_generation/llm_config.example.json
```

## 当前限制说明

- `template / summary` 仍然是规则生成
- 新增的 `llm` 模式目前默认通过 `mock` provider 测试链路，不代表真实大模型生成质量
- 不引入广告预算、广告主、ROI、出价等字段
- 不依赖已有 relevance / ranking checkpoint
- 第三版虽然能摘要 top-k 证据，但仍然是规则生成，不是训练好的生成模型
- 这版更像一个可运行的生成骨架，而不是最终广告系统

## 实验与评估

当前目录新增了一个轻量评估流程，用来把已有第一、二、三版能力整理成可对照展示的实验框架。

评估脚本：

- `evaluate_samples.py`
  - 从 `ad_generation/data/train_pairs.jsonl` 抽样生成小型评估集
  - 输出 `ad_generation/eval_cases.jsonl`
  - 同时生成 `ad_generation/eval_report.md`

评估会比较以下 baseline：

- `baseline_template_top1`
  - `template` 模式
  - `top_k=1`
  - 只按 query 级 top-1 商品生成文案
- `baseline_template_personalized`
  - `template` 模式
  - `top_k=1`
  - 带 `user_id / session_id` 的弱个性化来源标签
- `baseline_summary_topk`
  - `summary` 模式
  - `top_k=3`
  - 同时记录是否发生 fallback 以及原因
- `baseline_llm_topk`
  - `llm` 模式
  - `top_k=3`
  - 通过 `llm_generator.py` 生成文案
  - 记录 `llm_provider / llm_status / llm_fallback`

### 生成 eval_cases.jsonl 和 eval_report.md

```bash
python ad_generation/evaluate_samples.py ^
  --pairs_path ad_generation/data/train_pairs.jsonl ^
  --eval_cases_path ad_generation/eval_cases.jsonl ^
  --report_path ad_generation/eval_report.md ^
  --sample_size 39 ^
  --include_llm ^
  --llm_config ad_generation/llm_config.example.json
```

运行后会得到：

- `ad_generation/eval_cases.jsonl`
- `ad_generation/eval_report.md`

`eval_cases.jsonl` 的每条样本至少包含：

- `query`
- `user_id`
- `session_id`
- `evidence_items`
- `target_item`
- `target_copy`

`eval_report.md` 会汇总：

- 四组 baseline 的定义
- 样本数
- `summary` fallback 比例
- `llm` fallback 比例
- 至少 10 条对照样本
- 对四组 baseline 的定性总结

开启 LLM baseline 的方式：

- 加 `--include_llm`
- 用 `--llm_config` 指向配置文件
- 当前评估默认使用 `mock` provider，只验证链路，不会调用真实付费 API

## LLM 生成模式

当前 LLM 模式是在不破坏 `template / summary` 的前提下加上的可配置生成层。

新增参数：

- `--copy_tone`
  - `safe`：稳妥推荐型，更像推荐说明
  - `creative`：自然创意型，更像搜索广告文案，推荐真实 API 小规模验证时优先使用
  - `concise`：短句转化型，尽量压到 20~30 字

配置文件示例：

- `ad_generation/llm_config.example.json`

示例内容：

```json
{
  "provider": "mock",
  "model_name": "qwen-local-or-api-placeholder",
  "base_url": "https://your-openai-compatible-endpoint/v1",
  "api_key_env": "LLM_API_KEY",
  "temperature": 0.7,
  "max_new_tokens": 128,
  "timeout": 60,
  "provider_note": "Set provider to api_openai_compatible for small real-API validation only."
}
```

支持的 provider 结构：

- `mock`
  - 不调用真实模型
  - 返回模拟 LLM 文案
  - 适合先测试完整链路
- `local`
  - 预留给本地模型，如 Qwen / ChatGLM / Llama
  - 当前原型只保留接口，未接具体推理代码
- `api_openai_compatible`
  - 适配 OpenAI 兼容的 `chat/completions` 接口
  - 从环境变量读取 `LLM_API_KEY`
  - 需要自己填写 `base_url` 和 `model_name`
  - 当前只建议做小规模验证，不要直接跑全量数据

LLM 模式运行时流程：

1. 先按原有逻辑检索 `evidence_items`
2. 用 `build_llm_prompt(...)` 生成更严格且带 `copy_tone` 的提示词
3. 调用 `llm_generator.py`
4. 如果 LLM 调用失败，自动回退到 `summary`
5. 如果 `summary` 也触发 fallback，则最终回到 `template`

当前限制：

- `mock` provider 只是模拟生成，不代表真实大模型质量
- `local` 仍然只是预留接口
- 当前仍然是单条文案生成，不包含多候选采样、重排序和人工评估闭环
- 真实 API 接入只建议先跑少量 query，确认真实性约束和 fallback 正常

### 真实 API 小规模验证

1. 在系统环境变量中设置：

```bash
setx LLM_API_KEY "your_real_key_here"
```

2. 修改 `ad_generation/llm_config.example.json` 中的：

- `provider` 改为 `api_openai_compatible`
- `base_url` 改成你的 OpenAI 兼容服务地址，例如 `https://xxx/v1`
- `model_name` 改成你要调用的模型名

3. 只做小规模验证，例如：

```bash
python ad_generation/infer.py ^
  --query "华为手环" ^
  --corpus_path items_lite/train.jsonl ^
  --rank_path rank_lite/train.jsonl ^
  --users_path users_lite/train.jsonl ^
  --pairs_path ad_generation/data/train_pairs.jsonl ^
  --top_k 3 ^
  --copy_tone creative ^
  --copy_style llm ^
  --llm_config ad_generation/llm_config.example.json
```

建议先只跑这 3 条：

- `华为手环`
- `宿舍吹风机`
- `iqooz10x手机壳动漫`

不要直接跑全量数据，也不要把真实 key 写进代码或配置文件。

## 模型 A/B 测试

如果你想比较不同 LLM 在中文电商广告文案上的表现，可以使用：

- `ad_generation/llm_config.qwen_flash.json`
- `ad_generation/llm_config.qwen_plus.json`
- `ad_generation/llm_config.qwen_max.json`
- `ad_generation/llm_config.deepseek_chat.json`
- `ad_generation/compare_llm_models.py`

这组脚本会只跑 3 条固定 query：

- `华为手环`
- `宿舍吹风机`
- `iqooz10x手机壳动漫`

固定实验设置：

- `top_k=3`
- `copy_style=llm`
- `copy_tone=creative`

仅对 Qwen 三模型运行示例：

```bash
python ad_generation/compare_llm_models.py ^
  --corpus_path items_lite/train.jsonl ^
  --rank_path rank_lite/train.jsonl ^
  --users_path users_lite/train.jsonl ^
  --pairs_path ad_generation/data/train_pairs.jsonl ^
  --report_path ad_generation/model_compare_report.md
```

如果还想把 `deepseek-chat` 一起纳入对比：

```bash
python ad_generation/compare_llm_models.py ^
  --corpus_path items_lite/train.jsonl ^
  --rank_path rank_lite/train.jsonl ^
  --users_path users_lite/train.jsonl ^
  --pairs_path ad_generation/data/train_pairs.jsonl ^
  --report_path ad_generation/model_compare_report.md ^
  --include_deepseek ^
  --deepseek_config ad_generation/llm_config.deepseek_chat.json
```

脚本会输出：

- 每条 query 在当前比较模型集合下的最终文案
- `llm_status`
- `fallback_triggered`
- 是否命中坏句式检测
- 是否疑似出现价格/折扣/销量/功效类编造

注意事项：

- 运行前必须在同一个 PowerShell 会话中设置 `LLM_API_KEY`
- 如果启用了 `deepseek-chat`，还需要在同一个会话中设置 `DEEPSEEK_API_KEY`
- 如果当前进程拿不到所需环境变量，脚本会直接停止，不会拿 fallback 结果冒充模型对比
- 仅比较 Qwen 三模型时最多触发 9 次模型请求；加上 `deepseek-chat` 后最多触发 12 次

为什么 `qwen-plus` 可能更适合当前任务：

- 相比更轻量的 `qwen-flash`，通常更擅长中文句式组织和广告表达润色
- 相比更强但更贵的 `qwen-max`，在小规模中文电商文案场景里常常更均衡
- 当前项目强调“真实、自然、不过度营销”，`qwen-plus` 往往更适合作为第一候选做小规模验证

## SFT 数据准备

当前还没有进行真实训练，但已经可以把 `train_pairs.jsonl` 转成适合后续指令微调的 `messages` 格式数据。

转换脚本：

- `build_sft_data.py`
  - 输入：`ad_generation/data/train_pairs.jsonl`
  - 输出：`ad_generation/data/sft_train.jsonl`
  - 额外输出：`ad_generation/data/sft_sample.jsonl`

运行示例：

```bash
python ad_generation/build_sft_data.py ^
  --input_path ad_generation/data/train_pairs.jsonl ^
  --output_path ad_generation/data/sft_train.jsonl ^
  --sample_path ad_generation/data/sft_sample.jsonl ^
  --max_samples 5000 ^
  --top_k 3
```

SFT 样本结构：

- `messages`
  - `system`：固定角色“你是电商搜索广告文案生成助手。”
  - `user`：包含 `query`、用户画像、最近行为、top-k 商品证据和生成约束
  - `assistant`：当前使用 `target_copy`
- `metadata`
  - `query`
  - `user_id`
  - `session_id`
  - `source=template_pseudo_label`

为什么当前 `target_copy` 是伪标签：

- 它主要来自规则模板和现有证据拼接
- 不是人工标注文案
- 也不是经过真实大模型审核后的高质量金标准

因此，这批数据更适合：

- 作为后续 Qwen / Llama / ChatGLM 指令微调的起始数据
- 用来先验证训练格式、数据链路和微调脚本

当前不包含：

- 真实模型训练
- 大模型下载
- 外部 API 调用

## 真实 SFT 数据与 dry-run

当前已经补充了更贴近广告文案微调的数据与训练链路验证脚本：

- `build_real_sft_data.py`
  - 输入：`ad_generation/data/train_pairs.jsonl`
  - 输出：`outputs/real_sft_ad_copy.jsonl`
  - 额外输出：`outputs/real_sft_ad_copy_preview.md`
  - 优先调用 `deepseek_chat`，并走 V5 的 `parser -> validator -> ranker -> rewriter`
- `train_sft_ad_copy.py`
  - 输入：`outputs/real_sft_ad_copy.jsonl`
  - 输出：`outputs/sft_train_config.json`
  - 额外输出：`outputs/sft_train_preview.md`
  - 当前默认只做 `--dry_run`，验证 SFT 数据格式和训练文本拼接

说明：

- `real_sft_ad_copy.jsonl` 是由 `deepseek_chat` 生成并经过规则筛选的高质量伪标注数据
- 它不是人工金标，因此更适合先验证 SFT 数据链路、训练输入格式和后续微调脚本
- 当前 LoRA 训练不是必需项；即使没有真实训练出的本地模型，也不会影响 V5 的 `local_fake / deepseek_chat` 使用

## sft_local provider

V5 动态广告生成框架当前支持三类 provider：

- `local_fake`
- `deepseek_chat`
- `sft_local`

其中 `sft_local` 预留给未来的本地 SFT / LoRA 模型推理入口。

环境变量：

- `SFT_MODEL_PATH`
- `SFT_ADAPTER_PATH`（可选）
- `SFT_MAX_TOKENS`（默认 `512`）
- `SFT_TEMPERATURE`（默认 `0.7`）

当前行为：

- 如果没有配置 `SFT_MODEL_PATH`，V5 会返回 `skipped_no_sft_model`，并优雅 fallback，不会中断链路
- 如果 `SFT_MODEL_PATH` 已配置，但本地没有 `transformers / torch`，V5 会返回 `skipped_missing_dependencies`，并优雅 fallback
- 如果后续训练出了本地底模或 LoRA，可以通过 `SFT_MODEL_PATH / SFT_ADAPTER_PATH` 接回 V5，无需改动上层 `intent -> evidence -> style -> prompt -> parser -> validator -> ranker -> rewriter` 主链路

示例：

```bash
python -X utf8 ad_generation/demo_v5_pipeline.py --limit 5 --provider sft_local
```

## Windows 兼容性

实现中使用：

- `argparse`
- `pathlib.Path`
- UTF-8 读写

因此可以直接在 Windows PowerShell 环境中运行。
