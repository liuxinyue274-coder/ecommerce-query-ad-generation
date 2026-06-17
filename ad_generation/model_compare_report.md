# LLM Model Compare Report

本报告只对 3 条 query 做最小 A/B 测试，保持：`top_k=3`、`copy_style=llm`、`copy_tone=creative`。

对比模型：`qwen-flash`、`qwen-plus`、`qwen-max`、`deepseek-chat`

## Query: 华为手环

### qwen-flash
- final ad copy: 运动通勤都轻松记录？搜“华为手环”，可以先看看这款支持心率监测和睡眠分析的智能手环。
- llm_status: llm_ok
- fallback_triggered: False
- bad_copy_detected: False
- suspicious_claims: 无

### qwen-plus
- final ad copy: 想测睡眠质量、盯心率又不想换手机？华为手环10支持专业健康监测，适配安卓机型直接用。
- llm_status: llm_ok
- fallback_triggered: False
- bad_copy_detected: False
- suspicious_claims: 无

### qwen-max
- final ad copy: 想要全天候健康监测？搜“华为手环”，专业睡眠情绪心率分析，适合年轻活力的你。
- llm_status: llm_ok
- fallback_triggered: False
- bad_copy_detected: False
- suspicious_claims: 无

### deepseek-chat
- final ad copy: 围绕“华为手环”这类需求，不妨先看智能手环相关款，更容易找到贴合场景的选择。
- llm_status: llm_failed: api_openai_compatible http_error=402 detail={"error":{"message":"Insufficient Balance","type":"unknown_error","param":null,"code":"invalid_request_error"}}
- fallback_triggered: True
- bad_copy_detected: False
- suspicious_claims: 无

## Query: 宿舍吹风机

### qwen-flash
- final ad copy: 宿舍吹发怕吵人？选这款800W小功率负离子护发吹风机，不伤发还安静，学生党用正合适。
- llm_status: llm_ok
- fallback_triggered: False
- bad_copy_detected: False
- suspicious_claims: 无

### qwen-plus
- final ad copy: 宿舍空间小、用电有限？这款800W小型吹风机，负离子护发，学生党用着轻便又安心。
- llm_status: llm_ok
- fallback_triggered: False
- bad_copy_detected: False
- suspicious_claims: 无

### qwen-max
- final ad copy: 宿舍生活更便捷，轻便实用的负离子护发吹风机，男女都适用哦。
- llm_status: llm_ok
- fallback_triggered: False
- bad_copy_detected: False
- suspicious_claims: 无

### deepseek-chat
- final ad copy: 想找“宿舍吹风机”？不妨先看看更贴合这类需求的相关款，更容易挑到顺手的一款。
- llm_status: llm_failed: api_openai_compatible http_error=402 detail={"error":{"message":"Insufficient Balance","type":"unknown_error","param":null,"code":"invalid_request_error"}}
- fallback_triggered: True
- bad_copy_detected: False
- suspicious_claims: 无

## Query: iqooz10x手机壳动漫

### qwen-flash
- final ad copy: 想给iQOO Z10x换个性动漫风外壳？可以先看看KT猫主题的软硅胶保护套，可爱又防摔。
- llm_status: llm_ok
- fallback_triggered: False
- bad_copy_detected: False
- suspicious_claims: 无

### qwen-plus
- final ad copy: 想给 iQOO Z10x 换个动漫风外观？V2445A 液态硅胶软壳全包防摔，KT猫、透明款都有。
- llm_status: llm_ok
- fallback_triggered: False
- bad_copy_detected: False
- suspicious_claims: 无

### qwen-max
- final ad copy: 想给 iQOO Z10x 换个动漫风外观？可以先看看这些个性十足的手机壳，既可爱又防摔。
- llm_status: llm_ok
- fallback_triggered: False
- bad_copy_detected: False
- suspicious_claims: 无

### deepseek-chat
- final ad copy: 围绕“iqooz10x手机壳动漫”这类需求，不妨先看手机壳相关款，更容易找到贴合场景的选择。
- llm_status: llm_failed: api_openai_compatible http_error=402 detail={"error":{"message":"Insufficient Balance","type":"unknown_error","param":null,"code":"invalid_request_error"}}
- fallback_triggered: True
- bad_copy_detected: False
- suspicious_claims: 无

## Notes

- `fallback_triggered=True` 说明该模型在当前调用中没有成功给出最终 LLM 文案，结果已走现有 fallback 机制。
- `suspicious_claims` 只是轻量规则提示，不等价于严格事实核验。
- 这份报告不覆盖 `eval_report.md`，仅用于小规模模型对比。
