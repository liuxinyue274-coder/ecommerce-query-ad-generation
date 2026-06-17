# Ad Generation Eval Report

## 实验设置

- 评估样本文件：`ad_generation/eval_cases.jsonl`
- 报告文件：`ad_generation/eval_report.md`
- 样本数：`39`
- baseline 1 `baseline_template_top1`：`template` 模式，`top_k=1`，只按 query 级证据生成。
- baseline 2 `baseline_template_personalized`：`template` 模式，`top_k=1`，优先带 `user_id / session_id` 的弱个性化来源标签。
- baseline 3 `baseline_summary_topk`：`summary` 模式，`top_k=3`，输出 fallback 标记与原因。
- baseline 4 `baseline_llm_topk`：`llm` 模式，`top_k=3`，通过 `llm_generator.py` 生成文案，并记录 provider / status / fallback。

## 样本覆盖

- 域覆盖：`{'digital': 9, 'apparel': 9, 'food': 9, 'health': 9, 'misc': 3}`
- query 长度覆盖：`{'short': 13, 'medium': 13, 'long': 13}`

## 基线统计

- `baseline_template_top1` 样本数：`39`
- `baseline_template_personalized` 样本数：`39`
- `baseline_summary_topk` 样本数：`39`
- `baseline_llm_topk` 样本数：`39`
- `summary` fallback 数：`23/39`
- `summary` fallback 比例：`58.97%`
- 个性化文案与非个性化文案不同的样本数：`0/39`
- `summary` 原因分布：`{'shared_major_category': 14, 'insufficient_evidence_items': 22, 'no_summary_signal': 1, 'major_category_majority': 1, 'shared_brand_and_leaf_category': 1}`

## 对照样本（前 10 条）

## LLM 统计

- LLM provider 分布：`{'mock': 39}`
- `llm` fallback 数：`0/39`
- `llm` fallback 比例：`0.00%`
- `baseline_llm_topk` 与 `baseline_summary_topk` 不同的样本数：`39/39`
- `baseline_llm_topk` 与 `baseline_template_top1` 不同的样本数：`39/39`

### Case 1
- query：`华为手环`
- domain / length：`digital` / `short`
- baseline_template_top1：搜“华为手环”可以先看这款中性/NEUTRAL新款智能手表多功能运动手环可充电计步闹钟高颜值适用小米安卓机，由林创腕表店在售，信息清楚，适合先了解。
- baseline_template_personalized：搜“华为手环”可以先看这款中性/NEUTRAL新款智能手表多功能运动手环可充电计步闹钟高颜值适用小米安卓机，由林创腕表店在售，信息清楚，适合先了解。
- baseline_summary_topk：搜“华为手环”时，当前候选主要集中在手机/数码/电脑办公下的智能手环设备方向，建议先从这类商品里继续筛选。 (fallback=False, reason=shared_major_category)
- baseline_llm_topk：搜“华为手环”可先关注新款智能手表多功能运动手环可等商品，更贴近当前需求。 (provider=mock, status=llm_ok, fallback=False, reason=none)

### Case 2
- query：`宿舍吹风机`
- domain / length：`digital` / `short`
- baseline_template_top1：搜“宿舍吹风机”可以先看这款欧伊俪/OUYIL电吹风机家用负离子护发学生宿舍800w不伤发小型吹风机男女士专用，由欧伊俪生活旗舰店在售，信息清楚，适合先了解。
- baseline_template_personalized：搜“宿舍吹风机”可以先看这款欧伊俪/OUYIL电吹风机家用负离子护发学生宿舍800w不伤发小型吹风机男女士专用，由欧伊俪生活旗舰店在售，信息清楚，适合先了解。
- baseline_summary_topk：搜“宿舍吹风机”可以先看这款欧伊俪/OUYIL电吹风机家用负离子护发学生宿舍800w不伤发小型吹风机男女士专用，由欧伊俪生活旗舰店在售，信息清楚，适合先了解。 (fallback=True, reason=insufficient_evidence_items)
- baseline_llm_topk：搜“宿舍吹风机”可先关注电吹风机家用负离子护发学生宿等商品，更贴近当前需求。 (provider=mock, status=llm_ok, fallback=False, reason=none)

### Case 3
- query：`苹果6手机膜`
- domain / length：`digital` / `short`
- baseline_template_top1：搜“苹果6手机膜”可以先看这款卡通可爱适用苹果6钢化膜iPhone8Plus全屏7软边SE2防摔女款手机膜，由披风手机配件在售，信息清楚，适合先了解。
- baseline_template_personalized：搜“苹果6手机膜”可以先看这款卡通可爱适用苹果6钢化膜iPhone8Plus全屏7软边SE2防摔女款手机膜，由披风手机配件在售，信息清楚，适合先了解。
- baseline_summary_topk：搜“苹果6手机膜”可以先看这款卡通可爱适用苹果6钢化膜iPhone8Plus全屏7软边SE2防摔女款手机膜，由披风手机配件在售，信息清楚，适合先了解。 (fallback=True, reason=insufficient_evidence_items)
- baseline_llm_topk：搜“苹果6手机膜”可先关注卡通可爱适用苹果6钢化膜iP等商品，更贴近当前需求。 (provider=mock, status=llm_ok, fallback=False, reason=none)

### Case 4
- query：`88w超级快充oppo`
- domain / length：`digital` / `medium`
- baseline_template_top1：搜“88w超级快充oppo”可以先看这款佰卡朗88W适用华为荣耀OPPO手机超级快充充电器线套装快充Type-C数据线，由王者数码通讯在售，信息清楚，适合先了解。
- baseline_template_personalized：搜“88w超级快充oppo”可以先看这款佰卡朗88W适用华为荣耀OPPO手机超级快充充电器线套装快充Type-C数据线，由王者数码通讯在售，信息清楚，适合先了解。
- baseline_summary_topk：搜“88w超级快充oppo”可以先看这款佰卡朗88W适用华为荣耀OPPO手机超级快充充电器线套装快充Type-C数据线，由王者数码通讯在售，信息清楚，适合先了解。 (fallback=True, reason=insufficient_evidence_items)
- baseline_llm_topk：搜“88w超级快充oppo”可先关注88W适用华为荣耀OPPO手等商品，更贴近当前需求。 (provider=mock, status=llm_ok, fallback=False, reason=none)

### Case 5
- query：`iqooz10x手机壳动漫`
- domain / length：`digital` / `medium`
- baseline_template_top1：搜“iqooz10x手机壳动漫”可以先看这款适用于IQOOZ10X手机壳V2445A液态硅胶网红保护套全包时尚软壳，由安欣文化数码在售，信息清楚，适合先了解。
- baseline_template_personalized：搜“iqooz10x手机壳动漫”可以先看这款适用于IQOOZ10X手机壳V2445A液态硅胶网红保护套全包时尚软壳，由安欣文化数码在售，信息清楚，适合先了解。
- baseline_summary_topk：搜“iqooz10x手机壳动漫”时，当前候选主要集中在手机/数码/电脑办公下的手机壳配件方向，建议先从这类商品里继续筛选。 (fallback=False, reason=shared_major_category)
- baseline_llm_topk：搜“iqooz10x手机壳动漫”可先关注适用于IQOOZ10X手机壳等商品，更贴近当前需求。 (provider=mock, status=llm_ok, fallback=False, reason=none)

### Case 6
- query：`ooppo手机壳女`
- domain / length：`digital` / `medium`
- baseline_template_top1：搜“ooppo手机壳女”可以先看这款汇客凡品蛇年金色细闪【oppoa3pro】电镀防摔旋转支架手机壳，由宋壳在售，信息清楚，适合先了解。
- baseline_template_personalized：搜“ooppo手机壳女”可以先看这款汇客凡品蛇年金色细闪【oppoa3pro】电镀防摔旋转支架手机壳，由宋壳在售，信息清楚，适合先了解。
- baseline_summary_topk：搜“ooppo手机壳女”时，当前候选主要集中在手机/数码/电脑办公下的手机壳配件方向，建议先从这类商品里继续筛选。 (fallback=False, reason=shared_major_category)
- baseline_llm_topk：搜“ooppo手机壳女”可先关注汇客凡品蛇年金色细闪【opp等商品，更贴近当前需求。 (provider=mock, status=llm_ok, fallback=False, reason=none)

### Case 7
- query：`iphone 16 pro max后盖`
- domain / length：`digital` / `long`
- baseline_template_top1：搜“iphone 16 pro max后盖”可以先看这款适用于苹果16Promax后盖玻璃iPhone16pro后盖总成iPhone16后盖，由鸿源科技3C配件店在售，信息清楚，适合先了解。
- baseline_template_personalized：搜“iphone 16 pro max后盖”可以先看这款适用于苹果16Promax后盖玻璃iPhone16pro后盖总成iPhone16后盖，由鸿源科技3C配件店在售，信息清楚，适合先了解。
- baseline_summary_topk：搜“iphone 16 pro max后盖”可以先看这款适用于苹果16Promax后盖玻璃iPhone16pro后盖总成iPhone16后盖，由鸿源科技3C配件店在售，信息清楚，适合先了解。 (fallback=True, reason=insufficient_evidence_items)
- baseline_llm_topk：搜“iphone 16 pro max后盖”可先关注适用于苹果16Promax后等商品，更贴近当前。 (provider=mock, status=llm_ok, fallback=False, reason=none)

### Case 8
- query：`华为nova12挎包手机壳女可爱代`
- domain / length：`digital` / `long`
- baseline_template_top1：搜“华为nova12挎包手机壳女可爱代”可以先看这款适用华为nova12手机壳BLK-AL00新款小羊皮腕带支架带绳硅胶新年款，由糖潮3C在售，信息清楚，适合先了解。
- baseline_template_personalized：搜“华为nova12挎包手机壳女可爱代”可以先看这款适用华为nova12手机壳BLK-AL00新款小羊皮腕带支架带绳硅胶新年款，由糖潮3C在售，信息清楚，适合先了解。
- baseline_summary_topk：搜“华为nova12挎包手机壳女可爱代”时，当前候选主要集中在手机/数码/电脑办公下的手机壳配件方向，建议先从这类商品里继续筛选。 (fallback=False, reason=shared_major_category)
- baseline_llm_topk：搜“华为nova12挎包手机壳女可爱代”可先关注适用华为nova12手机壳B等商品，更贴近当前需求。 (provider=mock, status=llm_ok, fallback=False, reason=none)

### Case 9
- query：`华为鼎桥m40手机的原装专用。充电器`
- domain / length：`digital` / `long`
- baseline_template_top1：搜“华为鼎桥m40手机的原装专用。充电器”可以先看这款适用华为鼎桥/TD Tech M40 快充6A数据线TDT-MA01原正装品40w超级，由壮壮OV3c数码在售，信息清楚，适合先了解。
- baseline_template_personalized：搜“华为鼎桥m40手机的原装专用。充电器”可以先看这款适用华为鼎桥/TD Tech M40 快充6A数据线TDT-MA01原正装品40w超级，由壮壮OV3c数码在售，信息清楚，适合先了解。
- baseline_summary_topk：搜“华为鼎桥m40手机的原装专用。充电器”可以先看这款适用华为鼎桥/TD Tech M40 快充6A数据线TDT-MA01原正装品40w超级，由壮壮OV3c数码在售，信息清楚，适合先了解。 (fallback=True, reason=insufficient_evidence_items)
- baseline_llm_topk：搜“华为鼎桥m40手机的原装专用。充电器”可先关注适用华为鼎桥/TD Tech等商品，更贴近当前需。 (provider=mock, status=llm_ok, fallback=False, reason=none)

### Case 10
- query：`橘子树`
- domain / length：`apparel` / `short`
- baseline_template_top1：搜“橘子树”可以先看这款JZSZNR橘子树在哪儿美式复古工装裤女秋季设计感提臀阔腿显瘦大口袋裤子，由橘子树在哪儿服饰店在售，信息清楚，适合先了解。
- baseline_template_personalized：搜“橘子树”可以先看这款JZSZNR橘子树在哪儿美式复古工装裤女秋季设计感提臀阔腿显瘦大口袋裤子，由橘子树在哪儿服饰店在售，信息清楚，适合先了解。
- baseline_summary_topk：搜“橘子树”可以先看这款JZSZNR橘子树在哪儿美式复古工装裤女秋季设计感提臀阔腿显瘦大口袋裤子，由橘子树在哪儿服饰店在售，信息清楚，适合先了解。 (fallback=True, reason=insufficient_evidence_items)
- baseline_llm_topk：搜“橘子树”可先关注橘子树在哪儿美式复古工装裤女等商品，更贴近当前需求。 (provider=mock, status=llm_ok, fallback=False, reason=none)

## 定性总结

- `baseline_template_top1` 最像商品复述，因为它稳定围绕 top-1 商品标题、店铺和类目展开。
- `baseline_template_personalized` 在当前评估集上通常只改变命中标签，不一定改变最终文案；这说明弱个性化输入已经接通，但受限于 `train_pairs` 基本是一条 query 对应一条样本。
- `baseline_summary_topk` 在不 fallback 时更自然，也更适合表达“先看哪类商品方向”；它不强行复述某一个完整商品标题。
- 本次 `summary` 的 fallback 比例是 `58.97%`。 fallback 主要发生在证据不足或 top-k 之间缺少稳定共性时，此时系统会回退到稳妥的 template 文案。
- `baseline_llm_topk` 的 fallback 比例是 `0.00%`。 它的语言更自然，更接近真实广告文案，但需要真实性约束和 fallback 机制。
- `mock` provider 当前只用于验证链路，后续可以替换成本地 Qwen 或真实 API。
- 如果后续课程展示希望突出‘个性化确实生效’，更适合补一个重复 query 更多、同 query 多用户的评估子集。
