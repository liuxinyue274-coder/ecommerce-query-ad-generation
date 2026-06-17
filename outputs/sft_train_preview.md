# SFT Train Preview

当前 real_sft_ad_copy.jsonl 是由 deepseek_chat 生成并经过规则筛选的高质量伪标注数据，不是人工金标，主要用于 SFT 链路验证。

## Stats

- total_samples: `5`
- valid_samples: `5`
- invalid_samples: `0`
- avg_assistant_length: `24.2`
- provider_distribution: `{'deepseek_chat': 5}`
- evidence_confidence_distribution: `{'medium': 2, 'high': 3}`
- avg_score: `86.0`

## Config

```json
{
  "notice": "当前 real_sft_ad_copy.jsonl 是由 deepseek_chat 生成并经过规则筛选的高质量伪标注数据，不是人工金标，主要用于 SFT 链路验证。",
  "data_path": "outputs\\real_sft_ad_copy.jsonl",
  "dry_run": true,
  "model_name_or_path": "",
  "output_dir": "outputs\\sft_ad_copy_lora",
  "max_steps": 50,
  "batch_size": 1,
  "learning_rate": 2e-05,
  "use_lora": true,
  "dependency_status": {
    "torch": true,
    "transformers": false,
    "peft": false,
    "trl": false
  },
  "stats": {
    "total_samples": 5,
    "valid_samples": 5,
    "invalid_samples": 0,
    "avg_assistant_length": 24.2,
    "provider_distribution": {
      "deepseek_chat": 5
    },
    "evidence_confidence_distribution": {
      "medium": 2,
      "high": 3
    },
    "avg_score": 86.0,
    "invalid_reason_distribution": {}
  }
}
```

## Training Text Preview

### Sample 1

```text
<|system|>
你是电商搜索广告创意生成助手。请基于用户搜索意图、用户画像和商品证据，生成自然、短促、合规的广告文案。不得编造证据中没有的信息。
<|user|>
Query: "港味牛仔"2025早秋复古水洗正腰高腰弯刀宽松休闲牛仔裤
Intent: 意图总结：香朵儿/SHALLDOER / 2025 / 牛仔裤装
关注点：港味牛仔 / 早秋复古
属性提示：港味牛仔、早秋复古、水洗正腰、高腰弯刀、宽松休闲
User Profile: 用户画像：性别：F，年龄段：24-30，城市：龙岩
Evidence: 1. "港味牛仔"2025早秋复古水洗正腰高腰弯刀宽松休闲牛仔裤 | 品牌：香朵儿/SHALLDOER | 类目：女装 > 裤子 > 牛仔裤装 | 店铺：香朵儿服饰旗舰店
Style Examples: 1. 晨跑想看运动手环，计步心率一眼更省事。
2. 宿舍吹风机别太占地，小巧速干拿着更顺手。
3. 通勤双肩包装电脑和杂物，分层收纳背着更利落。
4. 敏感肌面霜换季脸干时，保湿修护这类更安心。
5. 手机壳别只看花样，防摔贴手的款日常更耐用。
请生成1条自然广告文案。
<|assistant|>
港味牛仔，正腰弯刀剪裁，早秋复古感，通勤休闲都百搭。
```

### Sample 2

```text
<|system|>
你是电商搜索广告创意生成助手。请基于用户搜索意图、用户画像和商品证据，生成自然、短促、合规的广告文案。不得编造证据中没有的信息。
<|user|>
Query: 'iqoo13
Intent: 意图总结：iQOO / iqoo13 / 手机设备
User Profile: 用户画像：性别：M，年龄段：31-40，城市：邯郸
最近行为：九通道合金遥控货柜车集装箱运输电动半挂牵引车拖挂车儿童玩具；青艺儿童遥控工程玩具车奔驰翻斗车电动自卸车半挂车货车汽车模型；儿童遥控半挂车RC电动货车仿真运输车充电集装箱卡车工程车；仿真遥控平板车大号卡车运输半挂平板拖车工程车儿童玩具男孩；想玩无线遥控车拖头半挂重型运输大卡车自卸车工程车儿童男孩玩具
Evidence: 1. iQOO 13 16GB+512GB 旗舰手机晓龙8至尊芯片大容量电池智能机 | 品牌：iQOO | 类目：手机/数码/电脑办公 > 手机及配件 > 手机设备 | 店铺：快手优选官方频道
2. 【L国补】iQOO 13 性能之光 5G智能手机 | 品牌：iQOO | 类目：手机/数码/电脑办公 > 手机及配件 > 手机设备 | 店铺：iQOO官方旗舰店
Style Examples: 1. 晨跑想看运动手环，计步心率一眼更省事。
2. 宿舍吹风机别太占地，小巧速干拿着更顺手。
3. 通勤双肩包装电脑和杂物，分层收纳背着更利落。
4. 敏感肌面霜换季脸干时，保湿修护这类更安心。
5. 手机壳别只看花样，防摔贴手的款日常更耐用。
请生成1条自然广告文案。
<|assistant|>
iQOO13性能之光，硬核配置流畅驾驭游戏与日常场景。
```
