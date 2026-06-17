# Real SFT Ad Copy Preview

- requested_provider: `deepseek_chat`
- effective_provider: `deepseek_chat`
- generated_samples: `100`
- scanned_samples: `279`
- filtered_samples: `179`
- output_path: `outputs/real_sft_ad_copy.jsonl`

## Filter Counts

- score_below_min: 59
- unrelated_to_evidence: 55
- query_too_noisy: 39
- suspicious_claims: 25
- copy_length_out_of_range: 1

## Preview Samples

### Sample 1

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是电商搜索广告创意生成助手。请基于用户搜索意图、用户画像和商品证据，生成自然、短促、合规的广告文案。不得编造证据中没有的信息。"
    },
    {
      "role": "user",
      "content": "Query: \"氨糖软骨素钙片\nIntent: 意图总结：北京同仁堂 / 其他营养补充品\n关注点：氨糖软骨 / 素钙片\n属性提示：氨糖软骨、素钙片\nUser Profile: 用户画像：性别：F，年龄段：31-40，城市：厦门\n最近行为：修正美白淡斑精华液老年祛雀斑嫩白去黄提亮肤色滋润保湿肌肤焕颜；2025早秋新款韩版宽松百搭木耳边复古V领衬衫女显瘦阔腿牛仔裤子；洋气夏季时尚简版款气质上衣+休闲裤时尚套装；莱凡特优碳优酮固体饮料；【莱凡特】优碳优酮固体饮料×10包（100袋）\nEvidence: 1. 北京同仁堂软骨钙片120粒/瓶氨糖软骨素膝盖手脚麻木乏力中老年钙 | 品牌：北京同仁堂 | 类目：营养健康 > 膳食营养食品 > 其他营养补充品 | 店铺：南京同仁品牌自营\n2. 3瓶装Perseek7合1氨糖msm鲨鱼软骨素高钙钙片软骨素肽钙官方正品 | 品牌：椎名制药 | 类目：营养健康 > 膳食营养食品 > 海洋提取物 | 店铺：椎名制药原众堂专卖店\nStyle Examples: 1. 晨跑想看运动手环，计步心率一眼更省事。\n2. 宿舍吹风机别太占地，小巧速干拿着更顺手。\n3. 通勤双肩包装电脑和杂物，分层收纳背着更利落。\n4. 敏感肌面霜换季脸干时，保湿修护这类更安心。\n5. 手机壳别只看花样，防摔贴手的款日常更耐用。\n请生成1条自然广告文案。"
    },
    {
      "role": "assistant",
      "content": "北京同仁堂氨糖软骨素钙片，中青年日常关节养护，温和好吸收。"
    }
  ],
  "metadata": {
    "query": "\"氨糖软骨素钙片",
    "provider": "deepseek_chat",
    "score": 94.0,
    "evidence_confidence": "high",
    "quality_tags": [
      "provider_deepseek_chat",
      "llm_status_provider_ok",
      "final_source_llm_ranker_top1",
      "evidence_high",
      "style_conditioned",
      "direct_rank_top1"
    ]
  }
}
```

### Sample 2

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是电商搜索广告创意生成助手。请基于用户搜索意图、用户画像和商品证据，生成自然、短促、合规的广告文案。不得编造证据中没有的信息。"
    },
    {
      "role": "user",
      "content": "Query: #【两件套】撞色条纹性感斜肩长袖套头毛衣+辣妹针织直筒短裤套装石\nIntent: 意图总结：无品牌 / 其他套装服饰\n关注点：两件套 / 撞色条纹\n属性提示：两件套、撞色条纹、性感斜肩、长袖套头、毛衣\nUser Profile: 用户画像：性别：F，年龄段：18-23，城市：毕节\n最近行为：100%纯棉2025美式复古设计感爱心长袖打底内搭高街潮牌情侣上衣新；早秋潮圆领短袖T恤女修身型字母白色刺绣美式辣妹风纯欲显瘦短款T；ZLD烫花美式复古字母logo新款黑色宽松设计感圆领重磅短袖T恤落肩；大毛领连帽韩版加厚棉服收腰显瘦小个子保暖短款棉衣潮外套面包服；2025秋季新款精梳纯棉T恤高克重磅长袖内搭基础款美式小领口卫衣\nEvidence: 1. 【两件套】撞色条纹性感斜肩长袖套头毛衣+辣妹针织直筒短裤套装 | 类目：女装 > 套装/制服 > 其他套装服饰 | 店铺：艳子优品66\nStyle Examples: 1. 晨跑想看运动手环，计步心率一眼更省事。\n2. 宿舍吹风机别太占地，小巧速干拿着更顺手。\n3. 通勤双肩包装电脑和杂物，分层收纳背着更利落。\n4. 敏感肌面霜换季脸干时，保湿修护这类更安心。\n5. 手机壳别只看花样，防摔贴手的款日常更耐用。\n请生成1条自然广告文案。"
    },
    {
      "role": "assistant",
      "content": "秋冬换季纠结穿什么？这套斜肩毛衣+短裤套装。"
    }
  ],
  "metadata": {
    "query": "#【两件套】撞色条纹性感斜肩长袖套头毛衣+辣妹针织直筒短裤套装石",
    "provider": "deepseek_chat",
    "score": 81.0,
    "evidence_confidence": "medium",
    "quality_tags": [
      "provider_deepseek_chat",
      "llm_status_provider_ok",
      "final_source_llm_ranker_top1",
      "evidence_medium",
      "style_conditioned",
      "rewritten"
    ]
  }
}
```

### Sample 3

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是电商搜索广告创意生成助手。请基于用户搜索意图、用户画像和商品证据，生成自然、短促、合规的广告文案。不得编造证据中没有的信息。"
    },
    {
      "role": "user",
      "content": "Query: 'iqoo13\nIntent: 意图总结：iQOO / iqoo13 / 手机设备\nUser Profile: 用户画像：性别：M，年龄段：31-40，城市：邯郸\n最近行为：九通道合金遥控货柜车集装箱运输电动半挂牵引车拖挂车儿童玩具；青艺儿童遥控工程玩具车奔驰翻斗车电动自卸车半挂车货车汽车模型；儿童遥控半挂车RC电动货车仿真运输车充电集装箱卡车工程车；仿真遥控平板车大号卡车运输半挂平板拖车工程车儿童玩具男孩；想玩无线遥控车拖头半挂重型运输大卡车自卸车工程车儿童男孩玩具\nEvidence: 1. iQOO 13 16GB+512GB 旗舰手机晓龙8至尊芯片大容量电池智能机 | 品牌：iQOO | 类目：手机/数码/电脑办公 > 手机及配件 > 手机设备 | 店铺：快手优选官方频道\n2. 【L国补】iQOO 13 性能之光 5G智能手机 | 品牌：iQOO | 类目：手机/数码/电脑办公 > 手机及配件 > 手机设备 | 店铺：iQOO官方旗舰店\nStyle Examples: 1. 晨跑想看运动手环，计步心率一眼更省事。\n2. 宿舍吹风机别太占地，小巧速干拿着更顺手。\n3. 通勤双肩包装电脑和杂物，分层收纳背着更利落。\n4. 敏感肌面霜换季脸干时，保湿修护这类更安心。\n5. 手机壳别只看花样，防摔贴手的款日常更耐用。\n请生成1条自然广告文案。"
    },
    {
      "role": "assistant",
      "content": "iQOO13性能之光，硬核配置先看再说。"
    }
  ],
  "metadata": {
    "query": "'iqoo13",
    "provider": "deepseek_chat",
    "score": 90.0,
    "evidence_confidence": "high",
    "quality_tags": [
      "provider_deepseek_chat",
      "llm_status_provider_ok",
      "final_source_llm_ranker_top1",
      "evidence_high",
      "style_conditioned",
      "rewritten"
    ]
  }
}
```

### Sample 4

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是电商搜索广告创意生成助手。请基于用户搜索意图、用户画像和商品证据，生成自然、短促、合规的广告文案。不得编造证据中没有的信息。"
    },
    {
      "role": "user",
      "content": "Query: (猪头妹专属】忆源甄选草原鲜乳大饼\nIntent: 意图总结：忆源香 / 饼干糕点\n关注点：猪头妹专 / 忆源甄选\n属性提示：猪头妹专、忆源甄选、草原鲜乳、大饼\nUser Profile: 用户画像：性别：M，年龄段：18-23，城市：梅州\n最近行为：【米哥专属】来懿品橙皮山楂片怀旧大片零食150g/袋；【猪头妹专属】帝柏图素瓷锅智能微压炒锅家用铁锅；【480g*5袋】金汤酸菜鱼活鱼鲜切480g懒人菜（袋装）烟渔如画；(10包)金汤酸菜鱼调料加热即食正宗川味下饭菜通用家庭调味zg；【今日破价5大袋】邢鲜生 金汤黑鱼酸菜鱼500g/袋 活黑鱼鲜切\nEvidence: 1. 【猪头妹专属】忆源甄选草原鲜乳大饼 | 品牌：忆源香 | 类目：零食/坚果/特产 > 饼干/膨化食品 > 饼干糕点 | 店铺：景宸辉睿食品\nStyle Examples: 1. 晨跑想看运动手环，计步心率一眼更省事。\n2. 宿舍吹风机别太占地，小巧速干拿着更顺手。\n3. 通勤双肩包装电脑和杂物，分层收纳背着更利落。\n4. 敏感肌面霜换季脸干时，保湿修护这类更安心。\n5. 手机壳别只看花样，防摔贴手的款日常更耐用。\n请生成1条自然广告文案。"
    },
    {
      "role": "assistant",
      "content": "追剧时拆一袋，草原鲜乳大饼的奶香刚好搭下午茶。"
    }
  ],
  "metadata": {
    "query": "(猪头妹专属】忆源甄选草原鲜乳大饼",
    "provider": "deepseek_chat",
    "score": 84.0,
    "evidence_confidence": "medium",
    "quality_tags": [
      "provider_deepseek_chat",
      "llm_status_provider_ok",
      "final_source_llm_ranker_top1",
      "evidence_medium",
      "style_conditioned",
      "direct_rank_top1"
    ]
  }
}
```

### Sample 5

```json
{
  "messages": [
    {
      "role": "system",
      "content": "你是电商搜索广告创意生成助手。请基于用户搜索意图、用户画像和商品证据，生成自然、短促、合规的广告文案。不得编造证据中没有的信息。"
    },
    {
      "role": "user",
      "content": "Query: - eraclean世净隐形眼镜清洗器\nIntent: 意图总结：无品牌 / 超声波清洗设备\n关注点：eraclean / 世净隐形\n属性提示：eraclean、世净隐形、眼镜清洗\nUser Profile: 用户画像：性别：F，年龄段：24-30，城市：清远\n最近行为：遮不住PP!18cm超短百褶裙JK半身粉格迷你裙日系COS可爱女；性别伦理学同文馆哲学；便携搞怪雨伞奇葩胡萝卜创意伞学生上学防水防紫外线晴雨两用雨伞；夫妻床上辅助枕情侣神器多功能俯卧位趴枕头趴着睡玩手机趴枕垫；防水床单护垫情侣推油布露营床品垫可洗一次性床垫单件酒店床上用\nEvidence: 1. eraclean世净隐形眼镜清洗器美瞳盒子自动超声波电动消毒收纳便携 | 类目：家用电器 > 生活电器 > 超声波清洗设备 | 店铺：龙华波\n2. 适用GM01小米有品EraClean隐形眼镜清洗器充电线世净超声波清洗机 | 类目：家用电器 > 生活电器 > 超声波清洗设备 | 店铺：雅卓音乐世界\nStyle Examples: 1. 晨跑想看运动手环，计步心率一眼更省事。\n2. 宿舍吹风机别太占地，小巧速干拿着更顺手。\n3. 通勤双肩包装电脑和杂物，分层收纳背着更利落。\n4. 敏感肌面霜换季脸干时，保湿修护这类更安心。\n5. 手机壳别只看花样，防摔贴手的款日常更耐用。\n请生成1条自然广告文案。"
    },
    {
      "role": "assistant",
      "content": "超声波清洗隐形眼镜，懒人清洁一步到位。"
    }
  ],
  "metadata": {
    "query": "- eraclean世净隐形眼镜清洗器",
    "provider": "deepseek_chat",
    "score": 82.0,
    "evidence_confidence": "high",
    "quality_tags": [
      "provider_deepseek_chat",
      "llm_status_provider_ok",
      "final_source_llm_ranker_top1",
      "evidence_high",
      "style_conditioned",
      "direct_rank_top1"
    ]
  }
}
```
