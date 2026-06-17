# Champagne Gold 高级感视觉重做

**日期**: 2026-05-24
**作者**: 设计师 + Claude
**目标文件**: `D:\hllm visualization\app.py`、`D:\hllm visualization\.streamlit\config.toml`

## 背景

dashboard 当前用 `#FF6B35` 橙作主色,色彩偏快销/淘宝感,与"LLM 驱动的高级电商广告平台"项目定位不匹配。需提升至 Linear/Vercel/88VIP 那种高级感,同时不破坏演示模式 C 端预览本身就该有的电商热烈调性。

## 设计原则

1. **浅色底 + 香槟金点缀** —— 不做暗色模式,主体仍是暖白,香槟金作为单一品牌 accent。
2. **点状金属拉丝高光** —— 仅在关键焦点(header、主 CTA、sac.steps 完成态、模式切换选中态)使用 linear-gradient + inset ring 模拟拉丝铜面。其他位置保持纯色,避免油腻。
3. **分模式哲学** —— 调试模式走极简金属高级风;演示模式 C 端预览保留淘宝橙 `#FF6B35`,形成"开发者会客室 vs 消费者商场"的反差。
4. **品牌名固化** —— **Ecommerce CopyGen Studio**,LLM 驱动的电商广告生成与诊断平台。

## 色板

| Token | Hex | 用途 |
| --- | --- | --- |
| `bg-base` | `#FAFAF9` | 全局页面底(warm off-white,stone-50) |
| `bg-card` | `#FFFFFF` | 卡片底 |
| `bg-tint` | `#F5F5F4` | 次要区块底(stone-100) |
| `text-primary` | `#1C1917` | 主文字(warm near-black,stone-900) |
| `text-muted` | `#78716C` | 次文字、caption(stone-500) |
| `border-default` | `#E7E5E4` | 默认边框(stone-200) |
| `border-focus` | `#C9A876` | 聚焦/重要卡片边框(浅香槟金) |
| `gold-light` | `#D4B27D` | 香槟金高光、渐变上端 |
| `gold-mid` | `#B8985C` | **主品牌色**,渐变中段、dot、链接 |
| `gold-deep` | `#8B6F3D` | 渐变底端、深边框、激活态文字 |
| `success` | `#15803D` | 成功状态(emerald-700,深沉) |
| `danger` | `#B91C1C` | 失败状态(red-700,深沉) |
| `taobao-orange` | `#FF6B35` | **C 端预览专属**,不外溢 |

## Typography

- 字体栈:`-apple-system, "PingFang SC", "Microsoft YaHei", "Inter", system-ui, sans-serif`(中文友好)
- 字距:主标题 `letter-spacing: -0.02em`,正文不调整
- 数字:`font-variant-numeric: tabular-nums`(rank_score、耗时、计数对齐)
- 字号节奏:
  - 主品牌标题(header) 30px / 700
  - section 标题(`### 🎯 系统产物...`) 18px / 600
  - 正文 14px / 400 / `line-height: 1.65`
  - caption 12px / 400 / `text-muted`

## 全局布局

- 主区限宽:`max-width: 1280px`,`margin: 0 auto`
- 区块垂直间距:`32px`
- 卡片:`padding: 20px`、`border-radius: 10px`、`border: 1px solid #E7E5E4`、**不加阴影**(影子是淘宝感,边框是高级感)
- 卡片 hover:边框过渡到 `#A8A29E`(stone-400)
- 重要卡片(配对表、当前结果):边框升级为 `1px solid #C9A876` + 极淡 `box-shadow: 0 1px 0 rgba(184,152,92,0.08)`

## 金属拉丝高光(点状,仅 4 处)

### 1. Header 品牌区

```
背景: linear-gradient(135deg, #FFFFFF 0%, #FAFAF9 60%, #F5E6CC 100%)
顶部边: 1px solid transparent
底部边: 1px solid #E7E5E4
内部: 标题 "Ecommerce CopyGen Studio" 30px/700,主色 text-primary,
      "Studio" 三字单独 color: gold-deep + font-weight: 800
副标题: "LLM 驱动的电商广告生成与诊断平台" 14px / text-muted
```

### 2. 主 CTA 按钮(▶ 运行 / 🔍 搜索)

```css
background: linear-gradient(180deg, #D4B27D 0%, #B8985C 50%, #8B6F3D 100%);
border: 1px solid #8B6F3D;
box-shadow:
  inset 0 1px 0 rgba(255, 255, 255, 0.3),  /* 上方拉丝高光 */
  inset 0 -1px 0 rgba(0, 0, 0, 0.1);        /* 下方阴影 */
color: #1C1917;
font-weight: 600;
hover: 渐变略上移 5%,光更亮
```

注意 Streamlit 原生 button 通过 `button[kind="primary"]` 选择器命中,需注入 CSS。

### 3. sac.steps 进度

- 完成 dot:`#B8985C` 实心
- 当前 dot:`#D4B27D` 实心 + 1px ring `#8B6F3D`
- 未到达 dot:`#D6D3D1`(stone-300)
- 连接线:完成段 `#C9A876`,未完成段 `#E7E5E4`

(若 sac.steps 不开放配色,降级为不动,只改 `colored_header` 的 color_name 配套)

### 4. 模式切换 segmented(调试 / 演示)

```css
选中项 background: linear-gradient(180deg, #D4B27D, #B8985C);
选中项 color: #1C1917;
未选中 background: #FFFFFF;
未选中 color: #78716C;
border: 1px solid #C9A876;
```

## 区块改造清单

按改动顺序:

1. **`.streamlit/config.toml`** — `primaryColor = "#B8985C"`、`textColor = "#1C1917"`、`backgroundColor = "#FAFAF9"`、`secondaryBackgroundColor = "#F5F5F4"`
2. **全局 CSS 注入**(`app.py` 顶部,`st.set_page_config` 之后立刻):
   - 字体栈
   - `body { background: #FAFAF9 }`
   - `.block-container { max-width: 1280px; padding-top: 1rem; }`
   - `[data-testid="stMetric"]` 卡片样式
   - `button[kind="primary"]` 金属渐变
   - `[data-testid="stExpander"]` 边框收紧
   - `[data-testid="stDataFrame"]` 行高/字号
   - `tabular-nums` 在数字 cell 上
3. **Header 重做** — 删 `colored_header(orange-70)`,改为自定义 `st.markdown` HTML 块,带渐变背景、品牌名 typography
4. **`render_status_strip`** — chip 配色从橙系改成金/灰二档(成功 = gold-mid,fallback = stone-500)
5. **`render_pair_table`** caption 上方加金色细分割线 + section 标题字号统一
6. **`render_consumer_view`** — 不动主调,只把"通用文案"灰 chip 配色微调,与整体协调
7. **演示模式 sidebar 隐藏 CSS** — 已有,不动
8. **Footer** — 新增极淡一行 `Ecommerce CopyGen Studio · v0.x · 2026` 居中,`color: #A8A29E`,12px,前后留 16px

## 不变项

- V5 主项目代码 `D:\hllm_env\KuaiSearch-main\` 不动
- C 端预览(`render_consumer_view`)的淘宝橙 `#FF6B35` 不动,只调灰阶 chip
- 不引入 Tailwind / 外部 CSS 框架
- 不做暗色模式 / 响应式
- 不动业务逻辑(_build_item_copy_pairs / V5 调用链)

## 影响范围

- `app.py` 新增约 100 行 CSS(单一 `<style>` 注入)
- `app.py` 替换 ~3 处渲染点(header / status_strip / footer)
- `.streamlit/config.toml` 改 4 行
- 不新增依赖

## 验证

1. `python -m py_compile app.py` 通过
2. `streamlit run app.py` 手动 smoke:
   - 调试模式:header 香槟金渐变出现,主标题 "Studio" 三字深古铜
   - "▶ 运行"按钮金属拉丝
   - sac.steps 完成态金色 dot
   - 模式切换 segmented 选中态金渐变
   - 切到演示模式:header 仍金调,但 C 端商品卡仍 `#FF6B35` 橙
   - 所有数字(rank_score、耗时)等宽对齐
   - 1280px 限宽,左右有留白
3. 视觉自查:页面没有任何位置出现纯黄/纯橙(C 端区除外),金属高光仅在 4 个声明位置出现

## YAGNI

- 不做暗色主题切换
- 不做响应式(假定 1280+ 桌面浏览器)
- 不替换 sac.steps / sui 组件实现
- 不做插画 / 图标库引入
- footer 不做版本号自动注入(写死)
