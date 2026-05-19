# NightLight · 卫星夜间灯光检测灾后备用电源

> Capstone 项目交接文档
> 作者：Zhiyuan Zhao · Qiushi Yu (2026 spring)
> 用途：从 VIIRS 夜间灯光卫星影像 + 关键设施 POI，识别灾后备用发电机部署模式

🌍 [English README](README.md)

## 一句话总结

跨 25 个灾害事件（飓风 / 地震 / 冬季风暴等），在 500 m 像素尺度上预测"灾后哪些像素是备用电源在工作"（LOEO AUC = 0.704），然后在 ZIP 尺度上回归"设施密度 vs 停电严重度"。

完整方法论与结果见 [MODELS.md](MODELS.md)，pipeline 概览见 [PIPELINE.md](PIPELINE.md)（注：PIPELINE.md 部分文件引用是旧名，权威叙事以 MODELS.md 为准）。

---

## 作者分工

- **Stage 0（NTL 下载）**、**Stage 1（EDA）**、**Stage 1.5（解释性建模——OLS / MixedLM / Logit / Cox PH）** 在 `project/modeling/` 下，**作者：Qiushi Yu**。代码原貌合并入仓，入口见 `project/modeling/README.md`。
- **Stage 2（像素级预测）**、**Stage 3（ZIP 级回归）**、dashboard、出图与 report 资产，**作者：Zhiyuan Zhao**。

**接手 `project/modeling/` 之前先修一个坑**——legacy 三个结果写出脚本（`legacy/02_fit_ols_mixed.py`、`legacy/03_fit_logit.py`、`legacy/05_fit_cox.py`）使用了 `pd.concat([old, new]).to_csv(...)` 的 **append 模式**而不是 overwrite。重跑会静默地往 CSV 末尾追加重复行。迭代前先改成覆盖或加版本号输出。

---

## 仓库布局

```
.
├── MODELS.md                          ← 方法学 + 结果（权威）
├── PIPELINE.md                        ← Pipeline 概览
├── README.md                          ← 本文档
│
├── project/
│   ├── modeling/                      ← Stage 1.5 · OLS / MixedLM / Logit / Cox PH（作者 Qiushi Yu）
│   │   ├── pipeline_lib.py                共用函数库——模型拟合 + IO + 出图
│   │   ├── pipelines/                     当前活跃入口（01 in-sample / 02 cross-event / 03 exploration）
│   │   ├── legacy/                        原始 4 个模型脚本（02_fit_ols_mixed / 03_fit_logit / 05_fit_cox …）
│   │   ├── output/                        结果 CSV（ols_results / mixedlm_results / logit_results / cox_results）
│   │   ├── config/                        JSON 配置（events_6/10.json、model_defaults.json …）
│   │   ├── experimental/、support/、pixel_data/
│   │   └── README.md                      他写的 pipeline 说明
│   │
│   ├── modeling_tracking/             ← 建模过程与 issue 日志（Qiushi Yu）
│   │
│   ├── script/                        ← 所有分析脚本（已清理，全部为当前版本）
│   │   ├── multi_event_ntl_download_v2.ipynb    Stage 0 · 通过 GEE 下载 25 个事件 NTL
│   │   ├── multi_event_eda.ipynb                Stage 1 · EDA / buffer / resilience curves
│   │   ├── stage2_25events.ipynb                Stage 2 · 训练像素级模型（主 notebook）
│   │   ├── stage2_15events_modelD.ipynb         Stage 2 · Model D 独立验证
│   │   ├── regen_modelD_prob_maps.py            Stage 2 · 重生成 25 个 prob_map_modelD.tif
│   │   ├── run_modelD_loeo_25events.py          Stage 2 · LOEO 25 事件交叉验证
│   │   ├── make_modelD_loeo_heatmap.py          Stage 2 · 渲染 LOEO heatmap
│   │   ├── miami_dade_visualization.py          Stage 2 · Miami-Dade 发电机执照叠加图
│   │   ├── miami_dade_groundtruth_validation.py Stage 2 · R/C 分类 ground-truth
│   │   ├── stage3_events.py                     Stage 3 · 事件定义与窗口
│   │   ├── stage3_ntl_download.py               Stage 3 · ZIP 级 NTL 时序下载
│   │   ├── stage3_osm_download.py               Stage 3 · OSM POI 抓取（含 km² 面积）
│   │   ├── stage3_export_all.py                 Stage 3 · 批量导出回归输入
│   │   ├── stage3_zipcode_analysis_modelD.py    Stage 3 · 构建 zipcode_panel_modelD.parquet
│   │   ├── stage3_modelD_full_regressions.py    Stage 3 · Model 1–9 主回归
│   │   ├── stage3_modelD_extra_regressions.py   Stage 3 · 补充回归 + 反向公平性
│   │   ├── regen_pre_figures.py                 出图 · docs/pre_figures/ 全套
│   │   ├── year_end_show.py                     出图 · Year End Show 4 张 + 元数据
│   │   └── data/                                 各事件云量筛选 CSV
│   │
│   ├── data/
│   │   ├── raw/                       ← 原始数据（1.2 GB，需要单独传输）
│   │   │   ├── Outage_Dataset_R1/     EAGLE-I 停电记录 2014-2023
│   │   │   ├── POI/                   OSM 关键设施
│   │   │   ├── counties/              Census 县级 shapefile
│   │   │   ├── zcta520/               Census ZCTA 2020 shapefile
│   │   │   ├── acs_zcta_2022.csv      ACS 人口/收入
│   │   │   ├── ibtracs_NA.csv         IBTrACS 飓风轨迹
│   │   │   ├── generator.csv          通用发电机执照
│   │   │   ├── generator_houston_dallas.csv  TX 发电机执照子集
│   │   │   ├── stage3_event_configs.json     Stage 3 事件配置
│   │   │   └── POI/                          区域专用设施提取脚本（TX + PR 含 HIFLD 重试逻辑）
│   │   │
│   │   ├── processed/                 Stage 0 落盘的 NTL TIFs，25 事件 × pre/post
│   │   │   └── {Event}-VNP46A2-{pre,post}/*.tif
│   │   │
│   │   ├── dade_test/                 Miami-Dade 发电机执照 shapefile
│   │   │
│   │   └── result/
│   │       ├── stage2/                ← 当前结果（全部为 Model D）
│   │       │   ├── pixel_panel.parquet                Stage 2 输入像素表
│   │       │   ├── building_coverage_panel.parquet
│   │       │   ├── rf_modelD.pkl / xgb_modelD.pkl     训练好的模型
│   │       │   ├── feature_importance_modelD.csv
│   │       │   ├── loeo_modelD_25events.csv           LOEO 结果
│   │       │   ├── {Event}_prob_map_modelD.tif × 25   每事件概率图
│   │       │   ├── poi_cache/                         OSM 缓存
│   │       │   ├── miami_dade_pointwise_probs.csv     Miami-Dade 验证
│   │       │   ├── miami_dade_validation.json
│   │       │   └── precision_recall_results.csv
│   │       │
│   │       └── stage3/
│   │           ├── zipcode_panel_modelD.parquet       Stage 3 ZIP 面板
│   │           ├── regression_results_modelD_full.json
│   │           ├── regression_results_modelD_extra.json
│   │           └── poi_cache/
│   │
│   ├── nightlight-dashboard/          ← Vue 3 + MapLibre 交互式 dashboard
│   │   ├── src/                       Vue 源码
│   │   ├── public/data/               导出的 GeoJSON / TIF
│   │   ├── export_to_dashboard_modelD.py  把 Stage 2 概率图转成 dashboard 数据
│   │   ├── export_cloud_stats.py
│   │   ├── export_ntl_frames.py
│   │   ├── enrich_poi_names.py
│   │   └── package.json
│   │
│   └── export_geojson_facilities.py   POI → GeoJSON
│
├── docs/
│   ├── pre_figures/                   Pre 用图（10 张 Model D 时代）
│   ├── year_end_show/                 Year End Show 展板（4 张 + intro + QR）
│   ├── BackupGen_Zhiyuan_Qiushi/      最终 PPT + Report
│   ├── reference/                     参考文献 PDF
│   ├── Capstone_Speech_10min.docx     演讲稿（10 min 版）
│   ├── Capstone_Speech_with_Models.docx
│   ├── NTL_speaker_script.docx
│   ├── Stage3_汇报完整版.docx
│   ├── Models ZH v6.docx
│   ├── dashboardvideo.mov             Dashboard demo 录屏
│   ├── mappreview.png
│   └── pre_prompt.md
│
└── .gitignore
```

---

## 快速跑通 pipeline

> 假设原始数据已经在 `project/data/raw/` 和 `project/data/processed/`

```bash
# Stage 2 ：重新训练 Model D + 重生成 25 个事件的 prob_map
python project/script/regen_modelD_prob_maps.py

# Stage 2 ：LOEO 交叉验证（25 折，~30 min）
python project/script/run_modelD_loeo_25events.py
python project/script/make_modelD_loeo_heatmap.py

# Stage 3 ：构建 ZIP 面板 + 跑 Model 1–9
python project/script/stage3_zipcode_analysis_modelD.py
python project/script/stage3_modelD_full_regressions.py
python project/script/stage3_modelD_extra_regressions.py

# 出图（一键再生 docs/pre_figures/ 全套）
python project/script/regen_pre_figures.py

# Dashboard
cd project/nightlight-dashboard
npm install
npm run dev          # 本地开发
python export_to_dashboard_modelD.py   # 重生成 dashboard 数据
npm run build        # 部署用产物
```

---

## 几个已知坑（前人栽坑、后人乘凉）

1. **Earthquake_SanJuan 的 bbox fallback**
   `export_to_dashboard_modelD.py` 里 NTL pre/post 目录是用 substring 匹配的，`Earthquake_SanJuan` 会错配到 `Earthquake_Hatay-VNP46A2-pre/` 上（土耳其的经度 36°），导致波多黎各的 POI 全部被过滤掉。**解决方案在 `export_to_dashboard_modelD.py` 顶部的 `EXPLICIT_PRE_DIR` dict**。新增事件时要显式加进去。

2. **Model D 概率分布很窄（0.5–0.7）**
   Model D 没有 spatial proximity 特征，AUC 只有 0.704，预测概率挤在中间区间。Dashboard 的 MapView 里 circle layer 用的是**每事件分位数色标**（min/p10/p50/midpoint/p90/max），不是固定 0–1 色标，否则全图一片均匀色。改了着色逻辑要保持分位数严格单调（代码里有 guard）。

3. **CRS 在 Stage 2 一定要 EPSG:5070（US Equal Area）**
   早期版本用 EPSG:4326 算 km² 面积全错。Stage 3 OSM POI 抓完一定要 reproject 到 5070 再计算密度。

4. **M7 raw β 看起来很大（+123.8），别慌**
   这是 outage_severity ~ mean_prob 的回归。outage_severity 的 SD ≈ 274（log-customer-hours），mean_prob 的 SD ≈ 0.176，所以**标准化 β ≈ 0.08**，是个小到中等的效应。讲故事的时候用 β_std 别用 raw。

5. **公平性结论是描述性的，不是规范性的**
   `Equity gap`（高停电 ZIP 设施密度只有低停电 ZIP 的 63%）是**跨事件**的横截面共现模式。事件内（M5/M6/M7 with event FE）则是设施密度正向预测概率。这两个不矛盾，是 Simpson's paradox。规范性解读（"灾害风险分布不公平"）需要更强的因果假设——一个直接的替代假设是"经常受灾的地方从来没发展成城市"。

6. **R 是时间序列，不是单值**
   Resilience Ratio `R(t) = NTL_post(t) / BAU`，是按天计算的。slide 里偶尔写成 median BAU，代码里实际用 mean BAU——展示前对齐一下。

---

## 数据来源

| 数据 | 来源 | 许可 |
|---|---|---|
| VIIRS NTL VNP46A2 | NASA via Google Earth Engine | 公开 |
| EAGLE-I outages | DOE Oak Ridge National Lab | 公开 |
| OSM POI | OpenStreetMap via Overpass API | ODbL |
| Census ACS / ZCTA | US Census Bureau | 公开 |
| IBTrACS | NOAA | 公开 |
| Miami-Dade 发电机执照 | Miami-Dade Open Data Portal | 公开 |

---

## 不在 repo 里的大文件

为了保持 repo 体积合理，下面这几样没有版本控制。跑 pipeline 之前自己下载放到对应路径：

| 文件 | 放到 | 来源 |
|---|---|---|
| `tl_2020_us_zcta520.{shp,shx,dbf,prj,cpg}`（约 820 MB） | `project/data/raw/zcta520/` | [Census TIGER 2020 ZCTA](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layers=ZCTA520) |
| `tl_2020_us_county.{shp,shx,dbf,prj,cpg}`（约 130 MB） | `project/data/raw/counties/` | [Census TIGER 2020 Counties](https://www.census.gov/cgi-bin/geo/shapefiles/index.php?year=2020&layers=COUNTY) |
| `ibtracs_NA.csv`（约 57 MB） | `project/data/raw/` | [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) |
| 最终演示 PPT 和 demo 视频 | 不在 repo——找作者拿 | 见下方联系方式 |

---

## 联系方式

- Zhiyuan Zhao · michaelzhao576@gmail.com
- 项目仓库 / dashboard 链接见 [docs/Project Website Link.txt](docs/Project%20Website%20Link.txt)

如果有问题，先看 [MODELS.md](MODELS.md) 第 8–11 节（Limitations + FAQ-ish 部分），大部分常见疑问那里都有讨论。
