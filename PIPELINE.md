# NightLight Project · 完整 Pipeline

> 演示用整理稿 — 基于 `project/script/` 实际代码 + `data/result/` 实际产出

---

## 📥 阶段 0 · 数据获取与预处理

| 步骤 | 脚本 | 产出 |
|---|---|---|
| **下载 NTL 影像** | `multi_event_ntl_download_v2.ipynb` | 通过 GEE 拉取 25 个事件的 VNP46A2 daily NTL 影像（pre/post 各 ~30-60 天）→ Google Drive |
| **云量筛选** | 同上（GEE 内置 `Mandatory_Quality_Flag`） | 每个事件一份 `*_cloud_screening.csv` 记录每张影像 cloud fraction |
| **本地落盘** | `wget` 命令 | `data/processed/{Event_City}-VNP46A2-{pre,post}/*.tif` |
| **辅助数据** | `stage3_osm_download.py`、原始下载 | OSM 关键设施 POI、EAGLE-I 停电、Census ZCTA、IBTrACS 飓风轨迹 |

**事件配置（25 个）**：飓风 18 + 地震 2 + 冬季风暴 3 + 龙卷/Derecho 2，覆盖 17 美国州 + 土耳其

---

## 📊 阶段 1 · EDA（探索性分析）

**脚本**：`multi_event_eda.ipynb` + `MariaEDAV2.ipynb` / `MichaelEDAV2.ipynb` / `EarthquakeEDAV2.ipynb`

**核心定义**
- **Buffer**：airport 1250 m，其它（hospital / fire / police / power_plant / water_works …）750 m
- **R (Resilience Ratio)**：`mean_NTL_post / mean_NTL_pre`
- **NTL Change Ratio**：Zhang et al. 2023 像素级公式

**对每个事件做的步骤**
1. 扫描 pre/post tif → 全图均值时间序列
2. Overpass API 拉 POI → 按类型构建 differentiated buffer
3. Buffer vs Non-buffer NTL zonal 统计（按 facility type 分层）
4. 计算 R 曲线 + 跨事件汇总
5. 输出 `result/plots/{event_id}/`、`{event_id}_resilience_by_facility_type.csv`、`{event_id}_ntl_change_ratio.tif`

**关键发现（用作 Stage 2 motivation）**
- Buffer 区 R 持续高于 Non-buffer，但有 **floor effect**（暗区域绝对值低）
- Hospital/airport 信号最强；clinic/police 最弱
- 大城市（Miami）vs 小城市（Lake Charles）异质性明显

---

## 🧠 阶段 2 · 像素级预测建模

**脚本**：`stage2_25events.ipynb`（主）、`stage2_15events_modelD.ipynb`（Model D 验证）

**Pixel Panel**：每事件每像素一行，`pixel_panel.parquet`（~33.7K 像素 × 25 事件）

**算法**：RandomForest + XGBoost + Logistic（baseline），**Ensemble = 0.7 RF + 0.3 XGB**

**验证**：LOEO（Leave-One-Event-Out CV，25 折）

### 4 个 Model 变体

| 模型 | Feature 集 | 设计意图 | 代表 AUC（LOEO） |
|---|---|---|---|
| **Model A** | 全 17 个 feature（NTL 时间 + 邻近性 + 设施类型 + 城市规模） | 上限基准 | **0.967** |
| **Model B** | 仅 post-disaster NTL 行为 + 邻近性，去掉 pre-NTL | 测试是否依赖事前亮度 | 0.954 |
| **Model C** | A 的 features + Building Footprint Coverage（Microsoft Building Footprints） | 加入物理建筑密度做对照 | 0.942 |
| **Model D** | **纯 NTL 时间序列**，去掉所有 spatial proximity feature | 测纯卫星信号上限（最诚实） | **0.704** |

> Model A/B/C 的高 AUC 主要来自 spatial proximity "作弊"。**Model D 的 0.704 才是真信号**——证明在 500 m 分辨率下确实能从 NTL 时间模式检测出 commercial backup generator，但信号 modest。

### 输出
- `rf_final.pkl` / `xgb_final.pkl` / `rf_modelB.pkl` / `rf_modelC.pkl`
- `loeo_strict.csv` / `loeo_full.csv` / `loeo_modelB.csv` / `loeo_modelC.csv` / `loeo_15events.csv`
- 每事件一份 `{Event}_prob_map.{html,png,tif}`（25 个事件 × 多版本）→ 喂给 dashboard
- `feature_importance.csv` + `unified_metrics_ABC.csv`

---

## 🌍 阶段 3 · ZIP-code 空间回归

**脚本**：`stage3_zipcode_analysis.py`

**Panel**：1,002 个 zip-event 观测，22 个事件，19 个州 → `zipcode_panel.parquet`

**变量**
- `mean_prob` ：zip 内所有像素的 Stage 2 预测概率均值
- `fac_density`：关键设施密度（个/km²）
- 控制：`log_pop_density`、`log_income`（Census ACS 2022）、`wind_exposure`（IBTrACS Holland 衰减，仅飓风）
- 因变量备选：`outage_severity_zip`（EAGLE-I 县级停电按权重分配到 zip）
- `γᵢ` ：事件固定效应

### 9 个 Model 规范（来自 `regression_results.json`）

| # | 模型 | 关键变量 | 结果 | 解释 |
|---|---|---|---|---|
| **M1** | OLS 基线（无 Census） | `fac_density` → `mean_prob` | β=+0.096, p=9.6e-54, **R²=0.475** | 设施密度独立显著 |
| **M1+** | OLS + Census 控制 | + log_pop, log_income | β=+0.094, **R²=0.625** | 控制城市化后效应几乎不变 |
| **M2** | Moran's I（Isaias_Newark） | 空间自相关检验 | I=0.329, p=0.001 | 残差有空间自相关 |
| **M3** | **Spatial Error Model** (GM_Error_Het) | KNN k=5 权重 | λ=0.648, β=+0.097 | 控制空间依赖后系数稳健 |
| **M4** | 子样本：城市规模 | large/medium/small ZIPs 分组 | β 全部显著 | 各规模都成立 |
| **M5** | 飓风子样本 + 风场暴露 | + `wind_exposure` | β_fac=+0.094, β_wind=-1.13 (p=0.01) | 风场对预测概率有抑制效应 |
| **M6** | + NTL 下降幅度 | + `ntl_drop` | ntl_drop p=0.83 | NTL drop 不直接预测概率 |
| **M7** | 飓风全控制 | fac + wind + ntl_drop | R²=0.508 | 飓风模型最完整 |
| **M8** | **闭环验证** | `mean_prob` → `outage_severity` | β=-32.8, p=0.0002 | 高概率区停电更轻 |
| **M8+** | M8 + Census 控制 | 同上 | β=-13.1, **p=0.184**（消失） | **被人口密度吸收**——核心 honest reveal |

### 公平性补充（基础设施分布分析）

| Model | 因变量 | 关键发现 |
|---|---|---|
| **A** | fac_density ~ pop + income | 收入边际显著（p=0.096），人口密度主导 |
| **B** | + outage_severity | outage_severity 系数 ≈ 0（p=0.96）——设施分布与灾害风险无关 |
| **C** | 飓风 + wind_exposure | 收入显著为负（p=0.013）——低收入社区设施反而更多 |

**Equity gap（最有政策意义的发现）**

| Outage 等级 | N | fac_density | mean_prob |
|---|---|---|---|
| 低停电 | 301 | 0.53 | 0.425 |
| 中停电 | 300 | 0.32 | 0.403 |
| 高停电 | 300 | **0.20** | 0.310 |

> **高停电社区设施密度只有低停电的 38%**（t=5.12, p<0.0001）

---

## 📈 三阶段递进逻辑（讲故事用）

```
Stage 1 (EDA)       → "Buffer 像素确实更亮，但有 floor effect"
                            ↓
Stage 2 (Pixel ML)  → "能预测 P(backup)，纯 NTL AUC=0.704"
                            ↓ 把 prob map 聚合到 zip
Stage 3 (ZIP OLS)   → "设施密度→概率成立 (M1+)，
                       但概率→停电严重度被城市化吸收 (M8+)
                       政策侧仍有重要发现：38% gap"
```

---

## 🌐 阶段 4 · 可视化与产品化

| 组件 | 脚本/文件 |
|---|---|
| Stage 2 概率图导出为 dashboard 格式 | `stage3_export_all.py`、`export_geojson_facilities.py` |
| Dashboard 前端 | `nightlight-dashboard/` (Vue 3 + MapLibre GL JS) |
| 部署 | GitHub Actions → GitHub Pages |

---

## PPT 大纲建议（10-15 min）

1. Title
2. Problem + Why proxy labels
3. **Pipeline overview 图**（上面那个递进框）
4. Data sources（3 列：NASA / EAGLE-I / OSM）
5. EDA 结果（Resilience curve）
6. Stage 2 表（4 个 model 变体的 AUC）
7. Stage 2 Probability map demo（嵌视频）
8. Stage 3 表（M1 / M3 / M8 三选一展开）
9. 38% equity gap
10. Limitations + future
11. Q&A
