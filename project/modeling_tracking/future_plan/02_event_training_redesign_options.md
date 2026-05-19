# Event Training Redesign Options (Concept Only)

## Objective
本文件仅做“事件训练方案改造”构想，不执行代码与训练切换。目标是评估如果改变事件集合或训练组织方式，结果会如何变化、代价是什么。

## Current Baseline
- 当前主线：六事件联合建模（Maria, Michael, PR Earthquake, Ida, Laura, Irma）
- 当前风险：跨模型方向冲突、事件异质性强、样本构成对系数敏感

## Option A: Keep 6 Events, Strengthen Features
### Design
- 不改事件集合，只强化特征（OSM设施、天气、空间相关控制、同样本锁定）

### Expected Result
- 工程可控、可追溯性高
- `in_buffer` 方向冲突有机会收敛，但不保证完全解决

### Cost / Risk
- 成本低到中
- 若冲突主因是事件结构差异，单纯加特征可能不足

## Option B: Expand Event Set (More Hurricanes / Blackouts)
### Design
- 在现有6事件外增加同类型飓风/停电事件，构建更大样本
- 要求每个新增事件满足 pre/post tif、POI、cloud screening 最低门槛

### Expected Result
- 提升外推能力，降低单事件主导效应
- 可能让 MixedLM 与 Cox 结论更稳定

### Cost / Risk
- 数据准备成本高
- 新事件质量不一致会引入新噪声

## Option C: Regional/Context-Stratified Training
### Design
- 按区域或场景分层：岛屿(PR) vs 大陆、城市核心 vs 郊区/农村
- 每层分别建模，再做汇总比较

### Expected Result
- 明显提升可解释性（减少“一个系数解释所有场景”的偏差）
- 可定位“在哪类场景发电机信号更强”

### Cost / Risk
- 每层样本量变小，显著性可能下降
- 报告复杂度提升

## Option D: Leave-One-Event-Out (LOEO) Generalization
### Design
- 每次留出一个事件做外推验证，其他事件训练
- 汇总外推性能与关键系数稳定性

### Expected Result
- 直接回答“模型能否泛化到未见事件”
- 可识别高杠杆事件（去掉后结论变化最大者）

### Cost / Risk
- 计算成本中等
- 若事件异质性过高，外推性能可能明显下降

## Recommended Next Milestone
推荐组合：`Option A + Option D`，并在下一里程碑逐步加入 `Option B`
- 先在当前六事件上完成特征与规格收敛（A）
- 立即执行 LOEO 验证稳健性（D）
- 再扩展新事件做外推增强（B）

## Practical Decision Rules
1. 若 A 后仍方向冲突：优先上 D（判定是否是个别事件驱动）。  
2. 若 D 显示泛化差：启动 B 扩充事件。  
3. 若 B 后仍冲突：采用 C 分层建模并在论文中给分层结论。  

## Expected Paper-level Impact
- `A` 提升内部效度（internal validity）
- `D` 提升外部效度（external validity）
- `B + C` 提升结论的可迁移性与情景解释力
