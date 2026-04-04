<template>
  <div class="results-page">
    <div class="results-inner">

      <!-- Page header -->
      <div class="page-header reveal">
        <div class="page-header__left">
          <div class="tag tag--cyan" style="margin-bottom:10px">Model Results</div>
          <h1 class="page-header__title">Prediction Performance</h1>
          <p class="page-header__desc">
            Leave-One-Event-Out cross-validation results, feature importance rankings,
            and per-event probability distributions from the RF + XGBoost ensemble.
          </p>
        </div>
      </div>

      <!-- ═══ LOEO AUC Table ═══ -->
      <div class="result-section reveal">
        <h2>LOEO Cross-Validation — AUC by Event</h2>
        <p style="color:var(--text-muted); font-size:14px; margin-bottom:10px">
          Each row = one held-out event. Switch models to compare feature ablation effects.
        </p>
        <div class="model-tabs" v-if="results && results.loeo_by_model">
          <button
            v-for="m in modelOptions"
            :key="m.key"
            class="model-tab"
            :class="{ active: activeModel === m.key }"
            @click="activeModel = m.key"
          >{{ m.label }}</button>
        </div>
        <div class="data-table" v-if="results">
          <table>
            <thead>
              <tr>
                <th>Held-Out Event</th>
                <th>RF AUC</th>
                <th>XGB AUC</th>
                <th>Logit AUC</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="r in activeLoeo" :key="r.held_out">
                <td>{{ r.held_out.replace('_', ' ') }}</td>
                <td class="mono" :style="{ color: aucColor(r.rf_auc) }">{{ r.rf_auc.toFixed(3) }}</td>
                <td class="mono" :style="{ color: aucColor(r.xgb_auc) }">{{ r.xgb_auc.toFixed(3) }}</td>
                <td class="mono" :style="{ color: aucColor(r.logit_auc) }">{{ r.logit_auc.toFixed(3) }}</td>
              </tr>
            </tbody>
            <tfoot v-if="results">
              <tr style="border-top:2px solid var(--border)">
                <td><strong>Mean</strong></td>
                <td class="mono"><strong>{{ meanAuc('rf_auc') }}</strong></td>
                <td class="mono"><strong>{{ meanAuc('xgb_auc') }}</strong></td>
                <td class="mono"><strong>{{ meanAuc('logit_auc') }}</strong></td>
              </tr>
            </tfoot>
          </table>
        </div>
      </div>

      <!-- ═══ LOEO AUC Bar Chart ═══ -->
      <div class="result-section reveal" v-if="results">
        <h2>AUC Comparison by Event</h2>
        <div class="chart-card">
          <svg viewBox="0 0 600 240" class="result-svg" preserveAspectRatio="xMidYMid meet">
            <!-- Grid -->
            <line v-for="v in [0.5, 0.6, 0.7, 0.8, 0.9]" :key="v"
              x1="120" x2="580" :y1="210 - (v - 0.4) * 400" :y2="210 - (v - 0.4) * 400"
              stroke="rgba(255,255,255,0.06)" stroke-width="0.5"/>
            <text v-for="v in [0.5, 0.6, 0.7, 0.8, 0.9]" :key="'t'+v"
              x="115" :y="214 - (v - 0.4) * 400" text-anchor="end" font-size="9" font-family="monospace" fill="var(--text-dim)">{{ v }}</text>

            <!-- Bars per event -->
            <template v-for="(r, i) in results.loeo" :key="r.held_out">
              <!-- RF bar -->
              <rect :x="130 + i * 50" :width="14" rx="2"
                :y="210 - (r.rf_auc - 0.4) * 400"
                :height="(r.rf_auc - 0.4) * 400"
                fill="rgba(0,229,160,0.6)"/>
              <!-- XGB bar -->
              <rect :x="146 + i * 50" :width="14" rx="2"
                :y="210 - (r.xgb_auc - 0.4) * 400"
                :height="(r.xgb_auc - 0.4) * 400"
                fill="rgba(0,212,255,0.5)"/>
              <!-- Label -->
              <text :x="145 + i * 50" y="225" text-anchor="middle" font-size="7" fill="var(--text-muted)"
                :transform="`rotate(-35, ${145 + i * 50}, 225)`">
                {{ r.held_out.split('_')[0] }}
              </text>
            </template>

            <!-- Legend -->
            <rect x="130" y="5" width="10" height="8" fill="rgba(0,229,160,0.6)" rx="1"/>
            <text x="144" y="12" font-size="9" fill="var(--text-muted)">Random Forest</text>
            <rect x="230" y="5" width="10" height="8" fill="rgba(0,212,255,0.5)" rx="1"/>
            <text x="244" y="12" font-size="9" fill="var(--text-muted)">XGBoost</text>
          </svg>
        </div>
      </div>

      <!-- ═══ Feature Importance ═══ -->
      <div class="result-section reveal" v-if="results">
        <h2>Feature Importance — Top 10</h2>
        <p style="color:var(--text-muted); font-size:14px; margin-bottom:16px">
          Average importance across RF and XGBoost. <code>log_pre_ntl</code> dominates — baseline
          brightness is the strongest predictor of buffer classification.
        </p>
        <div class="fi-list">
          <div v-for="(f, i) in results.feature_importance.slice(0, 10)" :key="f.feature" class="fi-item">
            <span class="fi-rank mono">{{ i + 1 }}</span>
            <span class="fi-name mono">{{ f.feature }}</span>
            <div class="fi-bar-wrap">
              <div class="fi-bar fi-bar--rf" :style="{ width: (f.rf_imp / maxImp * 100) + '%' }" />
              <div class="fi-bar fi-bar--xgb" :style="{ width: (f.xgb_imp / maxImp * 100) + '%' }" />
            </div>
            <span class="fi-val mono">{{ (f.avg_imp * 100).toFixed(1) }}%</span>
          </div>
          <div class="fi-legend">
            <span><span class="fi-dot" style="background:rgba(0,229,160,0.6)"/> RF</span>
            <span><span class="fi-dot" style="background:rgba(0,212,255,0.5)"/> XGB</span>
          </div>
        </div>
      </div>

      <!-- ═══ Probability Distribution ═══ -->
      <div class="result-section reveal" v-if="results">
        <h2>Predicted Probability Distribution by Event</h2>
        <p style="color:var(--text-muted); font-size:14px; margin-bottom:16px">
          Summary statistics of the ensemble model's per-pixel P(backup_power) output.
        </p>
        <div class="data-table">
          <table>
            <thead>
              <tr>
                <th>Event</th>
                <th>Pixels</th>
                <th>Min</th>
                <th>P25</th>
                <th>Median</th>
                <th>P75</th>
                <th>Max</th>
                <th>P &gt; 50%</th>
                <th>P &gt; 80%</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="ev in sortedProbEvents" :key="ev.id">
                <td><span class="dot" :style="{ background: ev.color }" />{{ ev.subtitle.split(',')[0] }}</td>
                <td class="mono">{{ results.prob_stats[ev.id]?.n?.toLocaleString() }}</td>
                <td class="mono">{{ results.prob_stats[ev.id]?.min }}</td>
                <td class="mono">{{ results.prob_stats[ev.id]?.p25 }}</td>
                <td class="mono" style="font-weight:600">{{ results.prob_stats[ev.id]?.p50 }}</td>
                <td class="mono">{{ results.prob_stats[ev.id]?.p75 }}</td>
                <td class="mono">{{ results.prob_stats[ev.id]?.max }}</td>
                <td class="mono" style="color:var(--green)">{{ results.prob_stats[ev.id]?.above_50 }}</td>
                <td class="mono" style="color:var(--cyan)">{{ results.prob_stats[ev.id]?.above_80 }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- ═══ Model Comparison ═══ -->
      <div class="result-section reveal" v-if="results && results.model_comparison">
        <h2>Model Comparison — 4 Ablation Variants</h2>
        <p style="color:var(--text-muted); font-size:14px; margin-bottom:16px">
          Each variant tests a different hypothesis about what drives prediction accuracy.
          Model D (pure NTL) isolates the nighttime light behavioral signal from spatial proximity.
        </p>
        <div class="data-table">
          <table>
            <thead>
              <tr>
                <th>Model</th>
                <th>Description</th>
                <th>LOEO AUC</th>
                <th>Std</th>
                <th>F1</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(m, key) in results.model_comparison" :key="key">
                <td style="font-weight:600">{{ m.name }}</td>
                <td style="color:var(--text-muted); font-size:12px">{{ modelDesc[key] }}</td>
                <td class="mono" :style="{ color: aucColor(m.mean_auc), fontWeight: 600 }">{{ m.mean_auc.toFixed(3) }}</td>
                <td class="mono">{{ m.std.toFixed(3) }}</td>
                <td class="mono">{{ m.f1.toFixed(3) }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- ═══ Takeaway ═══ -->
      <div class="takeaway reveal" v-if="results">
        <div class="takeaway__label">KEY RESULTS</div>
        <p class="takeaway__text">
          With CRS-corrected spatial features, the full model achieves <strong>mean LOEO AUC of
          {{ meanAuc('rf_auc') }}</strong> (RF) across {{ results.loeo.length }} held-out events.
          <strong>log_dist</strong> (distance to nearest facility) is the dominant feature,
          contributing +0.269 AUC over the pure NTL model.
          Model D (pure NTL behavior, no spatial features) achieves <strong>AUC 0.700</strong>,
          confirming that nighttime light changes alone carry a genuine backup power detection
          signal — above random but requiring spatial context for high accuracy.
        </p>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { EVENTS } from '@/data/events.js'

const BASE = import.meta.env.BASE_URL
const results = ref(null)
const activeModel = ref('A')
const modelOptions = [
  { key: 'A', label: 'Model A — Full' },
  { key: 'B', label: 'Model B — Post-only' },
  { key: 'C', label: 'Model C — +Buildings' },
  { key: 'D', label: 'Model D — Pure NTL' },
]

onMounted(async () => {
  try {
    const res = await fetch(`${BASE}data/results_summary.json`)
    if (res.ok) results.value = await res.json()
  } catch { /* ignore */ }

  // Scroll reveal
  revealObs = new IntersectionObserver(
    entries => entries.forEach(e => e.target.classList.toggle('visible', e.isIntersecting)),
    { threshold: 0.1 }
  )
  setTimeout(() => document.querySelectorAll('.results-page .reveal').forEach(el => revealObs.observe(el)), 100)
})

let revealObs
onUnmounted(() => revealObs?.disconnect())

const modelDesc = {
  model_a: 'All 17 features: NTL behavior + spatial proximity + city controls',
  model_b: 'Removes pre-disaster NTL (tests post-disaster signal alone)',
  model_c: 'Adds OSM building footprint coverage features',
  model_d: 'Only NTL magnitude/change features — no spatial proximity at all',
}

const activeLoeo = computed(() => {
  if (!results.value) return []
  if (results.value.loeo_by_model && results.value.loeo_by_model[activeModel.value]) {
    return results.value.loeo_by_model[activeModel.value]
  }
  return results.value.loeo
})

const maxImp = computed(() => {
  if (!results.value) return 1
  return Math.max(...results.value.feature_importance.map(f => Math.max(f.rf_imp, f.xgb_imp)))
})

const sortedProbEvents = computed(() =>
  EVENTS.filter(ev => results.value?.prob_stats?.[ev.id])
)

function meanAuc(key) {
  if (!activeLoeo.value.length) return '—'
  const vals = activeLoeo.value.map(r => r[key])
  return (vals.reduce((a, b) => a + b, 0) / vals.length).toFixed(3)
}

function aucColor(v) {
  if (v >= 0.75) return 'var(--green)'
  if (v >= 0.65) return 'var(--cyan)'
  if (v >= 0.55) return 'var(--text-bright)'
  return '#ff6b6b'
}
</script>

<style scoped>
.reveal {
  opacity: 0;
  transform: translateY(30px);
  transition: opacity 0.7s cubic-bezier(0.16, 1, 0.3, 1), transform 0.7s cubic-bezier(0.16, 1, 0.3, 1);
}
.reveal.visible {
  opacity: 1;
  transform: translateY(0);
}

.results-page {
  min-height: calc(100vh - var(--nav-h));
  background: transparent;
}
.results-inner {
  max-width: 1000px;
  margin: 0 auto;
  padding: 40px 32px 80px;
  display: flex;
  flex-direction: column;
  gap: 32px;
}

.page-header__title { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
.page-header__desc { font-size: 14px; color: var(--text-muted); max-width: 560px; line-height: 1.65; }

.result-section h2 {
  font-size: 18px; font-weight: 700; color: var(--text-bright);
  margin-bottom: 8px;
}

/* Model tabs */
.model-tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 14px;
  flex-wrap: wrap;
}
.model-tab {
  padding: 6px 14px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: transparent;
  color: var(--text-muted);
  font-family: var(--font-head);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.04em;
  cursor: pointer;
  transition: all var(--t-fast);
}
.model-tab:hover {
  color: var(--text-bright);
  border-color: var(--border-2);
}
.model-tab.active {
  color: var(--cyan);
  background: var(--cyan-dim);
  border-color: rgba(0,212,255,0.3);
}

/* Tables */
.data-table { overflow-x: auto; }
.data-table table { width: 100%; border-collapse: collapse; font-size: 13px; }
th {
  font-family: var(--font-head); font-size: 10px; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-dim);
  text-align: left; padding: 8px 12px;
  background: rgba(7,21,37,0.5); border-bottom: 1px solid var(--border);
}
td { padding: 8px 12px; border-bottom: 1px solid rgba(18,42,69,0.4); color: var(--text); }
tr:hover td { background: rgba(12,30,53,0.4); }
tfoot td { background: rgba(7,21,37,0.3); }
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 7px; vertical-align: middle; }
code {
  font-family: var(--font-mono); font-size: 12px;
  background: var(--bg-3); border: 1px solid var(--border);
  border-radius: 3px; padding: 1px 5px; color: var(--green);
}

/* Chart card */
.chart-card {
  background: rgba(7,21,37,0.5);
  border: 1px solid rgba(18,42,69,0.4);
  border-radius: var(--radius-lg);
  padding: 16px;
}
.result-svg { width: 100%; height: auto; display: block; }

/* Feature importance */
.fi-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.fi-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 0;
}
.fi-rank {
  width: 20px;
  font-size: 11px;
  color: var(--text-dim);
  text-align: right;
}
.fi-name {
  width: 180px;
  font-size: 12px;
  color: var(--cyan);
  flex-shrink: 0;
}
.fi-bar-wrap {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.fi-bar {
  height: 6px;
  border-radius: 3px;
  transition: width 0.5s cubic-bezier(0.16, 1, 0.3, 1);
}
.fi-bar--rf { background: rgba(0,229,160,0.6); }
.fi-bar--xgb { background: rgba(0,212,255,0.5); }
.fi-val {
  width: 45px;
  font-size: 11px;
  color: var(--text-muted);
  text-align: right;
}
.fi-legend {
  display: flex;
  gap: 16px;
  margin-top: 8px;
  font-size: 11px;
  color: var(--text-muted);
}
.fi-dot {
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 2px;
  margin-right: 4px;
  vertical-align: middle;
}

/* Takeaway */
.takeaway {
  background: linear-gradient(135deg, rgba(255,170,0,0.08), rgba(255,100,0,0.05));
  border: 1px solid rgba(255,170,0,0.25);
  border-left: 4px solid #ffaa00;
  border-radius: var(--radius-lg);
  padding: 24px 28px;
}
.takeaway__label {
  font-family: var(--font-head);
  font-size: 12px; font-weight: 700;
  letter-spacing: 0.16em; color: #ffaa00;
  margin-bottom: 12px;
}
.takeaway__text {
  font-size: 15px; line-height: 1.8; color: var(--text-bright);
}
.takeaway__text strong { color: #ffaa00; }

@media (max-width: 800px) {
  .results-inner { padding: 24px 16px 60px; }
  .page-header__title { font-size: 22px; }
  .fi-name { width: 120px; font-size: 11px; }
  .fi-val { width: 38px; font-size: 10px; }
  .takeaway { padding: 16px 18px; }
  .takeaway__text { font-size: 13px; }
}

@media (max-width: 480px) {
  .results-inner { padding: 20px 12px 60px; gap: 24px; }
  .page-header__title { font-size: 20px; }
  .result-section h2 { font-size: 15px; }
  .fi-item { gap: 6px; }
  .fi-name { width: 90px; font-size: 10px; }
  .fi-rank { width: 16px; font-size: 10px; }
  .data-table { margin: 0 -12px; }
  .data-table table { font-size: 11px; }
  th, td { padding: 6px 8px; }
}
</style>
