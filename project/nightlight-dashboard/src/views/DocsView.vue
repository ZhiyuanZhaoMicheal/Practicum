<template>
  <div class="docs-page">
    <div class="docs-inner">
      <div class="page-header reveal">
        <div class="tag tag--cyan" style="margin-bottom:10px">Technical Documentation</div>
        <h1 class="page-header__title">Methodology & Data</h1>
        <p class="page-header__desc">
          Overview of the analysis pipeline — from satellite data acquisition to predictive modeling.
          Click "Read More" on any section for detailed technical documentation.
        </p>
      </div>

      <div class="sections-grid">
        <RouterLink
          v-for="s in sections"
          :key="s.id"
          :to="`/docs/${s.id}`"
          class="section-card reveal"
          :style="{ transitionDelay: `${sections.indexOf(s) * 0.06}s` }"
        >
          <div class="section-card__num mono">{{ s.num }}</div>
          <h3 class="section-card__title">{{ s.title }}</h3>
          <p class="section-card__desc">{{ s.summary }}</p>
          <div class="section-card__tags" v-if="s.tags">
            <span v-for="t in s.tags" :key="t" class="tag tag--dim">{{ t }}</span>
          </div>
          <div class="section-card__cta">
            Read More <span class="arrow">→</span>
          </div>
        </RouterLink>
      </div>

      <!-- Quick stats -->
      <div class="quick-ref reveal">
        <div class="quick-ref__header">
          <span class="tag tag--amber">Quick Reference</span>
          <span style="font-size:11px; color:var(--text-muted)">Study Events</span>
        </div>
        <div class="data-table">
          <table>
            <thead><tr><th>Event</th><th>Location</th><th>Year</th><th>Type</th><th>Affected</th></tr></thead>
            <tbody>
              <tr v-for="ev in EVENTS" :key="ev.id">
                <td><span class="dot" :style="{ background: ev.color }" />{{ ev.name }}</td>
                <td>{{ ev.subtitle }}</td>
                <td class="mono">{{ ev.year }}</td>
                <td><span class="tag" :class="`tag--${ev.type}`" style="font-size:10px">{{ ev.type }}</span></td>
                <td class="mono">{{ ev.affectedUsers }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onUnmounted } from 'vue'
import { EVENTS } from '@/data/events.js'

let revealObs
onMounted(() => {
  revealObs = new IntersectionObserver(
    entries => entries.forEach(e => e.target.classList.toggle('visible', e.isIntersecting)),
    { threshold: 0.1 }
  )
  setTimeout(() => document.querySelectorAll('.docs-page .reveal').forEach(el => revealObs.observe(el)), 100)
})
onUnmounted(() => revealObs?.disconnect())

const sections = [
  {
    id: 'overview', num: '01', 
    title: 'Project Overview',
    summary: 'When hurricanes and earthquakes knock out the power grid, hospitals and airports rely on backup generators to keep the lights on. We use NASA satellite imagery that captures nighttime lights to detect which critical facilities actually maintained power during 9 major disasters across the U.S. and Turkey (2017–2023). The key insight: areas near facilities with generators stay brighter than their surroundings — we call this the "Resilience Advantage."',
    tags: ['VIIRS VNP46A2', 'Resilience Advantage'],
  },
  {
    id: 'data', num: '02', 
    title: 'Data Collection & Processing',
    summary: 'No public database of backup generators exists — so we turn to satellite imagery to detect them indirectly. We combine daily nighttime light images from NASA\'s Black Marble (500m resolution), power outage records from EAGLE-I, and critical facility locations from OpenStreetMap. Each pixel is labeled by its proximity to facilities (buffer zones at 750m/1250m), creating a panel of ~15,500 pixels across 9 events. This section also covers cloud masking, GEE data acquisition, and data quality challenges.',
    tags: ['VNP46A2', 'EAGLE-I', 'OSM', 'Buffer Zones', 'Cloud QC'],
  },
  {
    id: 'eda', num: '03', 
    title: 'Exploratory Data Analysis',
    summary: 'Before building models, we need to understand the data. We define the Resilience Ratio (R = NTL/BAU) and discover the "floor effect" — a critical confound where facilities in darker urban cores appear less resilient simply because they start dimmer. We show that different facility types (hospitals vs. fire stations) and city sizes (Miami vs. Lake Charles) produce systematically different resilience signals, with real data visualizations.',
    tags: ['Resilience Ratio', 'Floor Effect', 'Facility Groups', 'City Size'],
  },
  {
    id: 'interpretive', num: '04', 
    title: 'Interpretive Modeling',
    summary: 'Four statistical models (OLS, MixedLM, Logistic, Cox PH) attack the same hypothesis from different angles — triangulation. Key findings: buffer pixels show +2.8% less NTL decline (MixedLM, p=0.020), 32% lower damage odds (Logit, OR=0.68), and 13% faster recovery (Cox, HR=1.13). But land-use confounding partially explains the effect. LOEO cross-validation reveals severe generalization failure (AUC 0.73→0.45), motivating the predictive modeling phase.',
    tags: ['OLS', 'MixedLM', 'Logistic', 'Cox PH', 'Triangulation', 'NLCD Confounding'],
  },
  {
    id: 'features', num: '05', 
    title: 'Feature Engineering',
    summary: 'We engineer 17 predictive features from the raw data. A key challenge is the "floor effect" — in smaller cities, critical infrastructure is often located in darker areas, which confounds the analysis. We address this with city-level normalization and interaction terms that separate the genuine generator signal from urban brightness patterns.',
    tags: ['17 features', 'Floor effect', 'Interactions'],
  },
  {
    id: 'models', num: '06', 
    title: 'Predictive Models',
    summary: 'The interpretive phase validated in_buffer as a meaningful weak-supervision label (100% LOEO sign consistency). Now we flip the direction: instead of testing whether buffer zones are resilient, we predict which pixels have backup power from their NTL behavior. Random Forest + XGBoost ensemble with Leave-One-Event-Out cross-validation tests whether a generalizable "generator signature" exists.',
    tags: ['RF + XGB', 'LOEO', 'Weak Supervision', 'AUC'],
  },
  {
    id: 'maps', num: '07', 
    title: 'Probability Maps',
    summary: 'The trained model scores every urban pixel in each study area with a probability of backup power presence. These probability maps are exported as interactive heatmaps — the red-to-white gradient you see on the Map page. Brighter areas indicate higher predicted likelihood of generator-maintained power during the disaster.',
    tags: ['P_ensemble = 0.7×RF + 0.3×XGB'],
  },
  {
    id: 'repro', num: '08', 
    title: 'Reproducibility',
    summary: 'All data sources are publicly available (NASA, OpenStreetMap, DOE), and the full analysis pipeline is implemented in a single Jupyter notebook. This section provides the exact data product IDs, API endpoints, and key academic references needed to reproduce or extend the analysis.',
    tags: ['GEE', 'Jupyter', 'Open data'],
  },
]
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
.docs-page {
  min-height: calc(100vh - var(--nav-h));
  background: transparent;
}
.docs-inner {
  max-width: 1100px;
  margin: 0 auto;
  padding: 40px 32px 80px;
  display: flex;
  flex-direction: column;
  gap: 32px;
}

.page-header__title {
  font-size: 32px;
  font-weight: 700;
  margin-bottom: 10px;
  color: #ffffff;
}
.page-header__desc {
  font-size: 16px;
  color: #9cb3c9;
  max-width: 680px;
  line-height: 1.7;
}

/* Section cards grid */
.sections-grid {
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.section-card {
  display: flex;
  flex-direction: column;
  gap: 10px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 28px 32px;
  cursor: pointer;
  text-decoration: none;
  color: inherit;
  transition: all var(--t-med);
  position: relative;
  overflow: hidden;
}
.section-card:hover {
  border-color: var(--border-2);
  background: var(--bg-3);
  transform: translateY(-2px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.section-card__num {
  font-size: 13px;
  color: var(--cyan);
  letter-spacing: 0.14em;
  font-weight: 600;
}
.section-card__title {
  font-size: 22px;
  font-weight: 700;
  color: #ffffff;
}
.section-card__desc {
  font-size: 16px;
  color: #b8cce0;
  line-height: 1.8;
  flex: 1;
}
.section-card__tags {
  display: flex;
  gap: 5px;
  flex-wrap: wrap;
}
.tag--dim {
  font-size: 12px;
  padding: 4px 10px;
  background: rgba(0,212,255,0.08);
  border: 1px solid rgba(0,212,255,0.2);
  border-radius: 4px;
  color: var(--cyan);
  font-weight: 500;
}
.section-card__cta {
  font-family: var(--font-head);
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.08em;
  color: var(--cyan);
  margin-top: 10px;
  transition: all var(--t-fast);
}
.section-card__cta .arrow {
  display: inline-block;
  transition: transform var(--t-fast);
}
.section-card:hover .section-card__cta .arrow {
  transform: translateX(4px);
}

/* Quick reference table */
.quick-ref {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
}
.quick-ref__header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-3);
}
.data-table { overflow-x: auto; }
.data-table table { width: 100%; border-collapse: collapse; font-size: 13px; }
th {
  font-family: var(--font-head); font-size: 10px; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-dim);
  text-align: left; padding: 8px 12px;
  background: var(--bg-2); border-bottom: 1px solid var(--border);
}
td { padding: 8px 12px; border-bottom: 1px solid var(--border); color: var(--text); }
tr:last-child td { border-bottom: none; }
tr:hover td { background: var(--bg-3); }
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 7px; vertical-align: middle; }

@media (max-width: 900px) {
  .docs-inner { padding: 24px 16px 60px; }
  .sections-grid { grid-template-columns: 1fr; }
}
</style>
