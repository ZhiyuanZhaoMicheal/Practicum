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
    summary: 'Can satellites tell us where backup generators are running during a blackout? The challenge: no public database of generators exists, so there are no ground-truth labels to train on. Our solution: hospitals, airports, and fire stations are legally required to have generators — so we use them as proxy labels. If satellite pixels near these facilities stay brighter during outages, the brightness pattern itself becomes a detectable signal. We test this across 25 major disasters in 17 U.S. states and Turkey (2016–2023).',
    tags: ['VIIRS VNP46A2', '25 Events', 'Proxy Labels'],
  },
  {
    id: 'litreview', num: '02',
    title: 'Literature Review',
    summary: 'Previous research used nighttime lights to map where power went out during disasters. We invert the question: instead of detecting damage, we detect where lights stay on — a signal of backup power. Wang et al. (2018) pioneered satellite-based outage mapping; Zhang et al. (2023) extended it to damage assessment. Our contribution fills a gap: using the same data to find resilience rather than failure.',
    tags: ['Wang 2018', 'Zhang 2023', 'Energy Equity'],
  },
  {
    id: 'data', num: '03',
    title: 'Data Collection & Processing',
    summary: 'Since no generator registry exists, we build proxy labels from three open datasets: NASA Black Marble daily nighttime light images (500m resolution), DOE EAGLE-I power outage records, and OpenStreetMap critical facility locations. Pixels within 750m of a facility are labeled "near generator"; all others are the control group. This creates a panel of ~33,700 labeled pixels across 25 events — an imperfect but scalable alternative to ground-truth surveys. This section also covers cloud masking, Google Earth Engine workflows, and data quality challenges.',
    tags: ['VNP46A2', 'EAGLE-I', 'OSM', 'Buffer Zones', 'Cloud QC'],
  },
  {
    id: 'eda', num: '04',
    title: 'Exploratory Data Analysis',
    summary: 'Do pixels near proxy-labeled facilities actually behave differently during outages? We define the Resilience Ratio (R = NTL during outage / NTL before outage) and find that yes — but with complications. The "floor effect" means facilities in dimmer areas appear less resilient simply because they start darker. Facility type matters (hospitals vs. fire stations), and city size matters (Miami vs. Lake Charles). These patterns inform how we design the models.',
    tags: ['Resilience Ratio', 'Floor Effect', 'Facility Groups', 'City Size'],
  },
  {
    id: 'interpretive', num: '05',
    title: 'Interpretive Modeling',
    summary: 'We test whether the proxy labels carry a real signal using four statistical models (OLS, MixedLM, Logistic, Cox PH) — triangulation from different angles. Results: buffer pixels show +2.8% less NTL decline (p=0.020), 32% lower damage odds (OR=0.68), and 13% faster recovery (HR=1.13). However, land-use confounding partially explains the effect, and Leave-One-Event-Out cross-validation drops AUC from 0.73 to 0.45 — the statistical models don\'t generalize, motivating the predictive modeling phase.',
    tags: ['OLS', 'MixedLM', 'Logistic', 'Cox PH', 'Triangulation', 'NLCD Confounding'],
  },
  {
    id: 'features', num: '06',
    title: 'Feature Engineering',
    summary: 'We engineer 17 features to help models distinguish "generator running" from "just a bright area." The key challenge: our proxy labels are spatially biased — facilities cluster in certain urban zones. City-level normalization and interaction terms help separate the genuine generator brightness signal from the background urban brightness pattern.',
    tags: ['17 features', 'Floor effect', 'Interactions'],
  },
  {
    id: 'models', num: '07',
    title: 'Predictive Models & Probability Maps',
    summary: 'Can we predict which pixels have generators using only their nighttime light behavior — no location cheating? Four model variants answer this: Models A–C use spatial context (AUC ~0.967), but Model D uses only NTL temporal patterns (AUC 0.704). That 0.704 is the honest generator signal — real but modest at 500m resolution. The ensemble generates per-pixel backup power probability maps for all 25 events, validated against Miami-Dade generator permit records.',
    tags: ['RF + XGB', 'LOEO', '4 Model Variants', 'AUC 0.967'],
  },
  {
    id: 'stage3', num: '08',
    title: 'Zip-Code Analysis',
    summary: 'Scaling up from pixels to policy: do zip codes with more critical facilities (our proxy for generator density) experience less severe outages? Spatial regression across 1,002 zip codes and 22 events using Model D probabilities (R² = 0.747 with Census controls). Closed-loop validation against EAGLE-I outage records is positive (β=+123.8, p=0.012) — backup detection lines up with where outages actually strike, the natural deployment direction. Equity gap: high-outage zip codes have only 63% of the facility density of low-outage areas. Miami-Dade permit ground-truth confirms only commercial-scale generators are detectable at 500m.',
    tags: ['1,002 Zip Codes', 'Spatial Regression', 'OLS + SEM', 'Miami-Dade Validation'],
  },
  {
    id: 'conclusions', num: '09',
    title: 'Conclusions & Future Work',
    summary: 'The proxy-label approach works at commercial scale. Model D (pure NTL, AUC 0.704) achieves above-baseline probability at 83% of Miami-Dade commercial generator permits but only 14% of residential — the 500m / 25-hectare pixel cleanly resolves institutional backup behavior but loses household units in the noise. Stage 3 closed-loop validation aligns with the natural deployment direction: places that suffer worse outages show higher detected probability. Future sensors at ≤100m (Luojia-1 class) could push detection down to residential.',
    tags: ['Conclusions', 'Limitations', 'Future Work'],
  },
  {
    id: 'web', num: '10',
    title: 'Dashboard Development',
    summary: 'This interactive dashboard is built with Vue 3 + Vite, featuring MapLibre GL JS for WebGL-accelerated map rendering with per-event probability heatmaps, canvas-rendered facility icons, and satellite/vector basemap switching. The site supports mobile responsive layouts, scroll-triggered animations, and is deployed via GitHub Actions to GitHub Pages.',
    tags: ['Vue 3', 'Vite', 'MapLibre GL', 'GitHub Pages'],
  },
  {
    id: 'repro', num: '11',
    title: 'Reproducibility',
    summary: 'All data sources are publicly available (NASA, OpenStreetMap, DOE), and the full analysis pipeline is implemented in Jupyter notebooks. This section provides the exact data product IDs, API endpoints, and key academic references needed to reproduce or extend the analysis.',
    tags: ['GEE', 'Jupyter', 'Open data'],
  },
  {
    id: 'references', num: '12',
    title: 'References',
    summary: 'Full bibliography of cited works including Wang et al. (2018) on NASA Black Marble outage monitoring, Zhang et al. (2023) on damage assessment, Román et al. (2018) on the Black Marble product suite, and literature on energy equity and infrastructure resilience.',
    tags: ['Bibliography'],
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

@media (max-width: 600px) {
  .docs-inner { padding: 20px 12px 60px; gap: 24px; }
  .page-header__title { font-size: 24px; }
  .page-header__desc { font-size: 14px; }
  .section-card { padding: 18px 16px; }
  .section-card__title { font-size: 18px; }
  .section-card__desc { font-size: 14px; }
  .section-card__tags { gap: 4px; }
  .tag--dim { font-size: 10px; padding: 3px 7px; }
  .data-table { margin: 0 -12px; }
  .data-table table { font-size: 11px; }
  th, td { padding: 6px 8px; }
}
</style>
