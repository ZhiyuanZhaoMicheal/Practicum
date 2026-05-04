<template>
  <div class="slides">
    <!-- Fixed animated background -->
    <div class="slides__bg" aria-hidden="true">
      <div class="slides__earth" />
      <div class="slides__overlay" />
      <div class="slides__scanline" />
    </div>

    <!-- Progress bar -->
    <div class="slides__progress" :style="{ width: progress + '%' }" />

    <!-- ═══ 1 · Title ═══ -->
    <section class="slide slide--hero">
      <div class="slide__inner slide__inner--center">
        <div class="hero__label">Final Presentation · 2026</div>
        <h1 class="hero__title">
          Can We See Generators<br />
          <span class="hero__accent">from Space?</span>
        </h1>
        <p class="hero__sub">
          Detecting backup power from nighttime satellite imagery during disasters.
        </p>
        <p class="hero__authors">
          <strong>Zhiyuan Zhao</strong> · <strong>Qiushi Yu</strong>
        </p>
        <p class="hero__collab">University of Pennsylvania</p>
        <div class="hero__scroll" @click="scrollNext">
          <span class="hero__scroll-text">Scroll to begin</span>
          <div class="hero__scroll-arrow">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
              <path d="M4 8L10 14L16 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
        </div>
      </div>
    </section>

    <!-- ═══ 2 · The Question (Introduction) ═══ -->
    <section class="slide">
      <div class="slide__inner">
        <div class="seg-tag reveal">01 · Introduction</div>
        <h2 class="slide__title reveal">The Question</h2>
        <p class="slide__lead reveal">
          When a hurricane knocks out the grid, hospitals, airports, and fire stations
          stay lit on backup generators. <strong>Can satellites tell us where backup
          generators are running during a blackout?</strong>
        </p>
        <p class="slide__body reveal">
          Knowing this would matter for disaster response, energy equity, and
          infrastructure planning — but no public database of generators exists.
        </p>
      </div>
    </section>

    <!-- ═══ 3 · The Challenge ═══ -->
    <section class="slide">
      <div class="slide__inner">
        <div class="seg-tag reveal">01 · Introduction</div>
        <h2 class="slide__title reveal">The Core Challenge</h2>
        <p class="slide__lead reveal">
          With no ground-truth labels, a model can't learn "what a generator looks like
          from space." We need another way in.
        </p>

        <div class="three-col">
          <div class="callout reveal" style="transition-delay:0.1s">
            <div class="callout__num">No DB</div>
            <div class="callout__txt">No public registry of generator locations or status</div>
          </div>
          <div class="callout reveal" style="transition-delay:0.2s">
            <div class="callout__num">No labels</div>
            <div class="callout__txt">Without labels, supervised ML has nothing to learn</div>
          </div>
          <div class="callout reveal callout--accent" style="transition-delay:0.3s">
            <div class="callout__num">Proxy</div>
            <div class="callout__txt">Critical facilities (hospitals, airports, fire) are <em>required</em> to have backup — use them as proxy labels</div>
          </div>
        </div>
      </div>
    </section>

    <!-- ═══ 4 · Aims ═══ -->
    <section class="slide">
      <div class="slide__inner">
        <div class="seg-tag reveal">02 · Aims</div>
        <h2 class="slide__title reveal">What We Set Out to Do</h2>

        <div class="aims-grid">
          <div class="aim reveal" style="transition-delay:0.1s">
            <div class="aim__num">RQ1</div>
            <div class="aim__txt">Do facility-adjacent pixels behave differently during outages? (Interpretive)</div>
          </div>
          <div class="aim reveal" style="transition-delay:0.2s">
            <div class="aim__num">RQ2</div>
            <div class="aim__txt">Can we predict backup-power probability from NTL alone? (Predictive)</div>
          </div>
          <div class="aim reveal" style="transition-delay:0.3s">
            <div class="aim__num">RQ3</div>
            <div class="aim__txt">Do zip codes with more facilities experience less severe outages? (Equity)</div>
          </div>
        </div>

        <div class="stats-bar reveal" style="transition-delay:0.4s">
          <div v-for="s in heroStats" :key="s.label" class="stat">
            <span class="stat__value">{{ s.value }}</span>
            <span class="stat__label">{{ s.label }}</span>
          </div>
        </div>
      </div>
    </section>

    <!-- ═══ 5 · Data ═══ -->
    <section class="slide">
      <div class="slide__inner">
        <div class="seg-tag reveal">03 · Methodology</div>
        <h2 class="slide__title reveal">Data Pipeline</h2>
        <p class="slide__lead reveal">
          Three open datasets, joined at the pixel level.
        </p>

        <div class="three-col">
          <div class="data-card reveal" style="transition-delay:0.1s">
            <div class="data-card__tag">Satellite</div>
            <div class="data-card__name">NASA Black Marble<br/>VNP46A2</div>
            <div class="data-card__desc">Daily nighttime light, 500 m resolution, atmosphere-corrected. Fetched via Google Earth Engine.</div>
          </div>
          <div class="data-card reveal" style="transition-delay:0.2s">
            <div class="data-card__tag">Outages</div>
            <div class="data-card__name">DOE EAGLE-I</div>
            <div class="data-card__desc">15-minute county-level customer outage records — defines our event windows.</div>
          </div>
          <div class="data-card reveal" style="transition-delay:0.3s">
            <div class="data-card__tag">Proxy Labels</div>
            <div class="data-card__name">OpenStreetMap<br/>Critical Facilities</div>
            <div class="data-card__desc">Hospitals, airports, fire stations, police, power plants — pixels within 750 m get the "near generator" label.</div>
          </div>
        </div>

        <p class="slide__caption reveal" style="transition-delay:0.4s">
          Result: a panel of <strong>~33,700 labeled pixels</strong> across <strong>25 events</strong> in 17 U.S. states + Turkey, 2016–2023.
        </p>
      </div>
    </section>

    <!-- ═══ 6 · EDA — Recovery Chart ═══ -->
    <section class="slide">
      <div class="slide__inner">
        <div class="seg-tag reveal">03 · Methodology</div>
        <h2 class="slide__title reveal">Resilience Ratio — Buffer vs. Non-Buffer</h2>
        <p class="slide__lead reveal">
          For every event, we compute <strong>R = NTL during outage / NTL before outage</strong>.
          Buffer pixels (within 750 m of a facility) consistently retain more brightness — but a "floor effect" complicates the picture.
        </p>

        <div class="chart-wrap reveal" style="transition-delay:0.2s">
          <RecoveryChart :event="mariaEvent" :height="280" />
        </div>

        <p class="slide__caption reveal" style="transition-delay:0.3s">
          Hurricane Maria, San Juan — buffer zones recover faster but residual gap persists for months.
        </p>
      </div>
    </section>

    <!-- ═══ 7 · Modeling Strategy ═══ -->
    <section class="slide">
      <div class="slide__inner">
        <div class="seg-tag reveal">03 · Methodology</div>
        <h2 class="slide__title reveal">Two-Stage Modeling</h2>

        <div class="two-col">
          <div class="model-block reveal" style="transition-delay:0.1s">
            <div class="model-block__head">Interpretive</div>
            <div class="model-block__sub">Does the proxy signal exist?</div>
            <ul class="model-block__list">
              <li>OLS — effect on ΔNTL</li>
              <li>MixedLM — clustering correction</li>
              <li>Logistic — damage probability</li>
              <li>Cox PH — recovery speed</li>
            </ul>
            <div class="model-block__note">Triangulation across 4 specifications</div>
          </div>
          <div class="model-block reveal" style="transition-delay:0.2s">
            <div class="model-block__head">Predictive</div>
            <div class="model-block__sub">How much can NTL alone predict?</div>
            <ul class="model-block__list">
              <li>Model A — full feature set</li>
              <li>Model B — post-disaster only</li>
              <li>Model C — pre-event baseline</li>
              <li><strong>Model D — pure NTL temporal</strong></li>
            </ul>
            <div class="model-block__note">RF + XGB ensemble · LOEO cross-validation</div>
          </div>
        </div>
      </div>
    </section>

    <!-- ═══ 8 · Results — Statistical Triangulation ═══ -->
    <section class="slide">
      <div class="slide__inner">
        <div class="seg-tag reveal">04 · Results</div>
        <h2 class="slide__title reveal">Statistical Triangulation — Buffer Pixels Win</h2>

        <div class="result-row">
          <div class="result-card reveal" style="transition-delay:0.1s">
            <div class="result-card__metric">+2.8%</div>
            <div class="result-card__lbl">Less NTL decline</div>
            <div class="result-card__model">OLS · p = 0.020</div>
          </div>
          <div class="result-card reveal" style="transition-delay:0.2s">
            <div class="result-card__metric">0.68</div>
            <div class="result-card__lbl">Damage odds ratio</div>
            <div class="result-card__model">Logistic · 32% lower</div>
          </div>
          <div class="result-card reveal" style="transition-delay:0.3s">
            <div class="result-card__metric">1.13</div>
            <div class="result-card__lbl">Recovery hazard ratio</div>
            <div class="result-card__model">Cox PH · 13% faster</div>
          </div>
        </div>

        <div class="callout callout--warn reveal" style="transition-delay:0.4s; margin-top:24px">
          <div class="callout__num">⚠ Honest reveal</div>
          <div class="callout__txt">
            Leave-One-Event-Out CV drops AUC from <strong>0.73 → 0.45</strong> — statistical models
            don't generalize to unseen events. Land-use confounding is partially driving the
            in-sample effect. Motivates the predictive phase.
          </div>
        </div>
      </div>
    </section>

    <!-- ═══ 9 · Results — Predictive Models ═══ -->
    <section class="slide">
      <div class="slide__inner">
        <div class="seg-tag reveal">04 · Results</div>
        <h2 class="slide__title reveal">The Honest Generator Signal</h2>
        <p class="slide__lead reveal">
          Four model variants isolate what the algorithm <em>actually</em> learns from
          temporal NTL versus what it's borrowing from spatial context.
        </p>

        <div class="auc-grid">
          <div class="auc-cell reveal" style="transition-delay:0.05s">
            <div class="auc-cell__name">Model A</div>
            <div class="auc-cell__sub">Full features</div>
            <div class="auc-cell__val">0.967</div>
          </div>
          <div class="auc-cell reveal" style="transition-delay:0.1s">
            <div class="auc-cell__name">Model B</div>
            <div class="auc-cell__sub">Post-disaster</div>
            <div class="auc-cell__val">0.954</div>
          </div>
          <div class="auc-cell reveal" style="transition-delay:0.15s">
            <div class="auc-cell__name">Model C</div>
            <div class="auc-cell__sub">Pre-event baseline</div>
            <div class="auc-cell__val">0.942</div>
          </div>
          <div class="auc-cell auc-cell--hero reveal" style="transition-delay:0.2s">
            <div class="auc-cell__name">Model D</div>
            <div class="auc-cell__sub">Pure NTL temporal</div>
            <div class="auc-cell__val">0.704</div>
          </div>
        </div>

        <p class="slide__caption reveal" style="transition-delay:0.3s">
          <strong>0.704 is the truth.</strong> A modest but real signal — proves backup generators are
          detectable from space at 500 m, with the rest of A/B/C being spatial shortcuts.
        </p>
      </div>
    </section>

    <!-- ═══ 10 · Live Map Demo ═══ -->
    <section class="slide slide--demo">
      <div class="slide__inner slide__inner--full">
        <div class="demo-header">
          <div class="seg-tag">04 · Results · Live Demo</div>
          <h2 class="slide__title" style="margin:6px 0 0">Probability Maps — All 25 Events</h2>
          <p class="slide__caption" style="margin:6px 0 0">
            Per-pixel backup-power probability · {{ uniqueCities }} locations · click any pixel for details
          </p>
        </div>
        <div class="demo-frame">
          <iframe
            class="demo-iframe"
            :src="mapEmbedUrl"
            title="NightLight interactive map"
            loading="lazy"
          />
          <a :href="mapEmbedUrl" target="_blank" rel="noopener" class="demo-open">
            Open in new tab ↗
          </a>
        </div>
      </div>
    </section>

    <!-- ═══ 11 · Zip-Code Equity ═══ -->
    <section class="slide">
      <div class="slide__inner">
        <div class="seg-tag reveal">04 · Results</div>
        <h2 class="slide__title reveal">From Pixels to Policy — Zip-Code Equity</h2>
        <p class="slide__lead reveal">
          Across <strong>1,002 zip codes</strong> and 22 events, zip codes with more critical
          facilities experience less severe outages. Within-city analysis confirms the effect
          isn't just urbanization.
        </p>

        <div class="result-row">
          <div class="result-card reveal" style="transition-delay:0.1s">
            <div class="result-card__metric">R² = 0.475</div>
            <div class="result-card__lbl">Spatial regression fit</div>
            <div class="result-card__model">OLS + SEM · 1,002 ZIPs</div>
          </div>
          <div class="result-card reveal" style="transition-delay:0.2s">
            <div class="result-card__metric">p = 0.0001</div>
            <div class="result-card__lbl">Within-city facility effect</div>
            <div class="result-card__model">After controlling urbanization</div>
          </div>
          <div class="result-card result-card--equity reveal" style="transition-delay:0.3s">
            <div class="result-card__metric">38%</div>
            <div class="result-card__lbl">Facility density gap</div>
            <div class="result-card__model">High-outage vs. low-outage ZIPs</div>
          </div>
        </div>

        <p class="slide__caption reveal" style="transition-delay:0.4s">
          The most outage-prone communities have <strong>only 38%</strong> of the critical-facility
          density of the least-outage areas — an equity signal worth surfacing.
        </p>
      </div>
    </section>

    <!-- ═══ 12 · Conclusions ═══ -->
    <section class="slide">
      <div class="slide__inner">
        <div class="seg-tag reveal">05 · Conclusions</div>
        <h2 class="slide__title reveal">What We Learned</h2>

        <div class="takeaways">
          <div class="takeaway reveal" style="transition-delay:0.1s">
            <span class="takeaway__num">1</span>
            <div>
              <strong>Proxy labels work.</strong> Without ground truth, treating critical
              facilities as labels surfaces a real, statistically significant signal.
            </div>
          </div>
          <div class="takeaway reveal" style="transition-delay:0.2s">
            <span class="takeaway__num">2</span>
            <div>
              <strong>500 m is the ceiling.</strong> Pure NTL temporal AUC of 0.704 is
              meaningful but modest — at this resolution, generator signal blends with urban structure.
            </div>
          </div>
          <div class="takeaway reveal" style="transition-delay:0.3s">
            <span class="takeaway__num">3</span>
            <div>
              <strong>Commercial yes, residential no.</strong> Hospital and airport-scale
              generators are detectable; small residential units are below the noise floor.
            </div>
          </div>
          <div class="takeaway reveal" style="transition-delay:0.4s">
            <span class="takeaway__num">4</span>
            <div>
              <strong>Equity gap is measurable.</strong> Disaster-prone communities have
              systematically less backup-power infrastructure.
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ═══ 13 · Limitations + Future ═══ -->
    <section class="slide">
      <div class="slide__inner">
        <div class="seg-tag reveal">05 · Conclusions</div>
        <h2 class="slide__title reveal">Limitations & What's Next</h2>

        <div class="two-col">
          <div class="model-block reveal" style="transition-delay:0.1s">
            <div class="model-block__head">Limitations</div>
            <ul class="model-block__list">
              <li>Proxy labels ≠ ground truth (where generators <em>should</em> be, not where they <em>ran</em>)</li>
              <li>500 m / nightly cadence misses small or short events</li>
              <li>Cloud cover removes ~30% of usable observations</li>
              <li>Land-use confounding partially explains in-sample effects</li>
            </ul>
          </div>
          <div class="model-block model-block--accent reveal" style="transition-delay:0.2s">
            <div class="model-block__head">Future Work</div>
            <ul class="model-block__list">
              <li>≤ 100 m sensors (Luojia-1, future SDGSAT) to separate generators from urban structure</li>
              <li>Generator permit databases as actual ground truth</li>
              <li>Real-time disaster response dashboards</li>
              <li>Equity-aware infrastructure planning</li>
            </ul>
          </div>
        </div>
      </div>
    </section>

    <!-- ═══ 14 · Thank You ═══ -->
    <section class="slide slide--end">
      <div class="slide__inner slide__inner--center">
        <div class="hero__label">Thank You</div>
        <h2 class="end__title">Questions?</h2>
        <p class="hero__authors" style="margin-top:24px">
          <strong>Zhiyuan Zhao</strong> · <strong>Qiushi Yu</strong>
        </p>
        <p class="hero__collab">University of Pennsylvania</p>

        <div class="end__links">
          <RouterLink to="/map" class="end__link">Live Map →</RouterLink>
          <RouterLink to="/docs" class="end__link end__link--ghost">Full Documentation →</RouterLink>
        </div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { EVENTS } from '@/data/events.js'
import RecoveryChart from '@/components/RecoveryChart.vue'

const base = import.meta.env.BASE_URL
const mapEmbedUrl = computed(() => `${base}#/map`)

const mariaEvent = EVENTS.find(e => e.id === 'maria') || EVENTS[0]
const uniqueCities = new Set(EVENTS.map(ev => ev.subtitle.split(',')[0])).size
const heroStats = [
  { value: '25',           label: 'Disaster Events' },
  { value: '17 + 1',       label: 'States + Turkey' },
  { value: '~33.7K',       label: 'Labeled Pixels' },
  { value: '2016–23',      label: 'Study Period' },
]

// Scroll progress bar
const progress = ref(0)
let onScroll
function scrollNext() {
  const sections = document.querySelectorAll('.slide')
  for (const s of sections) {
    if (s.getBoundingClientRect().top > 10) {
      s.scrollIntoView({ behavior: 'smooth' })
      return
    }
  }
}

let observer
onMounted(() => {
  onScroll = () => {
    const max = document.documentElement.scrollHeight - window.innerHeight
    progress.value = max > 0 ? (window.scrollY / max) * 100 : 0
  }
  window.addEventListener('scroll', onScroll, { passive: true })
  onScroll()

  observer = new IntersectionObserver(
    entries => entries.forEach(e => {
      if (e.isIntersecting) e.target.classList.add('visible')
      else e.target.classList.remove('visible')
    }),
    { threshold: 0.15 }
  )
  document.querySelectorAll('.slides .reveal').forEach(el => observer.observe(el))
})
onUnmounted(() => {
  observer?.disconnect()
  if (onScroll) window.removeEventListener('scroll', onScroll)
})
</script>

<style scoped>
/* ───── Reveal animation (shared with HomeView) ───── */
.reveal {
  opacity: 0;
  transform: translateY(30px);
  transition: opacity 0.7s cubic-bezier(0.16, 1, 0.3, 1), transform 0.7s cubic-bezier(0.16, 1, 0.3, 1);
}
.reveal.visible {
  opacity: 1;
  transform: translateY(0);
}

/* ───── Background (fixed, like Home) ───── */
.slides {
  position: relative;
  flex: 1;
}
.slides__bg {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
}
.slides__earth {
  position: absolute;
  inset: 0;
  background: url('/earth-night.jpg') center center / cover no-repeat;
  animation: panEarth 90s linear infinite alternate;
}
@keyframes panEarth {
  0%   { transform: scale(1.15) translateX(-5%); }
  100% { transform: scale(1.15) translateX(5%); }
}
.slides__overlay {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse at center, rgba(3,13,26,0.25) 0%, rgba(3,13,26,0.7) 75%),
    linear-gradient(180deg, rgba(3,13,26,0.2) 0%, rgba(3,13,26,0.55) 100%);
}
.slides__scanline {
  position: absolute;
  left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, rgba(0,212,255,0.15), transparent);
  animation: scanline 8s linear infinite;
}
@keyframes scanline { 0% { top: -2px; } 100% { top: 100%; } }

/* ───── Progress bar ───── */
.slides__progress {
  position: fixed;
  top: var(--nav-h);
  left: 0;
  height: 2px;
  background: var(--cyan);
  box-shadow: 0 0 10px var(--cyan-glow);
  z-index: 50;
  transition: width 0.1s linear;
}

/* ───── Slide section (flow layout — no viewport snap) ───── */
.slide {
  position: relative;
  z-index: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: clamp(28px, 4vh, 56px) 32px;
}
.slide__inner {
  width: 100%;
  max-width: 1080px;
  display: flex;
  flex-direction: column;
  gap: 18px;
}
.slide__inner--center {
  align-items: center;
  text-align: center;
  max-width: 800px;
}
.slide__inner--full {
  max-width: 1280px;
  gap: 12px;
}

.seg-tag {
  font-family: var(--font-head);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--cyan);
  opacity: 0.9;
}
.slide__title {
  font-size: clamp(28px, 4vw, 44px);
  font-weight: 700;
  color: #fff;
  line-height: 1.15;
  letter-spacing: -0.01em;
}
.slide__lead {
  font-size: 19px;
  color: #d4e3f1;
  line-height: 1.7;
  max-width: 820px;
}
.slide__lead strong { color: #fff; }
.slide__body {
  font-size: 16px;
  color: #9cb3c9;
  line-height: 1.7;
  max-width: 720px;
}
.slide__caption {
  font-size: 14px;
  color: var(--text-muted);
  line-height: 1.6;
  max-width: 820px;
}
.slide__caption strong { color: var(--text-bright); }

/* ───── Hero (Title + End) ───── */
.hero__label {
  font-family: var(--font-head);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.2em;
  color: var(--cyan);
  text-transform: uppercase;
  opacity: 0.85;
  margin-bottom: 12px;
}
.hero__title {
  font-size: clamp(40px, 6vw, 68px);
  line-height: 1.05;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: -0.015em;
  animation: fadeUp 0.9s cubic-bezier(0.16,1,0.3,1) both;
}
.hero__accent {
  color: var(--cyan);
  text-shadow: 0 0 60px var(--cyan-glow);
}
.hero__sub {
  font-size: 18px;
  color: #b8cce0;
  line-height: 1.7;
  max-width: 600px;
  margin-top: 14px;
  animation: fadeUp 0.9s 0.15s cubic-bezier(0.16,1,0.3,1) both;
}
.hero__authors {
  font-family: var(--font-head);
  font-size: 16px;
  font-weight: 500;
  color: var(--text-bright);
  letter-spacing: 0.04em;
  margin-top: 22px;
  animation: fadeUp 0.9s 0.3s cubic-bezier(0.16,1,0.3,1) both;
}
.hero__authors strong { color: #fff; font-weight: 700; }
.hero__collab {
  font-family: var(--font-head);
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  margin-top: 4px;
  animation: fadeUp 0.9s 0.4s cubic-bezier(0.16,1,0.3,1) both;
}
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.hero__scroll {
  margin-top: 24px;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  color: var(--text-muted);
  transition: color var(--t-fast);
}
.hero__scroll:hover { color: var(--cyan); }
.hero__scroll-text {
  font-family: var(--font-head);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.hero__scroll-arrow { animation: bounce 2s ease infinite; }
@keyframes bounce {
  0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(8px); }
  60% { transform: translateY(4px); }
}

/* ───── Three / Two column layouts ───── */
.three-col {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-top: 14px;
}
.two-col {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 18px;
  margin-top: 12px;
}

/* ───── Callout ───── */
.callout {
  background: rgba(7,21,37,0.7);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(18,42,69,0.6);
  border-radius: var(--radius-lg);
  padding: 22px 24px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-height: 130px;
}
.callout--accent {
  border-color: rgba(0,212,255,0.35);
  background: rgba(0,212,255,0.06);
}
.callout--warn {
  border-color: rgba(255,180,80,0.4);
  background: rgba(255,180,80,0.05);
  flex-direction: row;
  align-items: flex-start;
  gap: 18px;
  min-height: 0;
}
.callout--warn .callout__num {
  flex-shrink: 0;
  color: #ffaa00;
  font-size: 14px;
}
.callout--warn .callout__txt {
  font-size: 14px;
  line-height: 1.7;
}
.callout__num {
  font-family: var(--font-head);
  font-size: 13px;
  font-weight: 700;
  color: var(--cyan);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.callout__txt {
  font-size: 14px;
  color: #b8cce0;
  line-height: 1.6;
}
.callout--accent .callout__txt { color: #d4e3f1; }

/* ───── Aims grid ───── */
.aims-grid {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-top: 14px;
}
.aim {
  display: flex;
  gap: 18px;
  align-items: center;
  background: rgba(7,21,37,0.7);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(18,42,69,0.6);
  border-radius: var(--radius-lg);
  padding: 18px 24px;
}
.aim__num {
  flex-shrink: 0;
  font-family: var(--font-head);
  font-size: 14px;
  font-weight: 700;
  color: var(--cyan);
  letter-spacing: 0.08em;
  width: 48px;
}
.aim__txt { color: #d4e3f1; font-size: 16px; line-height: 1.6; }
.aim__txt em { color: var(--cyan); font-style: italic; }

/* ───── Stats bar ───── */
.stats-bar {
  display: flex;
  justify-content: space-around;
  flex-wrap: wrap;
  margin-top: 24px;
  padding: 18px 0;
  border-top: 1px solid var(--border);
  border-bottom: 1px solid var(--border);
}
.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 6px 24px;
}
.stat__value {
  font-family: var(--font-head);
  font-size: 22px;
  font-weight: 700;
  color: var(--cyan);
  letter-spacing: 0.04em;
}
.stat__label {
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-muted);
}

/* ───── Data card ───── */
.data-card {
  background: rgba(7,21,37,0.7);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(18,42,69,0.6);
  border-radius: var(--radius-lg);
  padding: 22px 24px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 200px;
}
.data-card__tag {
  font-family: var(--font-head);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: var(--cyan);
}
.data-card__name {
  font-size: 18px;
  font-weight: 700;
  color: #fff;
  line-height: 1.25;
}
.data-card__desc {
  font-size: 13px;
  color: #9cb3c9;
  line-height: 1.6;
}

/* ───── Chart wrap ───── */
.chart-wrap {
  background: rgba(7,21,37,0.7);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(18,42,69,0.6);
  border-radius: var(--radius-lg);
  padding: 16px 20px;
  margin-top: 6px;
}

/* ───── Model block (two-col) ───── */
.model-block {
  background: rgba(7,21,37,0.7);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(18,42,69,0.6);
  border-radius: var(--radius-lg);
  padding: 22px 26px;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.model-block--accent {
  border-color: rgba(0,212,255,0.3);
  background: rgba(0,212,255,0.04);
}
.model-block__head {
  font-family: var(--font-head);
  font-size: 13px;
  font-weight: 700;
  color: var(--cyan);
  letter-spacing: 0.12em;
  text-transform: uppercase;
}
.model-block__sub {
  font-size: 14px;
  color: #b8cce0;
  margin-bottom: 6px;
}
.model-block__list {
  list-style: none;
  padding: 0;
  margin: 6px 0 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.model-block__list li {
  font-size: 14px;
  color: #d4e3f1;
  padding-left: 16px;
  position: relative;
  line-height: 1.55;
}
.model-block__list li::before {
  content: '›';
  position: absolute;
  left: 0;
  color: var(--cyan);
  opacity: 0.7;
}
.model-block__list li strong { color: #fff; }
.model-block__list li em { color: var(--cyan); font-style: italic; }
.model-block__note {
  margin-top: 10px;
  font-size: 11px;
  color: var(--text-muted);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  font-family: var(--font-head);
}

/* ───── Result row + cards ───── */
.result-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-top: 14px;
}
.result-card {
  background: rgba(7,21,37,0.7);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(18,42,69,0.6);
  border-radius: var(--radius-lg);
  padding: 24px 22px;
  text-align: center;
}
.result-card--equity {
  border-color: rgba(255,180,80,0.4);
  background: rgba(255,180,80,0.05);
}
.result-card__metric {
  font-family: var(--font-head);
  font-size: 36px;
  font-weight: 700;
  color: var(--cyan);
  letter-spacing: 0.02em;
  line-height: 1;
}
.result-card--equity .result-card__metric { color: #ffaa00; }
.result-card__lbl {
  font-size: 14px;
  color: #d4e3f1;
  margin-top: 10px;
  font-weight: 500;
}
.result-card__model {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 6px;
  letter-spacing: 0.06em;
  font-family: var(--font-head);
  text-transform: uppercase;
}

/* ───── AUC grid ───── */
.auc-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 14px;
  margin-top: 14px;
}
.auc-cell {
  background: rgba(7,21,37,0.6);
  border: 1px solid rgba(18,42,69,0.5);
  border-radius: var(--radius);
  padding: 18px 16px;
  text-align: center;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.auc-cell--hero {
  border-color: var(--cyan);
  background: rgba(0,212,255,0.08);
  box-shadow: 0 0 30px rgba(0,212,255,0.15);
}
.auc-cell__name {
  font-family: var(--font-head);
  font-size: 12px;
  font-weight: 700;
  color: #fff;
  letter-spacing: 0.08em;
}
.auc-cell__sub {
  font-size: 10px;
  color: var(--text-muted);
  letter-spacing: 0.04em;
  margin-bottom: 4px;
}
.auc-cell__val {
  font-family: var(--font-head);
  font-size: 28px;
  font-weight: 700;
  color: var(--cyan);
  margin-top: 4px;
}
.auc-cell--hero .auc-cell__val { font-size: 34px; }

/* ───── Demo slide (iframe) ───── */
.demo-header {
  text-align: center;
}
.demo-frame {
  position: relative;
  background: #030d1a;
  border: 1px solid var(--cyan);
  border-radius: var(--radius-lg);
  overflow: hidden;
  box-shadow: 0 0 40px rgba(0,212,255,0.15);
  height: clamp(420px, 70vh, 760px);
}
.demo-iframe {
  width: 100%;
  height: 100%;
  border: 0;
  display: block;
}
.demo-open {
  position: absolute;
  top: 12px;
  right: 12px;
  z-index: 5;
  background: rgba(3,13,26,0.85);
  backdrop-filter: blur(4px);
  border: 1px solid rgba(0,212,255,0.4);
  color: var(--cyan);
  text-decoration: none;
  font-family: var(--font-head);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.08em;
  padding: 8px 14px;
  border-radius: var(--radius);
  transition: all var(--t-fast);
}
.demo-open:hover {
  background: rgba(0,212,255,0.15);
  border-color: var(--cyan);
}

/* ───── Takeaways ───── */
.takeaways {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-top: 14px;
}
.takeaway {
  display: flex;
  gap: 18px;
  align-items: flex-start;
  background: rgba(7,21,37,0.6);
  border: 1px solid rgba(18,42,69,0.5);
  border-radius: var(--radius-lg);
  padding: 18px 24px;
}
.takeaway > div {
  font-size: 15px;
  color: #d4e3f1;
  line-height: 1.65;
}
.takeaway > div strong { color: #fff; }
.takeaway__num {
  flex-shrink: 0;
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--cyan-dim);
  border: 1px solid rgba(0,212,255,0.3);
  border-radius: 50%;
  color: var(--cyan);
  font-family: var(--font-head);
  font-size: 14px;
  font-weight: 700;
}

/* ───── End slide ───── */
.end__title {
  font-size: clamp(36px, 5vw, 56px);
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.01em;
  animation: fadeUp 0.9s cubic-bezier(0.16,1,0.3,1) both;
}
.end__links {
  display: flex;
  gap: 14px;
  margin-top: 36px;
  flex-wrap: wrap;
  justify-content: center;
}
.end__link {
  font-family: var(--font-head);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  text-decoration: none;
  padding: 12px 22px;
  border-radius: var(--radius);
  background: var(--cyan);
  color: #02101f;
  transition: all var(--t-fast);
}
.end__link:hover { box-shadow: 0 0 24px var(--cyan-glow); transform: translateY(-1px); }
.end__link--ghost {
  background: transparent;
  color: var(--cyan);
  border: 1px solid rgba(0,212,255,0.4);
}
.end__link--ghost:hover {
  background: rgba(0,212,255,0.08);
  box-shadow: none;
  transform: none;
}

/* ───── Responsive ───── */
@media (max-width: 900px) {
  .three-col, .auc-grid, .result-row { grid-template-columns: 1fr 1fr; }
  .two-col { grid-template-columns: 1fr; }
  .demo-frame { height: 60vh; min-height: 360px; }
}
@media (max-width: 600px) {
  .slide { padding: 40px 18px; }
  .three-col, .auc-grid, .result-row { grid-template-columns: 1fr; }
  .stats-bar { padding: 12px 0; gap: 4px; }
  .stat { padding: 6px 12px; }
  .stat__value { font-size: 18px; }
  .slide__lead { font-size: 16px; }
  .callout--warn { flex-direction: column; gap: 8px; }
  .end__links { flex-direction: column; align-items: stretch; }
}
</style>
