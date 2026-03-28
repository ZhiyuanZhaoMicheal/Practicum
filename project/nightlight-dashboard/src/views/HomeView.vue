<template>
  <div class="home">
    <!-- Fixed background for entire page -->
    <div class="home__bg" aria-hidden="true">
      <div class="hero__earth" />
      <div class="hero__overlay" />
      <div class="hero__scanline" />
    </div>

    <!-- ═══ Full-screen hero ═══ -->
    <section class="hero">

      <!-- Content -->
      <div class="hero__content">
        <div class="hero__label">NASA Black Marble · VIIRS VNP46A2</div>
        <h1 class="hero__title">
          Critical Infrastructure<br />
          <span class="hero__accent">Resilience Detection</span>
        </h1>
        <p class="hero__sub">
          Detecting backup generator activation at hospitals, airports, and power plants
          during major disasters — using nighttime satellite imagery from space.
        </p>
        <p class="hero__authors">
          <strong>Zhiyuan Zhao</strong> · University of Pennsylvania
        </p>
        <p class="hero__collab">
          Temple University &times; Arizona State University
        </p>
        <div class="hero__ctas">
          <RouterLink to="/map" class="btn btn--primary">Interactive Map</RouterLink>
          <RouterLink to="/docs" class="btn btn--outline">Technical Docs</RouterLink>
        </div>
      </div>

      <!-- Scroll indicator -->
      <div class="hero__scroll" @click="scrollToEvents">
        <span class="hero__scroll-text">Explore Events</span>
        <div class="hero__scroll-arrow">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M4 8L10 14L16 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
      </div>
    </section>

    <!-- ═══ Stats bar ═══ -->
    <section class="stats reveal">
      <div class="stats__inner">
        <div v-for="(s, i) in stats" :key="s.label" class="stat reveal" :style="{ transitionDelay: `${i * 0.08}s` }">
          <span class="stat__value">{{ s.value }}</span>
          <span class="stat__label">{{ s.label }}</span>
        </div>
      </div>
    </section>

    <!-- ═══ Event cards ═══ -->
    <section id="events" class="events-section">
      <div class="section-header reveal">
        <h2 class="section-title">Study Events</h2>
        <span class="tag tag--amber">{{ EVENTS.length }} Disasters · 2017–2023</span>
      </div>

      <div class="events-grid">
        <div
          v-for="(ev, i) in EVENTS"
          :key="ev.id"
          class="event-card reveal"
          :style="{ transitionDelay: `${i * 0.06}s` }"
          @click="goToMap(ev.id)"
        >
          <div class="event-card__accent" :style="{ background: ev.color }" />
          <div class="event-card__top">
            <span class="tag" :class="`tag--${ev.type}`">{{ ev.type }}</span>
            <span class="event-card__year mono">{{ ev.year }}</span>
          </div>
          <h3 class="event-card__name">{{ ev.name }}</h3>
          <p class="event-card__location">{{ ev.subtitle }}</p>
          <p class="event-card__desc">{{ ev.description }}</p>
          <div class="event-card__stats">
            <div class="event-card__stat">
              <span class="event-card__stat-label">Affected</span>
              <span class="event-card__stat-value mono">{{ ev.affectedUsers }}</span>
            </div>
            <div class="event-card__stat">
              <span class="event-card__stat-label">Outage</span>
              <span class="event-card__stat-value mono">{{ ev.outageDuration }}</span>
            </div>
            <div class="event-card__stat">
              <span class="event-card__stat-label">Facilities</span>
              <span class="event-card__stat-value mono">{{ ev.facilities.length }}</span>
            </div>
          </div>
          <div class="event-card__cta">View on Map →</div>
        </div>
      </div>
    </section>

    <!-- ═══ Pipeline ═══ -->
    <section class="pipeline-section">
      <div class="section-header reveal">
        <h2 class="section-title">Analysis Pipeline</h2>
        <span class="tag tag--cyan">8 Stages</span>
      </div>
      <div class="pipeline-grid">
        <RouterLink
          v-for="(step, i) in pipeline"
          :key="step.id"
          :to="`/docs/${step.id}`"
          class="pipeline-card reveal"
          :style="{ transitionDelay: `${i * 0.06}s` }"
        >
          <div class="pipeline-card__num mono">{{ step.num }}</div>
          <div class="pipeline-card__title">{{ step.title }}</div>
          <div class="pipeline-card__desc">{{ step.desc }}</div>
          <div class="pipeline-card__cta">Read Docs →</div>
        </RouterLink>
      </div>
    </section>
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { onMounted, onUnmounted } from 'vue'
import { EVENTS } from '@/data/events.js'

const router = useRouter()

function goToMap(eventId) {
  router.push({ path: '/map', query: { event: eventId } })
}

function scrollToEvents() {
  document.getElementById('events')?.scrollIntoView({ behavior: 'smooth' })
}

const totalFacilities = EVENTS.reduce((sum, ev) => sum + ev.facilities.length, 0)
const stats = [
  { value: String(EVENTS.length), label: 'Disaster Events' },
  { value: String(totalFacilities), label: 'Critical Facilities' },
  { value: '2017–23',             label: 'Study Period' },
  { value: 'VIIRS',               label: 'Satellite Sensor' },
  { value: '500m',                label: 'Resolution' },
  { value: 'LOEO',                label: 'Cross-Validation' },
]

const pipeline = [
  { id: 'overview',     num: '01', title: 'Project Overview',           desc: 'Research motivation, hypothesis, and study events' },
  { id: 'data',         num: '02', title: 'Data Collection',            desc: 'NASA Black Marble, EAGLE-I outages, OSM facilities' },
  { id: 'eda',          num: '03', title: 'Exploratory Analysis',       desc: 'Resilience ratio, floor effect, facility type comparison' },
  { id: 'interpretive', num: '04', title: 'Interpretive Modeling',      desc: 'OLS, MixedLM, Logistic, Cox PH — triangulation' },
  { id: 'features',     num: '05', title: 'Feature Engineering',        desc: '17 features addressing floor effect and city size' },
  { id: 'models',       num: '06', title: 'Predictive Models',          desc: 'RF + XGBoost ensemble, LOEO cross-validation' },
  { id: 'maps',         num: '07', title: 'Probability Maps',           desc: 'Per-pixel backup power probability heatmaps' },
  { id: 'repro',        num: '08', title: 'Reproducibility',            desc: 'Data sources, code, and references' },
]

// Scroll-triggered animations — reversible (re-animate on every scroll in/out)
let observer
onMounted(() => {
  observer = new IntersectionObserver(
    entries => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible')
        } else {
          entry.target.classList.remove('visible')
        }
      })
    },
    { threshold: 0.1 }
  )
  document.querySelectorAll('.reveal').forEach(el => observer.observe(el))
})
onUnmounted(() => observer?.disconnect())
</script>

<style scoped>
.home {
  flex: 1;
  position: relative;
}

/* Scroll reveal animation */
.reveal {
  opacity: 0;
  transform: translateY(30px);
  transition: opacity 0.7s cubic-bezier(0.16, 1, 0.3, 1), transform 0.7s cubic-bezier(0.16, 1, 0.3, 1);
}
.reveal.visible {
  opacity: 1;
  transform: translateY(0);
}

/* ═══════════════════════════════════════════ */
/* Hero — full viewport                       */
/* ═══════════════════════════════════════════ */
.hero {
  position: relative;
  height: calc(100vh - var(--nav-h));
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}

/* Fixed background for entire home page */
.home__bg {
  position: fixed;
  inset: 0;
  pointer-events: none;
  z-index: 0;
}
.hero__earth {
  position: absolute;
  inset: 0;
  background: url('/earth-night.jpg') center center / cover no-repeat;
  animation: panEarth 60s linear infinite alternate;
  transform-origin: center;
}
@keyframes panEarth {
  0%   { transform: scale(1.15) translateX(-5%); }
  100% { transform: scale(1.15) translateX(5%); }
}
.hero__overlay {
  position: absolute;
  inset: 0;
  background:
    radial-gradient(ellipse at center, rgba(3,13,26,0.15) 0%, rgba(3,13,26,0.55) 75%),
    linear-gradient(180deg, rgba(3,13,26,0.1) 0%, rgba(3,13,26,0.4) 100%);
}
.hero__scanline {
  position: absolute;
  left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, rgba(0,212,255,0.15), transparent);
  animation: scanline 8s linear infinite;
}
@keyframes scanline { 0% { top: -2px; } 100% { top: 100%; } }

/* Hero content */
.hero__content {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  gap: 16px;
  max-width: 720px;
  padding: 0 32px;
  animation: fadeUp 0.8s ease both;
}
.hero__label {
  font-family: var(--font-head);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.2em;
  color: var(--cyan);
  text-transform: uppercase;
  opacity: 0.8;
}
.hero__title {
  font-size: clamp(32px, 5vw, 56px);
  line-height: 1.1;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: -0.01em;
}
.hero__accent {
  color: var(--cyan);
  text-shadow: 0 0 60px var(--cyan-glow);
}
.hero__sub {
  font-size: 17px;
  color: #9cb3c9;
  line-height: 1.7;
  max-width: 560px;
}
.hero__authors {
  font-family: var(--font-head);
  font-size: 15px;
  font-weight: 500;
  letter-spacing: 0.04em;
  color: var(--text-bright);
}
.hero__authors strong {
  font-weight: 700;
  color: #ffffff;
}
.hero__collab {
  font-family: var(--font-head);
  font-size: 12px;
  font-weight: 500;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  margin-top: -8px;
}
.hero__ctas {
  display: flex;
  gap: 12px;
  margin-top: 8px;
}

/* Scroll indicator */
.hero__scroll {
  position: absolute;
  bottom: 32px;
  left: 50%;
  transform: translateX(-50%);
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  cursor: pointer;
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
.hero__scroll-arrow {
  animation: bounce 2s ease infinite;
}
@keyframes bounce {
  0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
  40% { transform: translateY(8px); }
  60% { transform: translateY(4px); }
}

/* ═══════════════════════════════════════════ */
/* Stats bar                                   */
/* ═══════════════════════════════════════════ */
.stats {
  border: none;
  background: transparent;
}
.stats__inner {
  display: flex;
  justify-content: space-around;
  flex-wrap: wrap;
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 80px;
}
.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: 18px 24px;
  border-right: none;
}
.stat:last-child { border-right: none; }
.stat__value {
  font-family: var(--font-head);
  font-size: 20px;
  font-weight: 700;
  color: var(--cyan);
  letter-spacing: 0.05em;
}
.stat__label {
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-muted);
}

/* ═══════════════════════════════════════════ */
/* Events section                              */
/* ═══════════════════════════════════════════ */
.events-section {
  padding: 56px 80px 80px;
  max-width: 1280px;
  margin: 0 auto;
  width: 100%;
  scroll-margin-top: var(--nav-h);
}
.section-header {
  display: flex;
  align-items: center;
  gap: 14px;
  margin-bottom: 28px;
}
.section-title {
  font-size: 16px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-bright);
}
.events-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 16px;
}
.event-card {
  position: relative;
  background: rgba(7,21,37,0.6);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(18,42,69,0.5);
  border-radius: var(--radius-lg);
  padding: 20px;
  cursor: pointer;
  overflow: hidden;
  transition: all var(--t-med);
}
.event-card:hover {
  border-color: rgba(28,58,90,0.7);
  background: rgba(12,30,53,0.7);
  transform: translateY(-2px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.event-card__accent {
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  opacity: 0.7;
}
.event-card__top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}
.event-card__year { font-size: 12px; color: var(--text-muted); }
.event-card__name { font-size: 16px; font-weight: 600; margin-bottom: 3px; }
.event-card__location { font-size: 12px; color: var(--text-muted); margin-bottom: 10px; }
.event-card__desc { font-size: 13px; color: var(--text-muted); line-height: 1.5; margin-bottom: 16px; }
.event-card__stats {
  display: flex;
  gap: 16px;
  padding-top: 12px;
  border-top: 1px solid rgba(18,42,69,0.5);
  margin-bottom: 12px;
}
.event-card__stat { display: flex; flex-direction: column; gap: 2px; }
.event-card__stat-label { font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-dim); }
.event-card__stat-value { font-size: 13px; font-weight: 500; color: var(--text-bright); }
.event-card__cta {
  font-family: var(--font-head);
  font-size: 11px;
  letter-spacing: 0.08em;
  color: var(--cyan);
  opacity: 0;
  transform: translateX(-4px);
  transition: all var(--t-fast);
}
.event-card:hover .event-card__cta {
  opacity: 1;
  transform: translateX(0);
}

/* ═══════════════════════════════════════════ */
/* Pipeline                                    */
/* ═══════════════════════════════════════════ */
.pipeline-section {
  padding: 32px 80px 80px;
  max-width: 1280px;
  margin: 0 auto;
  width: 100%;
}
.pipeline-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}
.pipeline-card {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 18px;
  background: rgba(7,21,37,0.5);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(18,42,69,0.4);
  border-radius: var(--radius-lg);
  text-decoration: none;
  color: inherit;
  transition: all var(--t-med);
  cursor: pointer;
}
.pipeline-card:hover {
  border-color: rgba(0,212,255,0.3);
  background: rgba(12,30,53,0.6);
  transform: translateY(-2px);
}
.pipeline-card__num {
  font-size: 11px;
  color: var(--cyan);
  letter-spacing: 0.14em;
  font-weight: 600;
}
.pipeline-card__title {
  font-family: var(--font-head);
  font-size: 14px;
  font-weight: 600;
  color: var(--text-bright);
}
.pipeline-card__desc {
  font-size: 12px;
  color: var(--text-muted);
  line-height: 1.5;
  flex: 1;
}
.pipeline-card__cta {
  font-family: var(--font-head);
  font-size: 11px;
  letter-spacing: 0.06em;
  color: var(--cyan);
  opacity: 0;
  transition: opacity var(--t-fast);
  margin-top: 4px;
}
.pipeline-card:hover .pipeline-card__cta {
  opacity: 1;
}

/* ═══════════════════════════════════════════ */
/* Responsive                                  */
/* ═══════════════════════════════════════════ */
@media (max-width: 900px) {
  .hero__globe { display: none; }
  .hero__content { padding: 0 24px; }
  .stats__inner { padding: 0 24px; }
  .events-section { padding: 40px 24px; }
  .pipeline-section { padding: 24px 24px 60px; }
  .pipeline-grid { grid-template-columns: repeat(2, 1fr); }
}
</style>
