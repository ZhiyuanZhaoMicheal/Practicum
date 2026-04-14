<template>
  <div class="home">
    <!-- Fixed animated background -->
    <div class="home__bg" aria-hidden="true">
      <div class="hero__earth" />
      <div class="hero__overlay" />
      <div class="hero__scanline" />
    </div>

    <!-- ═══ Full-screen hero ═══ -->
    <section class="hero">
      <div class="hero__content">
        <h1 class="hero__title">
          Can We See Generators<br />
          <span class="hero__accent">from Space?</span>
        </h1>
        <p class="hero__sub">
          Detecting backup power from nighttime satellite imagery during disasters.
        </p>
        <p class="hero__authors">
          Zhiyuan Zhao · <span style="opacity:0.6">University of Pennsylvania</span>
        </p>
      </div>

      <!-- Scroll indicator -->
      <div class="hero__scroll" @click="scrollToIntro">
        <span class="hero__scroll-text">Learn More</span>
        <div class="hero__scroll-arrow">
          <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
            <path d="M4 8L10 14L16 8" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </div>
      </div>
    </section>

    <!-- ═══ What We Did ═══ -->
    <section id="intro" class="intro-section">
      <div class="intro-inner">
        <div class="section-header reveal" style="transition-delay:0.15s">
          <h2 class="section-title">The Problem</h2>
        </div>
        <p class="intro-text reveal" style="transition-delay:0.25s">
          Millions of backup generators power hospitals, businesses, and homes during
          blackouts — yet <strong>no public database records where they are, who owns them,
          or whether they actually work</strong>. This data gap touches energy security,
          environmental justice, and disaster preparedness. But without ground-truth labels,
          a machine learning model can't simply learn "what a running generator looks like
          from space." We need another way in.
        </p>
      </div>
    </section>

    <!-- ═══ Our Approach ═══ -->
    <section class="approach-section">
      <div class="approach-inner">
        <div class="section-header reveal">
          <h2 class="section-title">Our Approach</h2>
        </div>
        <p class="intro-text reveal">
          Hospitals, airports, and fire stations are <strong>legally required</strong> to
          have backup generators — so we use them as <strong>proxy labels</strong>. If
          satellite pixels near these facilities stay brighter during outages, the
          brightness pattern itself becomes a learnable signal. Using daily NASA Black
          Marble imagery (500 m resolution) across {{ stateCount }} U.S. states and 25
          disaster events, we train models on these proxy labels and test whether the
          nighttime light signal alone — without knowing facility locations — can
          identify backup power. <strong>The answer: a real but modest signal (AUC 0.704),
          showing what remote sensing can reveal at current resolution, and where
          higher resolution or ground truth would be needed to go further.</strong>
        </p>

        <div class="stats reveal">
          <div class="stats__inner">
            <div v-for="(s, i) in stats" :key="s.label" class="stat" :style="{ transitionDelay: `${i * 0.08}s` }">
              <span class="stat__value">{{ s.value }}</span>
              <span class="stat__label">{{ s.label }}</span>
            </div>
          </div>
        </div>

        <!-- Docs card first -->
        <RouterLink to="/docs" class="feature-card reveal" style="margin-top:24px">
          <div class="feature-card__icon">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
              <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/>
              <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/>
              <line x1="9" y1="7" x2="17" y2="7"/>
              <line x1="9" y1="11" x2="17" y2="11"/>
              <line x1="9" y1="15" x2="13" y2="15"/>
            </svg>
          </div>
          <div class="feature-card__content">
            <h3 class="feature-card__title">Methodology & Results</h3>
            <p class="feature-card__desc">
              How far can nighttime lights take us? Follow the full pipeline from raw satellite
              data to predictive models. Four model variants isolate what the algorithm actually
              learns versus spatial shortcuts.
            </p>
            <div class="feature-card__details">
              <span class="tag tag--cyan">12 Sections</span>
              <span class="tag tag--dim">4 Model Variants</span>
            </div>
          </div>
          <div class="feature-card__cta">Read Docs →</div>
        </RouterLink>

        <!-- Map card — left text, right screenshot -->
        <RouterLink to="/map" class="map-card reveal" style="margin-top:16px; transition-delay:0.08s">
          <div class="map-card__text">
            <div class="map-card__icon">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M1 6v16l7-4 8 4 7-4V2l-7 4-8-4-7 4z"/>
                <path d="M8 2v16"/>
                <path d="M16 6v16"/>
              </svg>
            </div>
            <h3 class="map-card__title">Interactive Probability Map</h3>
            <p class="map-card__desc">
              Explore prediction results across {{ stateCount }} states. Pixel-level backup
              power probability with facility overlays.
            </p>
            <div class="map-card__tags">
              <span class="tag tag--cyan">{{ uniqueCities }} Locations</span>
              <span class="tag tag--dim">5 Basemaps</span>
            </div>
            <span class="map-card__cta">Explore Map →</span>
          </div>
          <div class="map-card__img">
            <img :src="`${base}map_preview.png`" alt="Map preview" />
          </div>
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
const base = import.meta.env.BASE_URL

function scrollToIntro() {
  document.getElementById('intro')?.scrollIntoView({ behavior: 'smooth' })
}


const stateCount = 19
const uniqueCities = new Set(EVENTS.map(ev => ev.subtitle.split(',')[0])).size
const stats = [
  { value: String(uniqueCities),  label: 'Study Areas' },
  { value: String(EVENTS.length), label: 'Disaster Events' },
  { value: '2016–23',             label: 'Study Period' },
  { value: 'VIIRS VNP46A2',      label: 'Data Source' },
]

// Scroll-driven hero fade + reveal animations
let observer
let scrollHandler

onMounted(() => {
  const heroContent = document.querySelector('.hero__content')
  const heroScroll = document.querySelector('.hero__scroll')

  // Hero fade on scroll
  scrollHandler = () => {
    const scrollY = window.scrollY
    const fadeEnd = window.innerHeight * 0.4
    const opacity = Math.max(0, 1 - scrollY / fadeEnd)
    if (heroContent) heroContent.style.opacity = opacity
    if (heroScroll) heroScroll.style.opacity = opacity
  }
  window.addEventListener('scroll', scrollHandler, { passive: true })

  // Reveal observer — threshold 0.15 means element must be 15% visible
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
    { threshold: 0.15 }
  )
  document.querySelectorAll('.reveal').forEach(el => observer.observe(el))
})
onUnmounted(() => {
  observer?.disconnect()
  if (scrollHandler) window.removeEventListener('scroll', scrollHandler)
})
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

/* ═══════════════════════════════════════════ */
/* Intro + Approach sections                   */
/* ═══════════════════════════════════════════ */
.intro-section {
  padding: 80px 80px 20px;
  max-width: 1280px;
  margin: 0 auto;
  width: 100%;
}
.intro-inner {
  max-width: 760px;
  margin: 0 auto;
}
.approach-section {
  padding: 0 80px 60px;
  max-width: 1280px;
  margin: 0 auto;
  width: 100%;
}
.approach-inner {
  max-width: 760px;
  margin: 0 auto;
}
.intro-text {
  font-size: 17px;
  color: #9cb3c9;
  line-height: 1.8;
  margin-bottom: 32px;
}
.intro-text strong {
  color: var(--text-bright);
}

/* ═══════════════════════════════════════════ */
/* Feature cards (Map + Docs)                  */
/* ═══════════════════════════════════════════ */
.feature-card {
  display: flex;
  align-items: flex-start;
  gap: 20px;
  padding: 28px 32px;
  background: rgba(7,21,37,0.6);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(18,42,69,0.5);
  border-radius: var(--radius-lg);
  text-decoration: none;
  color: inherit;
  cursor: pointer;
  transition: all var(--t-med);
  position: relative;
}
.feature-card:hover {
  border-color: rgba(0,212,255,0.3);
  background: rgba(12,30,53,0.7);
  transform: translateY(-2px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.feature-card__icon {
  flex-shrink: 0;
  width: 48px;
  height: 48px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--cyan-dim);
  border-radius: var(--radius);
  color: var(--cyan);
}
.feature-card__content {
  flex: 1;
  min-width: 0;
}
.feature-card__title {
  font-size: 20px;
  font-weight: 700;
  color: #ffffff;
  margin-bottom: 8px;
}
.feature-card__desc {
  font-size: 14px;
  color: #9cb3c9;
  line-height: 1.7;
  margin-bottom: 12px;
}
.feature-card__details {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.feature-card__cta {
  position: absolute;
  bottom: 16px;
  right: 24px;
  font-family: var(--font-head);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.06em;
  color: var(--cyan);
  opacity: 0;
  transition: opacity var(--t-fast);
}
.feature-card:hover .feature-card__cta {
  opacity: 1;
}

/* Map card — left/right layout */
.map-card {
  display: flex;
  background: rgba(7,21,37,0.6);
  backdrop-filter: blur(6px);
  border: 1px solid rgba(18,42,69,0.5);
  border-radius: var(--radius-lg);
  overflow: hidden;
  text-decoration: none;
  color: inherit;
  cursor: pointer;
  transition: all var(--t-med);
}
.map-card:hover {
  border-color: rgba(0,212,255,0.3);
  background: rgba(12,30,53,0.7);
  transform: translateY(-2px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}
.map-card__text {
  flex: 1;
  padding: 24px 28px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  min-width: 0;
}
.map-card__icon {
  width: 42px;
  height: 42px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--cyan-dim);
  border-radius: var(--radius);
  color: var(--cyan);
  flex-shrink: 0;
}
.map-card__title {
  font-size: 18px;
  font-weight: 700;
  color: #ffffff;
}
.map-card__desc {
  font-size: 13px;
  color: #9cb3c9;
  line-height: 1.6;
}
.map-card__tags {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.map-card__cta {
  font-family: var(--font-head);
  font-size: 13px;
  font-weight: 600;
  color: var(--cyan);
  opacity: 0;
  transition: opacity var(--t-fast);
  margin-top: auto;
}
.map-card:hover .map-card__cta {
  opacity: 1;
}
.map-card__img {
  width: 280px;
  flex-shrink: 0;
  border-left: 1px solid rgba(18,42,69,0.4);
}
.map-card__img img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  opacity: 0.8;
  transition: opacity var(--t-fast);
}
.map-card:hover .map-card__img img {
  opacity: 1;
}
@media (max-width: 600px) {
  .map-card { flex-direction: column; }
  .map-card__img { width: 100%; height: 180px; border-left: none; border-top: 1px solid rgba(18,42,69,0.4); }
  .map-card__text { padding: 20px 16px; }
  .map-card__cta { opacity: 1; }
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

@media (max-width: 600px) {
  .hero__content { padding: 0 16px; gap: 12px; }
  .hero__title { font-size: 28px; }
  .hero__sub { font-size: 14px; }
  .hero__authors { font-size: 13px; }
  .hero__ctas { flex-direction: column; gap: 8px; width: 100%; align-items: center; }
  .hero__ctas .btn { width: 100%; max-width: 260px; text-align: center; }

  .stats__inner { padding: 0 16px; gap: 0; }
  .stat { padding: 12px 16px; }
  .stat__value { font-size: 16px; }

  .events-section { padding: 32px 16px; }
  .section-header { flex-direction: column; align-items: flex-start; gap: 8px; }

  .intro-section { padding: 40px 16px 16px; }
  .approach-section { padding: 0 16px 40px; }
  .intro-text { font-size: 14px; }
  .feature-card { flex-direction: column; gap: 12px; padding: 20px; }
  .feature-card__title { font-size: 17px; }
  .feature-card__cta { position: static; opacity: 1; margin-top: 8px; }
}
</style>
