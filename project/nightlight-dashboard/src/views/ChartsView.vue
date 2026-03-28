<template>
  <div class="charts-page">
    <div class="charts-inner">

      <!-- Page header -->
      <div class="page-header reveal">
        <div class="page-header__left">
          <div class="tag tag--cyan" style="margin-bottom:10px">Recovery Analysis</div>
          <h1 class="page-header__title">Resilience Advantage Curves</h1>
          <p class="page-header__desc">
            Daily NTL ratio (R = NTL / BAU) for buffer zones vs. non-buffer areas.
            Hover over any chart to inspect daily values. The shaded region shows the resilience advantage.
          </p>
        </div>
        <div class="page-header__controls">
          <div class="view-toggle">
            <button
              v-for="v in views"
              :key="v.id"
              class="view-toggle__btn"
              :class="{ active: activeView === v.id }"
              @click="activeView = v.id"
            >{{ v.label }}</button>
          </div>
        </div>
      </div>

      <!-- Summary stats table -->
      <div class="summary-panel reveal">
        <div class="summary-panel__header">
          <span class="tag tag--amber">Summary Statistics</span>
          <span style="font-size:11px; color:var(--text-muted)">Mean Resilience Advantage over post-disaster window</span>
        </div>
        <div class="summary-table-wrap">
          <table class="summary-table">
            <thead>
              <tr>
                <th>Event</th>
                <th>Type</th>
                <th>Year</th>
                <th>Mean RA</th>
                <th>Min NTL</th>
                <th>Recovery Day</th>
                <th>Buffer R̄</th>
                <th>Non-Buf R̄</th>
              </tr>
            </thead>
            <tbody>
              <tr
                v-for="ev in EVENTS"
                :key="ev.id"
                class="summary-table__row"
                :class="{ active: activeEventId === ev.id }"
                @click="activeEventId = ev.id; activeView = 'single'"
              >
                <td>
                  <span class="event-dot" :style="{ background: ev.color }" />
                  {{ ev.name }}
                </td>
                <td>
                  <span class="tag" :class="`tag--${ev.type}`" style="font-size:10px">{{ ev.type }}</span>
                </td>
                <td class="mono">{{ ev.year }}</td>
                <td :class="raClass(evStats[ev.id]?.ra)">
                  {{ evStats[ev.id] ? `+${(evStats[ev.id].ra * 100).toFixed(1)}%` : '—' }}
                </td>
                <td class="mono">{{ evStats[ev.id] ? `${(evStats[ev.id].minR * 100).toFixed(1)}%` : '—' }}</td>
                <td class="mono">{{ evStats[ev.id]?.recoveryDay != null ? `+${evStats[ev.id].recoveryDay}d` : 'N/A' }}</td>
                <td class="mono">{{ evStats[ev.id] ? `${(evStats[ev.id].meanBuf * 100).toFixed(1)}%` : '—' }}</td>
                <td class="mono">{{ evStats[ev.id] ? `${(evStats[ev.id].meanNonBuf * 100).toFixed(1)}%` : '—' }}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- Grid view: all 6 events -->
      <div v-if="activeView === 'grid'" class="charts-grid reveal">
        <RecoveryChart
          v-for="ev in EVENTS"
          :key="ev.id"
          :event="ev"
          :height="200"
          class="chart-cell"
          @click="activeEventId = ev.id; activeView = 'single'"
        />
      </div>

      <!-- Single event view -->
      <div v-else class="single-view reveal">
        <!-- Event selector pills -->
        <div class="event-pills">
          <button
            v-for="ev in EVENTS"
            :key="ev.id"
            class="ev-pill"
            :class="{ active: activeEventId === ev.id }"
            :style="{ '--ec': ev.color }"
            @click="activeEventId = ev.id"
          >
            <span class="ev-pill__dot" />
            {{ ev.subtitle.split(',')[0] }}
          </button>
        </div>

        <!-- Large chart -->
        <RecoveryChart
          v-if="activeEvent"
          :event="activeEvent"
          :height="280"
          class="chart-single"
        />

        <!-- Annotation panel -->
        <div class="annotation-panel" v-if="activeEvent">
          <div class="annotation-panel__col">
            <div class="ap-title">Event Context</div>
            <p>{{ activeEvent.description }}</p>
            <div class="ap-meta">
              <div><span class="lbl">Date</span><span class="val mono">{{ activeEvent.date }}</span></div>
              <div><span class="lbl">Affected Users</span><span class="val mono">{{ activeEvent.affectedUsers }}</span></div>
              <div><span class="lbl">Outage Duration</span><span class="val mono">{{ activeEvent.outageDuration }}</span></div>
            </div>
          </div>
          <div class="annotation-panel__col">
            <div class="ap-title">Interpretation</div>
            <p>
              The <strong>resilience advantage (RA)</strong> is the mean difference between
              buffer zone and non-buffer NTL ratios over the post-disaster window.
              A positive RA indicates faster NTL recovery inside critical infrastructure
              buffers, consistent with backup generator activation.
            </p>
            <div class="callout callout--green" style="margin-top:8px; padding:10px 12px; font-size:12px">
              <div>
                Mean RA for this event:
                <strong style="color:var(--green); font-family:var(--font-mono)">
                  {{ evStats[activeEvent.id] ? `+${(evStats[activeEvent.id].ra * 100).toFixed(1)}%` : '...' }}
                </strong>
              </div>
            </div>
          </div>
        </div>
      </div>

    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { EVENTS } from '@/data/events.js'
import { computeResilienceStats } from '@/data/timeseries.js'
import { loadTimeSeries } from '@/data/loader.js'
import RecoveryChart from '@/components/RecoveryChart.vue'

const activeView    = ref('grid')
const activeEventId = ref(EVENTS[0].id)
const activeEvent   = computed(() => EVENTS.find(e => e.id === activeEventId.value))

const views = [
  { id: 'grid',   label: 'All Events' },
  { id: 'single', label: 'Detail View' },
]

// Load stats async from real data (falls back to mock)
const evStats = ref({})

onMounted(async () => {
  const results = await Promise.all(
    EVENTS.map(async ev => {
      const series = await loadTimeSeries(ev)
      return [ev.id, computeResilienceStats(series)]
    })
  )
  evStats.value = Object.fromEntries(results)
})

function raClass(ra) {
  if (ra == null) return ''
  return ra > 0.05 ? 'ra-positive' : ra < 0 ? 'ra-negative' : 'ra-neutral'
}

// Scroll reveal
let revealObs
onMounted(() => {
  revealObs = new IntersectionObserver(
    entries => entries.forEach(e => e.target.classList.toggle('visible', e.isIntersecting)),
    { threshold: 0.1 }
  )
  setTimeout(() => document.querySelectorAll('.charts-page .reveal').forEach(el => revealObs.observe(el)), 100)
})
onUnmounted(() => revealObs?.disconnect())
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
.charts-page {
  min-height: calc(100vh - var(--nav-h));
  background: transparent;
}
.charts-inner {
  max-width: 1240px;
  margin: 0 auto;
  padding: 40px 32px 80px;
  display: flex;
  flex-direction: column;
  gap: 32px;
}

/* Header */
.page-header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 24px;
}
.page-header__title {
  font-size: 28px;
  font-weight: 700;
  margin-bottom: 8px;
}
.page-header__desc {
  font-size: 13px;
  color: var(--text-muted);
  max-width: 560px;
  line-height: 1.65;
}
.view-toggle {
  display: flex;
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}
.view-toggle__btn {
  padding: 7px 18px;
  background: none;
  border: none;
  cursor: pointer;
  font-family: var(--font-head);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
  transition: all var(--t-fast);
}
.view-toggle__btn:hover { color: var(--text-bright); background: var(--bg-3); }
.view-toggle__btn.active { color: var(--cyan); background: var(--cyan-dim); }

/* Summary panel */
.summary-panel {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  overflow: hidden;
}
.summary-panel__header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-3);
}
.summary-table-wrap { overflow-x: auto; }
.summary-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
.summary-table th {
  font-family: var(--font-head);
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--text-dim);
  text-align: left;
  padding: 8px 14px;
  background: var(--bg-3);
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
.summary-table td {
  padding: 9px 14px;
  border-bottom: 1px solid rgba(18,42,69,0.6);
  color: var(--text);
  white-space: nowrap;
}
.summary-table__row {
  cursor: pointer;
  transition: background var(--t-fast);
}
.summary-table__row:hover td { background: var(--bg-3); }
.summary-table__row.active td { background: var(--cyan-dim); }
.summary-table__row:last-child td { border-bottom: none; }
.event-dot {
  display: inline-block;
  width: 7px; height: 7px;
  border-radius: 50%;
  margin-right: 7px;
  vertical-align: middle;
}
.ra-positive { color: var(--green) !important; font-family: var(--font-mono); font-weight: 600; }
.ra-negative { color: var(--red)   !important; font-family: var(--font-mono); }
.ra-neutral  { color: var(--amber) !important; font-family: var(--font-mono); }

/* Grid */
.charts-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(380px, 1fr));
  gap: 16px;
}
.chart-cell {
  cursor: pointer;
  transition: border-color var(--t-fast);
}
.chart-cell:hover { border-color: var(--border-2) !important; }

/* Single view */
.single-view {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
.event-pills {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}
.ev-pill {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 5px 12px;
  border-radius: 20px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  cursor: pointer;
  font-size: 12px;
  color: var(--text-muted);
  font-family: var(--font-body);
  transition: all var(--t-fast);
}
.ev-pill:hover { border-color: var(--ec); color: var(--text-bright); }
.ev-pill.active { border-color: var(--ec); background: rgba(255,255,255,.04); color: var(--text-bright); }
.ev-pill__dot { width: 7px; height: 7px; border-radius: 50%; background: var(--ec); flex-shrink: 0; }

.chart-single { cursor: default; }

/* Annotation panel */
.annotation-panel {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
}
.annotation-panel__col {
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 18px;
}
.ap-title {
  font-family: var(--font-head);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-dim);
  margin-bottom: 10px;
}
.annotation-panel__col p {
  font-size: 13px;
  color: var(--text-muted);
  line-height: 1.65;
  margin-bottom: 12px;
}
.annotation-panel__col strong { color: var(--text-bright); }
.ap-meta { display: flex; flex-direction: column; gap: 6px; }
.ap-meta > div { display: flex; justify-content: space-between; }
.lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); }
.val { font-size: 12px; color: var(--text-bright); }

.callout { display:flex; gap:10px; border-radius:var(--radius); }
.callout--green { background: var(--green-dim); border: 1px solid rgba(0,229,160,.15); color: var(--text); }

@media (max-width: 800px) {
  .charts-inner { padding: 24px 16px 60px; }
  .charts-grid  { grid-template-columns: 1fr; }
  .annotation-panel { grid-template-columns: 1fr; }
  .page-header  { flex-direction: column; }
}
</style>
