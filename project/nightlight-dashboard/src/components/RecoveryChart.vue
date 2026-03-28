<template>
  <div class="recovery-chart" ref="rootEl">
    <!-- Header -->
    <div class="rc__header">
      <div class="rc__title">
        <span class="rc__event-name">{{ event?.name }}</span>
        <span class="rc__event-loc">{{ event?.subtitle }}</span>
      </div>
      <div class="rc__legend">
        <span class="rc__legend-item rc__legend-item--buf">
          <span class="rc__legend-dot rc__legend-dot--buf" />
          Buffer Zone
        </span>
        <span class="rc__legend-item rc__legend-item--non">
          <span class="rc__legend-dot rc__legend-dot--non" />
          Non-Buffer
        </span>
      </div>
    </div>

    <!-- Chart SVG -->
    <div class="rc__chart-wrap" ref="chartWrap">
      <svg
        :width="svgW"
        :height="svgH"
        class="rc__svg"
        @mousemove="onMouseMove"
        @mouseleave="tooltip.visible = false"
      >
        <defs>
          <!-- Green gradient fill for buffer zone -->
          <linearGradient :id="gradBufId" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%"   stop-color="#00e5a0" stop-opacity="0.25" />
            <stop offset="100%" stop-color="#00e5a0" stop-opacity="0.01" />
          </linearGradient>
          <!-- Cyan gradient fill for non-buffer -->
          <linearGradient :id="gradNonId" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%"   stop-color="#00d4ff" stop-opacity="0.12" />
            <stop offset="100%" stop-color="#00d4ff" stop-opacity="0.01" />
          </linearGradient>
          <!-- Clip to chart area -->
          <clipPath :id="clipId">
            <rect :x="pad.l" :y="pad.t" :width="innerW" :height="innerH" />
          </clipPath>
        </defs>

        <!-- Grid lines (horizontal) -->
        <g class="grid-lines">
          <line
            v-for="tick in yTicks"
            :key="tick"
            :x1="pad.l"
            :x2="pad.l + innerW"
            :y1="yScale(tick)"
            :y2="yScale(tick)"
            stroke="rgba(18,42,69,0.8)"
            stroke-width="1"
          />
        </g>

        <!-- Disaster event line -->
        <line
          :x1="xScale(0)"
          :x2="xScale(0)"
          :y1="pad.t"
          :y2="pad.t + innerH"
          stroke="rgba(255,100,100,0.5)"
          stroke-width="1"
          stroke-dasharray="4 3"
        />
        <text
          :x="xScale(0) + 4"
          :y="pad.t + 12"
          fill="rgba(255,120,120,0.8)"
          font-size="9"
          font-family="JetBrains Mono, monospace"
        >EVENT</text>

        <!-- R=1 reference line -->
        <line
          :x1="pad.l"
          :x2="pad.l + innerW"
          :y1="yScale(1)"
          :y2="yScale(1)"
          stroke="rgba(255,255,255,0.08)"
          stroke-width="1"
        />

        <!-- Clip group for paths -->
        <g :clip-path="`url(#${clipId})`">
          <!-- Resilience advantage shaded area -->
          <path
            :d="advantageAreaPath"
            fill="rgba(0,229,160,0.08)"
          />

          <!-- Non-buffer fill area -->
          <path
            :d="areaPath('R_nonBuffer')"
            :fill="`url(#${gradNonId})`"
          />
          <!-- Buffer fill area -->
          <path
            :d="areaPath('R_buffer')"
            :fill="`url(#${gradBufId})`"
          />

          <!-- Non-buffer line -->
          <path
            :d="linePath('R_nonBuffer')"
            fill="none"
            stroke="#00d4ff"
            stroke-width="1.5"
            stroke-opacity="0.7"
          />
          <!-- Buffer line -->
          <path
            :d="linePath('R_buffer')"
            fill="none"
            stroke="#00e5a0"
            stroke-width="2"
          />
        </g>

        <!-- X axis labels -->
        <g>
          <text
            v-for="tick in xTicksDays"
            :key="tick"
            :x="xScale(tick)"
            :y="pad.t + innerH + 16"
            text-anchor="middle"
            font-size="9"
            font-family="JetBrains Mono, monospace"
            fill="rgba(74,106,138,0.9)"
          >{{ tick >= 0 ? '+' : '' }}{{ tick }}</text>
        </g>

        <!-- Y axis labels -->
        <g>
          <text
            v-for="tick in yTicks"
            :key="tick"
            :x="pad.l - 6"
            :y="yScale(tick) + 3"
            text-anchor="end"
            font-size="9"
            font-family="JetBrains Mono, monospace"
            fill="rgba(74,106,138,0.9)"
          >{{ tick.toFixed(1) }}</text>
        </g>

        <!-- Axis labels -->
        <text
          :x="pad.l + innerW / 2"
          :y="svgH - 2"
          text-anchor="middle"
          font-size="9"
          font-family="JetBrains Mono, monospace"
          fill="rgba(74,106,138,0.7)"
        >Days Relative to Disaster</text>

        <!-- Hover scrubber line -->
        <line
          v-if="tooltip.visible"
          :x1="tooltip.x"
          :x2="tooltip.x"
          :y1="pad.t"
          :y2="pad.t + innerH"
          stroke="rgba(200,223,240,0.3)"
          stroke-width="1"
        />
        <circle
          v-if="tooltip.visible && tooltip.data"
          :cx="tooltip.x"
          :cy="yScale(tooltip.data.R_buffer)"
          r="3.5"
          fill="#00e5a0"
        />
        <circle
          v-if="tooltip.visible && tooltip.data"
          :cx="tooltip.x"
          :cy="yScale(tooltip.data.R_nonBuffer)"
          r="3"
          fill="#00d4ff"
        />
      </svg>

      <!-- Floating tooltip -->
      <div
        v-if="tooltip.visible && tooltip.data"
        class="rc__tooltip"
        :style="{
          left: (tooltip.x + (tooltip.x > svgW * 0.6 ? -140 : 12)) + 'px',
          top: (pad.t + 8) + 'px',
        }"
      >
        <div class="rc__tt-day mono">Day {{ tooltip.data.day >= 0 ? '+' : '' }}{{ tooltip.data.day }}</div>
        <div class="rc__tt-row">
          <span class="rc__tt-dot" style="background:#00e5a0" />
          <span class="rc__tt-lbl">Buffer</span>
          <span class="rc__tt-val mono">{{ (tooltip.data.R_buffer * 100).toFixed(1) }}%</span>
        </div>
        <div class="rc__tt-row">
          <span class="rc__tt-dot" style="background:#00d4ff" />
          <span class="rc__tt-lbl">Non-Buffer</span>
          <span class="rc__tt-val mono">{{ (tooltip.data.R_nonBuffer * 100).toFixed(1) }}%</span>
        </div>
        <div class="rc__tt-ra" :class="raClass(tooltip.data)">
          RA = {{ ((tooltip.data.R_buffer - tooltip.data.R_nonBuffer) * 100).toFixed(1) }}%
        </div>
      </div>
    </div>

    <!-- Stats row -->
    <div class="rc__stats" v-if="stats">
      <div class="rc__stat">
        <span class="rc__stat-lbl">Mean RA</span>
        <span class="rc__stat-val" :class="stats.ra > 0 ? 'positive' : 'neutral'">
          {{ stats.ra > 0 ? '+' : '' }}{{ (stats.ra * 100).toFixed(1) }}%
        </span>
      </div>
      <div class="rc__stat">
        <span class="rc__stat-lbl">Min NTL</span>
        <span class="rc__stat-val">{{ (stats.minR * 100).toFixed(1) }}%</span>
      </div>
      <div class="rc__stat">
        <span class="rc__stat-lbl">Post-Event Buf.</span>
        <span class="rc__stat-val">{{ (stats.meanBuf * 100).toFixed(1) }}%</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, watch, onMounted, onUnmounted } from 'vue'
import { generateRecoverySeries, computeResilienceStats } from '@/data/timeseries.js'
import { loadTimeSeries } from '@/data/loader.js'

const props = defineProps({
  event: { type: Object, required: true },
  height: { type: Number, default: 200 },
})

// Unique IDs per instance to avoid SVG id collisions
const uid = `rc-${Math.random().toString(36).slice(2, 8)}`
const clipId = `${uid}-clip`
const gradBufId = `${uid}-grad-buf`
const gradNonId = `${uid}-grad-non`

const chartWrap = ref(null)
const rootEl    = ref(null)
const svgW      = ref(560)
const svgH      = computed(() => props.height)

const pad = { t: 16, b: 28, l: 32, r: 16 }
const innerW = computed(() => svgW.value - pad.l - pad.r)
const innerH = computed(() => svgH.value - pad.t - pad.b)

// Series: loaded async from /data/ts_<id>.json, fallback to mock
const series = ref([])
const stats  = computed(() => computeResilienceStats(series.value))

watch(() => props.event?.id, async (id) => {
  if (!id) return
  series.value = await loadTimeSeries(props.event)
}, { immediate: true })

// Y range: adaptive to data
const yMin = 0
const yMax = computed(() => {
  if (!series.value.length) return 1.35
  const maxVal = Math.max(...series.value.map(d => Math.max(d.R_buffer ?? 0, d.R_nonBuffer ?? 0)))
  return Math.max(1.35, Math.ceil(maxVal * 4) / 4 + 0.1)  // round up to nearest 0.25 + margin
})
const yTicks = computed(() => {
  const ticks = []
  for (let t = 0; t <= yMax.value; t += 0.25) {
    ticks.push(parseFloat(t.toFixed(2)))
  }
  return ticks
})

const xTicksDays = [-30, -20, -10, 0, 10, 20, 30, 40, 50, 60]

// Day range
const dayMin = computed(() => series.value[0]?.day ?? -30)
const dayMax = computed(() => series.value[series.value.length - 1]?.day ?? 60)

function xScale(day) {
  const frac = (day - dayMin.value) / (dayMax.value - dayMin.value)
  return pad.l + frac * innerW.value
}

function yScale(val) {
  const frac = (val - yMin) / (yMax.value - yMin)
  return pad.t + innerH.value * (1 - frac)
}

function linePath(key) {
  if (!series.value.length) return ''
  return series.value.map((d, i) => {
    const x = xScale(d.day)
    const y = yScale(d[key])
    return `${i === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

function areaPath(key) {
  if (!series.value.length) return ''
  const pts = series.value.map(d => ({
    x: xScale(d.day),
    y: yScale(d[key]),
  }))
  const baseline = pad.t + innerH.value
  let d = `M${pts[0].x.toFixed(1)},${baseline}`
  pts.forEach(p => { d += ` L${p.x.toFixed(1)},${p.y.toFixed(1)}` })
  d += ` L${pts[pts.length - 1].x.toFixed(1)},${baseline} Z`
  return d
}

// Shaded area between buf and non-buf (advantage zone)
const advantageAreaPath = computed(() => {
  if (!series.value.length) return ''
  const fwd = series.value.map(d => `L${xScale(d.day).toFixed(1)},${yScale(d.R_buffer).toFixed(1)}`)
  const bwd = [...series.value].reverse().map(d => `L${xScale(d.day).toFixed(1)},${yScale(d.R_nonBuffer).toFixed(1)}`)
  const start = series.value[0]
  return `M${xScale(start.day).toFixed(1)},${yScale(start.R_buffer).toFixed(1)} ${fwd.slice(1).join(' ')} ${bwd.join(' ')} Z`
})

// Tooltip
const tooltip = ref({ visible: false, x: 0, data: null })

function onMouseMove(e) {
  const rect = e.currentTarget.getBoundingClientRect()
  const mx   = e.clientX - rect.left
  if (mx < pad.l || mx > pad.l + innerW.value) { tooltip.value.visible = false; return }

  // Find closest data point
  const day   = dayMin.value + ((mx - pad.l) / innerW.value) * (dayMax.value - dayMin.value)
  const closest = series.value.reduce((best, d) =>
    Math.abs(d.day - day) < Math.abs(best.day - day) ? d : best
  )
  tooltip.value = { visible: true, x: xScale(closest.day), data: closest }
}

function raClass(d) {
  const ra = d.R_buffer - d.R_nonBuffer
  return ra > 0.05 ? 'positive' : ra < -0.03 ? 'negative' : 'neutral'
}

// Responsive width
let ro
onMounted(() => {
  if (chartWrap.value) {
    ro = new ResizeObserver(entries => {
      svgW.value = entries[0].contentRect.width || 560
    })
    ro.observe(chartWrap.value)
  }
})
onUnmounted(() => ro?.disconnect())
</script>

<style scoped>
.recovery-chart {
  display: flex;
  flex-direction: column;
  gap: 12px;
  background: var(--bg-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 16px;
}

/* Header */
.rc__header {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
}
.rc__title {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.rc__event-name {
  font-family: var(--font-head);
  font-size: 13px;
  font-weight: 600;
  color: var(--text-bright);
}
.rc__event-loc {
  font-size: 11px;
  color: var(--text-muted);
}
.rc__legend {
  display: flex;
  align-items: center;
  gap: 14px;
}
.rc__legend-item {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--text-muted);
}
.rc__legend-dot {
  width: 20px;
  height: 2px;
  border-radius: 1px;
}
.rc__legend-dot--buf { background: #00e5a0; }
.rc__legend-dot--non { background: #00d4ff; }

/* Chart */
.rc__chart-wrap {
  position: relative;
  width: 100%;
}
.rc__svg {
  width: 100%;
  display: block;
  cursor: crosshair;
}

/* Tooltip */
.rc__tooltip {
  position: absolute;
  background: rgba(7, 21, 37, 0.97);
  border: 1px solid var(--border-2);
  border-radius: var(--radius);
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  pointer-events: none;
  width: 130px;
  z-index: 10;
}
.rc__tt-day {
  font-size: 10px;
  color: var(--text-muted);
  letter-spacing: 0.08em;
  margin-bottom: 2px;
}
.rc__tt-row {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--text);
}
.rc__tt-dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.rc__tt-lbl { flex: 1; }
.rc__tt-val { font-weight: 600; color: var(--text-bright); }
.rc__tt-ra {
  font-family: var(--font-mono);
  font-size: 11px;
  padding-top: 4px;
  border-top: 1px solid var(--border);
  margin-top: 2px;
  font-weight: 600;
}
.rc__tt-ra.positive { color: var(--green); }
.rc__tt-ra.negative { color: var(--red); }
.rc__tt-ra.neutral  { color: var(--amber); }

/* Stats row */
.rc__stats {
  display: flex;
  gap: 0;
  border-top: 1px solid var(--border);
  padding-top: 10px;
}
.rc__stat {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 0 12px;
  border-right: 1px solid var(--border);
}
.rc__stat:first-child { padding-left: 0; }
.rc__stat:last-child  { border-right: none; }
.rc__stat-lbl {
  font-size: 9px;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  color: var(--text-dim);
}
.rc__stat-val {
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 600;
  color: var(--text-bright);
}
.rc__stat-val.positive { color: var(--green); }
.rc__stat-val.neutral  { color: var(--text-bright); }
</style>
