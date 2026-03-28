<template>
  <div class="map-page">
    <!-- Full-screen map container -->
    <div ref="mapContainer" class="map-container" :class="{ 'map-container--fading': mapFading }" />

    <!-- ── Left sidebar: layer controls ── -->
    <aside class="sidebar" :class="{ collapsed: sidebarCollapsed }">
      <button class="sidebar__toggle btn btn--ghost" @click="sidebarCollapsed = !sidebarCollapsed">
        {{ sidebarCollapsed ? '›' : '‹' }}
      </button>

      <div class="sidebar__inner" v-show="!sidebarCollapsed">
        <div class="sidebar__section">
          <div class="sidebar__section-title">Layers</div>

          <label v-for="layer in layers" :key="layer.id" class="layer-toggle">
            <input
              type="checkbox"
              :checked="layer.visible"
              @change="toggleLayer(layer)"
            />
            <span class="layer-toggle__box" :style="{ '--lc': layer.color }" />
            <span class="layer-toggle__label">{{ layer.label }}</span>
          </label>

          <label class="layer-toggle" style="margin-top:4px">
            <input type="checkbox" :checked="showLabels" @change="toggleLabels" />
            <span class="layer-toggle__box" :style="{ '--lc': '#c8dff0' }" />
            <span class="layer-toggle__label">Facility Labels</span>
          </label>
        </div>

        <hr class="divider" style="margin:16px 0" />

        <!-- Basemap selector -->
        <div class="sidebar__section">
          <div class="sidebar__section-title">Basemap</div>
          <div class="basemap-list">
            <button
              v-for="bm in basemaps"
              :key="bm.id"
              class="basemap-btn"
              :class="{ active: activeBasemap === bm.id }"
              @click="switchBasemap(bm.id)"
            >{{ bm.label }}</button>
          </div>
        </div>

        <hr class="divider" style="margin:16px 0" />

        <!-- Probability legend -->
        <div class="sidebar__section">
          <div class="sidebar__section-title">Probability Legend</div>
          <div class="legend">
            <div class="legend__bar" :class="{ 'legend__bar--light': activeBasemap === 'positron' || activeBasemap === 'voyager' }" />
            <div class="legend__labels">
              <span>0%</span>
              <span>50%</span>
              <span>100%</span>
            </div>
            <div class="legend__caption">Predicted resilience probability</div>
          </div>
        </div>

        <hr class="divider" style="margin:16px 0" />

        <!-- Facility type legend -->
        <div class="sidebar__section">
          <div class="sidebar__section-title">Facility Types</div>
          <div class="fac-legend">
            <div class="fac-legend__item">
              <span class="fac-legend__icon" style="color:#00e5a0">&#x2795;</span>
              <span>Hospital</span>
            </div>
            <div class="fac-legend__item">
              <span class="fac-legend__icon" style="color:#00d4ff">&#x2708;</span>
              <span>Airport</span>
            </div>
            <div class="fac-legend__item">
              <span class="fac-legend__icon" style="color:#ffaa00">&#x1F525;</span>
              <span>Fire Station</span>
            </div>
            <div class="fac-legend__item">
              <span class="fac-legend__icon" style="color:#c084fc">&#x26A1;</span>
              <span>Power Plant</span>
            </div>
            <div class="fac-legend__item">
              <span class="fac-legend__icon" style="color:#60a5fa">&#x1F6E1;</span>
              <span>Police</span>
            </div>
            <div class="fac-legend__item">
              <span class="fac-legend__icon" style="color:#f97316">&#x1F3DB;</span>
              <span>Government</span>
            </div>
            <div class="fac-legend__item">
              <span class="fac-legend__icon" style="color:#f59e0b">&#x1F50C;</span>
              <span>Substation</span>
            </div>
            <div class="fac-legend__item">
              <span class="fac-legend__icon" style="color:#38bdf8">&#x1F4A7;</span>
              <span>Water Works</span>
            </div>
          </div>
        </div>

        <hr class="divider" style="margin:16px 0" />

        <!-- Selected event info -->
        <div class="sidebar__section" v-if="activeEvent">
          <div class="sidebar__section-title">Active Event</div>
          <div class="event-info">
            <div class="event-info__name">{{ activeEvent.name }}</div>
            <div class="event-info__sub">{{ activeEvent.subtitle }}</div>
            <div class="event-info__row">
              <span class="tag" :class="`tag--${activeEvent.type}`">{{ activeEvent.type }}</span>
              <span class="mono" style="font-size:11px; color:var(--text-muted)">{{ activeEvent.year }}</span>
            </div>
            <div class="event-info__stats">
              <div><span class="lbl">Affected</span><span class="val mono">{{ activeEvent.affectedUsers }}</span></div>
              <div><span class="lbl">Outage</span><span class="val mono">{{ activeEvent.outageDuration }}</span></div>
              <div><span class="lbl">Facilities</span><span class="val mono">{{ activeEvent.facilities.length }}</span></div>
            </div>
          </div>
        </div>
      </div>
    </aside>

    <!-- ── Right sidebar: event selector ── -->
    <aside class="event-panel" :class="{ collapsed: eventPanelCollapsed }">
      <button class="event-panel__toggle btn btn--ghost" @click="eventPanelCollapsed = !eventPanelCollapsed">
        {{ eventPanelCollapsed ? '‹' : '›' }}
      </button>
      <div class="event-panel__inner" v-show="!eventPanelCollapsed">
        <div class="event-panel__label">EVENTS</div>
        <button
          v-for="ev in EVENTS"
          :key="ev.id"
          class="event-pill"
          :class="{ active: activeEventId === ev.id }"
          :style="{ '--ec': ev.color }"
          @click="flyToEvent(ev)"
        >
          <span class="event-pill__dot" />
          <span class="event-pill__text">{{ ev.subtitle.split(',')[0] }}</span>
          <span class="event-pill__type mono">{{ ev.type === 'hurricane' ? 'Hurricane' : 'Earthquake' }} {{ ev.year }}</span>
        </button>
      </div>
    </aside>

    <!-- ── Feature popup ── -->
    <Teleport to="body">
      <div
        v-if="popup.visible"
        class="map-popup"
        :style="{ left: popup.x + 'px', top: popup.y + 'px' }"
      >
        <button class="map-popup__close" @click="popup.visible = false">×</button>
        <div class="map-popup__type tag" :class="`tag--${popup.facilityType}`">
          {{ facilityTypeLabel(popup.facilityType) }}
        </div>
        <div class="map-popup__name">{{ popup.name }}</div>
        <div class="map-popup__prob">
          <span class="lbl">Predicted Probability</span>
          <div class="prob-bar-wrap">
            <div class="prob-bar" :style="{ width: (popup.probability * 100) + '%', background: probColor(popup.probability) }" />
          </div>
          <span class="prob-val mono" :style="{ color: probColor(popup.probability) }">
            {{ (popup.probability * 100).toFixed(1) }}%
          </span>
        </div>
      </div>
    </Teleport>

    <!-- ── Pixel hover tooltip ── -->
    <div
      v-if="pixelTip.visible"
      class="pixel-tooltip"
      :style="{ left: pixelTip.x + 'px', top: pixelTip.y + 'px' }"
    >
      <div class="pixel-tooltip__row">
        <span class="pixel-tooltip__lbl">Probability</span>
        <span class="pixel-tooltip__val mono" :style="{ color: probColor(pixelTip.prob) }">
          {{ (pixelTip.prob * 100).toFixed(1) }}%
        </span>
      </div>
      <div class="pixel-tooltip__bar">
        <div class="pixel-tooltip__fill" :style="{ width: (pixelTip.prob * 100) + '%', background: probColor(pixelTip.prob) }" />
      </div>
      <div class="pixel-tooltip__meta mono">
        {{ pixelTip.inBuffer ? 'Inside Buffer' : 'Outside Buffer' }}
      </div>
    </div>

    <!-- ── Bottom status bar ── -->
    <div class="status-bar">
      <span class="mono" style="font-size:10px; color:var(--text-muted)">
        Base tiles: CARTO Dark Matter · Data: NASA Black Marble VNP46A2
      </span>
      <span class="mono" style="font-size:10px; color:var(--text-dim); margin-left:auto">
        Probability values are model predictions
      </span>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { useRoute } from 'vue-router'
import maplibregl from 'maplibre-gl'
import RecoveryChart from '@/components/RecoveryChart.vue'
import {
  EVENTS as RAW_EVENTS,
  FACILITY_TYPES,
  BUFFER_RADII,
} from '@/data/events.js'
import { loadProbabilityGeoJSON, loadFacilityGeoJSON } from '@/data/loader.js'

// Custom display order: San Juan events first, Miami last
const EVENT_ORDER = ['maria', 'eq-pr', 'ida', 'laura', 'michael', 'ian-charlotte', 'ian-fortmyers', 'eq-hatay', 'irma']
const EVENTS = EVENT_ORDER.map(id => RAW_EVENTS.find(e => e.id === id)).filter(Boolean)

// Build buffer polygons from real loaded facility GeoJSON
// (replaces generateBufferGeoJSON which used hardcoded ev.facilities)
function buildBufferFromGeoJSON(facilityGeoJSON) {
  const features = []
  for (const f of facilityGeoJSON.features) {
    const type     = f.properties.type
    const radiusM  = BUFFER_RADII[type] ?? 750
    const radiusDeg = radiusM / 111320
    const [lon, lat] = f.geometry.coordinates
    const steps = 64
    const coords = []
    for (let i = 0; i <= steps; i++) {
      const angle = (i / steps) * 2 * Math.PI
      coords.push([
        lon + radiusDeg * Math.cos(angle),
        lat + radiusDeg * Math.sin(angle) * 0.85,
      ])
    }
    features.push({
      type: 'Feature',
      geometry: { type: 'Polygon', coordinates: [coords] },
      properties: { name: f.properties.name, type, probability: f.properties.probability, radiusM },
    })
  }
  return { type: 'FeatureCollection', features }
}

const route        = useRoute()
const mapContainer = ref(null)
let map            = null

const activeEventId    = ref(null)
const activeEvent      = ref(null)
const sidebarCollapsed = ref(false)
const eventPanelCollapsed = ref(false)
const popup = ref({ visible: false, x: 0, y: 0, name: '', facilityType: '', probability: 0 })
const chartPanelOpen = ref(false)
const showLabels = ref(true)
const pixelTip = ref({ visible: false, x: 0, y: 0, prob: 0, inBuffer: false })
const activeBasemap = ref('dark')

// ── Layer definitions ──
const layers = ref([
  { id: 'heatmap',   label: 'Probability Heatmap', visible: true,  color: '#ff6b35' },
  { id: 'buffers',   label: 'Buffer Zones',         visible: false, color: '#00d4ff' },
  { id: 'facilities',label: 'Critical Facilities',  visible: true,  color: '#00e5a0' },
])

// ── Basemap options ──
const basemaps = [
  { id: 'dark',       label: 'Dark Matter',   url: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json' },
  { id: 'positron',   label: 'Positron',      url: 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json' },
  { id: 'voyager',    label: 'Voyager',       url: 'https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json' },
  { id: 'dark-nolbl', label: 'Dark (No Labels)', url: 'https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json' },
]

// ── Utility: probability → color ──
function probColor(p) {
  if (p >= 0.8) return '#00e5a0'
  if (p >= 0.6) return '#00d4ff'
  if (p >= 0.4) return '#ffaa00'
  return '#ff4444'
}

function facilityTypeLabel(t) {
  return FACILITY_TYPES[t]?.label ?? t
}

// ── Facility icon rendering ──
const ICON_SIZE = 32
const ICON_DEFS = {
  hospital: { color: '#00e5a0', colorLight: '#008060', draw(ctx, s) {
    // Plus/cross
    const w = s * 0.22, h = s * 0.55, cx = s / 2, cy = s / 2
    ctx.fillRect(cx - w / 2, cy - h / 2, w, h)
    ctx.fillRect(cx - h / 2, cy - w / 2, h, w)
  }},
  airport: { color: '#00d4ff', colorLight: '#0070b0', draw(ctx, s) {
    // Airplane silhouette
    const cx = s / 2, cy = s / 2
    ctx.beginPath()
    ctx.moveTo(cx, cy - s * 0.35)
    ctx.lineTo(cx + s * 0.08, cy - s * 0.15)
    ctx.lineTo(cx + s * 0.35, cy + s * 0.02)
    ctx.lineTo(cx + s * 0.35, cy + s * 0.1)
    ctx.lineTo(cx + s * 0.08, cy + s * 0.05)
    ctx.lineTo(cx + s * 0.08, cy + s * 0.18)
    ctx.lineTo(cx + s * 0.18, cy + s * 0.28)
    ctx.lineTo(cx + s * 0.18, cy + s * 0.33)
    ctx.lineTo(cx, cy + s * 0.25)
    // Mirror left side
    ctx.lineTo(cx - s * 0.18, cy + s * 0.33)
    ctx.lineTo(cx - s * 0.18, cy + s * 0.28)
    ctx.lineTo(cx - s * 0.08, cy + s * 0.18)
    ctx.lineTo(cx - s * 0.08, cy + s * 0.05)
    ctx.lineTo(cx - s * 0.35, cy + s * 0.1)
    ctx.lineTo(cx - s * 0.35, cy + s * 0.02)
    ctx.lineTo(cx - s * 0.08, cy - s * 0.15)
    ctx.closePath()
    ctx.fill()
  }},
  fire: { color: '#ffaa00', colorLight: '#cc8800', draw(ctx, s) {
    // Flame
    const cx = s / 2, cy = s / 2
    ctx.beginPath()
    ctx.moveTo(cx, cy - s * 0.35)
    ctx.quadraticCurveTo(cx + s * 0.25, cy - s * 0.1, cx + s * 0.2, cy + s * 0.1)
    ctx.quadraticCurveTo(cx + s * 0.18, cy + s * 0.3, cx, cy + s * 0.35)
    ctx.quadraticCurveTo(cx - s * 0.18, cy + s * 0.3, cx - s * 0.2, cy + s * 0.1)
    ctx.quadraticCurveTo(cx - s * 0.25, cy - s * 0.1, cx, cy - s * 0.35)
    ctx.closePath()
    ctx.fill()
  }},
  power: { color: '#c084fc', colorLight: '#7c3aed', draw(ctx, s) {
    // Lightning bolt
    const cx = s / 2, cy = s / 2
    ctx.beginPath()
    ctx.moveTo(cx + s * 0.05, cy - s * 0.35)
    ctx.lineTo(cx - s * 0.15, cy + s * 0.02)
    ctx.lineTo(cx + s * 0.02, cy + s * 0.02)
    ctx.lineTo(cx - s * 0.05, cy + s * 0.35)
    ctx.lineTo(cx + s * 0.15, cy - s * 0.02)
    ctx.lineTo(cx - s * 0.02, cy - s * 0.02)
    ctx.closePath()
    ctx.fill()
  }},
  police: { color: '#60a5fa', colorLight: '#2563eb', draw(ctx, s) {
    // Shield
    const cx = s / 2, cy = s / 2
    ctx.beginPath()
    ctx.moveTo(cx, cy - s * 0.32)
    ctx.lineTo(cx + s * 0.25, cy - s * 0.18)
    ctx.lineTo(cx + s * 0.25, cy + s * 0.05)
    ctx.quadraticCurveTo(cx + s * 0.2, cy + s * 0.3, cx, cy + s * 0.35)
    ctx.quadraticCurveTo(cx - s * 0.2, cy + s * 0.3, cx - s * 0.25, cy + s * 0.05)
    ctx.lineTo(cx - s * 0.25, cy - s * 0.18)
    ctx.closePath()
    ctx.fill()
  }},
  government: { color: '#f97316', colorLight: '#c2410c', draw(ctx, s) {
    // Building with columns
    const cx = s / 2, cy = s / 2
    // Roof triangle
    ctx.beginPath()
    ctx.moveTo(cx, cy - s * 0.32)
    ctx.lineTo(cx + s * 0.3, cy - s * 0.1)
    ctx.lineTo(cx - s * 0.3, cy - s * 0.1)
    ctx.closePath()
    ctx.fill()
    // Base
    ctx.fillRect(cx - s * 0.28, cy + s * 0.22, s * 0.56, s * 0.06)
    // Columns
    for (const dx of [-0.18, -0.06, 0.06, 0.18]) {
      ctx.fillRect(cx + s * dx - s * 0.03, cy - s * 0.08, s * 0.06, s * 0.3)
    }
  }},
  substation: { color: '#f59e0b', colorLight: '#b45309', draw(ctx, s) {
    // Transformer / electrical box
    const cx = s / 2, cy = s / 2
    ctx.fillRect(cx - s * 0.2, cy - s * 0.25, s * 0.4, s * 0.45)
    // Connectors on top
    ctx.fillRect(cx - s * 0.08, cy - s * 0.35, s * 0.04, s * 0.1)
    ctx.fillRect(cx + s * 0.04, cy - s * 0.35, s * 0.04, s * 0.1)
    // Base
    ctx.fillRect(cx - s * 0.25, cy + s * 0.2, s * 0.5, s * 0.06)
  }},
  water_works: { color: '#38bdf8', colorLight: '#0284c7', draw(ctx, s) {
    // Water drop
    const cx = s / 2, cy = s / 2
    ctx.beginPath()
    ctx.moveTo(cx, cy - s * 0.32)
    ctx.quadraticCurveTo(cx + s * 0.3, cy + s * 0.05, cx + s * 0.2, cy + s * 0.2)
    ctx.quadraticCurveTo(cx, cy + s * 0.38, cx, cy + s * 0.35)
    ctx.quadraticCurveTo(cx, cy + s * 0.38, cx - s * 0.2, cy + s * 0.2)
    ctx.quadraticCurveTo(cx - s * 0.3, cy + s * 0.05, cx, cy - s * 0.32)
    ctx.closePath()
    ctx.fill()
  }},
}

// Default icon for unknown types
const DEFAULT_ICON = { color: '#ffffff', colorLight: '#666666', draw(ctx, s) {
  ctx.beginPath()
  ctx.arc(s / 2, s / 2, s * 0.25, 0, Math.PI * 2)
  ctx.fill()
}}

function createFacilityIcon(type, light) {
  const def = ICON_DEFS[type] || DEFAULT_ICON
  const size = ICON_SIZE
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')

  // Background circle
  ctx.beginPath()
  ctx.arc(size / 2, size / 2, size / 2 - 1, 0, Math.PI * 2)
  ctx.fillStyle = light ? '#ffffff' : '#0a1628'
  ctx.fill()
  ctx.strokeStyle = light ? (def.colorLight || def.color) : def.color
  ctx.lineWidth = 2
  ctx.stroke()

  // Icon shape
  ctx.fillStyle = light ? (def.colorLight || def.color) : def.color
  def.draw(ctx, size)

  return ctx.getImageData(0, 0, size, size)
}

function addFacilityIcons(mapInstance) {
  const light = isLightBasemap()
  const types = [...Object.keys(ICON_DEFS), '_default']
  for (const type of types) {
    const imgName = `fac-${type}`
    if (!mapInstance.hasImage(imgName)) {
      const imgData = createFacilityIcon(type === '_default' ? null : type, light)
      mapInstance.addImage(imgName, imgData, { pixelRatio: 2 })
    }
  }
}

// ── Initialize map ──
onMounted(() => {
  map = new maplibregl.Map({
    container: mapContainer.value,
    style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    center: [-88, 27],
    zoom: 4.5,
    attributionControl: false,
  })

  map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
  map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-left')

  map.on('load', async () => {
    addFacilityIcons(map)
    // Add all event layers (async, loads real data if available)
    await Promise.all(EVENTS.map(ev => addEventLayers(ev)))

    // If route has an event query param, fly there
    const qEvent = route.query.event
    if (qEvent) {
      const ev = EVENTS.find(e => e.id === qEvent)
      if (ev) flyToEvent(ev)
    } else {
      flyToEvent(EVENTS[0])
    }
  })

  // Click handler on facilities
  map.on('click', (e) => {
    // Check all facility layers (only query existing ones)
    const facilityLayerIds = EVENTS.map(ev => `facilities-${ev.id}`).filter(id => map.getLayer(id))
    if (!facilityLayerIds.length) return
    const features = map.queryRenderedFeatures(e.point, { layers: facilityLayerIds })
    if (!features.length) {
      popup.value.visible = false
      return
    }
    const f = features[0]
    const screenPoint = map.project(f.geometry.coordinates)
    popup.value = {
      visible: true,
      x: screenPoint.x + 16,
      y: screenPoint.y - 60,
      name: f.properties.name,
      facilityType: f.properties.type,
      probability: f.properties.probability,
    }
  })

  // Change cursor on hover + pixel probability tooltip
  map.on('mousemove', (e) => {
    const facilityLayerIds = EVENTS.map(ev => `facilities-${ev.id}`).filter(id => map.getLayer(id))
    const facFeatures = facilityLayerIds.length ? map.queryRenderedFeatures(e.point, { layers: facilityLayerIds }) : []
    map.getCanvas().style.cursor = facFeatures.length ? 'pointer' : ''

    // Query pixel probability from hit layers
    if (!facFeatures.length) {
      const hitLayerIds = EVENTS.map(ev => `prob-hit-${ev.id}`).filter(id => map.getLayer(id))
      if (hitLayerIds.length) {
        const pxFeatures = map.queryRenderedFeatures(e.point, { layers: hitLayerIds })
        if (pxFeatures.length) {
          const f = pxFeatures[0]
          pixelTip.value = {
            visible: true,
            x: e.point.x + 16,
            y: e.point.y - 40,
            prob: f.properties.probability,
            inBuffer: !!f.properties.inBuffer,
          }
          map.getCanvas().style.cursor = 'crosshair'
          return
        }
      }
    }
    pixelTip.value.visible = false
  })

  map.on('mouseleave', () => {
    pixelTip.value.visible = false
  })
})

onUnmounted(() => {
  map?.remove()
})

// ── Data cache (avoid re-fetching on basemap switch) ──
const dataCache = {}

// ── Heatmap color ramps ──
function isLightBasemap() {
  return activeBasemap.value === 'positron' || activeBasemap.value === 'voyager'
}

const HEATMAP_COLORS_DARK = [
  'interpolate', ['linear'], ['heatmap-density'],
  0,    'rgba(0,0,0,0)',
  0.1,  'rgba(65,0,130,0.4)',
  0.25, 'rgba(180,0,60,0.55)',
  0.4,  'rgba(255,80,0,0.65)',
  0.55, 'rgba(255,170,0,0.75)',
  0.7,  'rgba(255,255,50,0.8)',
  0.85, 'rgba(180,255,100,0.85)',
  1,    'rgba(255,255,255,0.95)',
]

const HEATMAP_COLORS_LIGHT = [
  'interpolate', ['linear'], ['heatmap-density'],
  0,    'rgba(0,0,0,0)',
  0.1,  'rgba(0,0,120,0.25)',
  0.25, 'rgba(0,60,180,0.4)',
  0.4,  'rgba(0,130,200,0.55)',
  0.55, 'rgba(200,0,80,0.6)',
  0.7,  'rgba(220,50,0,0.7)',
  0.85, 'rgba(200,0,0,0.8)',
  1,    'rgba(140,0,0,0.9)',
]

// ── Per-event probability stats for adaptive weight ──
function getProbStats(probGeoJSON) {
  const probs = probGeoJSON.features.map(f => f.properties.probability).sort((a, b) => a - b)
  const n = probs.length
  const min = probs[0]
  const max = probs[n - 1]
  const p10 = probs[Math.floor(n * 0.1)]
  const p50 = probs[Math.floor(n * 0.5)]
  const p90 = probs[Math.floor(n * 0.9)]
  return { min, max, p10, p50, p90 }
}

// ── Add layers for one event (async: tries real data first) ──
async function addEventLayers(ev) {
  // Use cached data if available
  if (!dataCache[ev.id]) {
    const [probGeoJSON, facilityGeoJSON] = await Promise.all([
      loadProbabilityGeoJSON(ev),
      loadFacilityGeoJSON(ev),
    ])
    const bufferGeoJSON = buildBufferFromGeoJSON(facilityGeoJSON)
    const probStats = getProbStats(probGeoJSON)
    dataCache[ev.id] = { probGeoJSON, facilityGeoJSON, bufferGeoJSON, probStats }
  }
  const { probGeoJSON, facilityGeoJSON, bufferGeoJSON, probStats } = dataCache[ev.id]

  // Skip if source already exists
  if (map.getSource(`prob-${ev.id}`)) return

  const light = isLightBasemap()

  // ── Heatmap source + layer ──
  map.addSource(`prob-${ev.id}`, { type: 'geojson', data: probGeoJSON })
  map.addLayer({
    id: `heatmap-${ev.id}`,
    type: 'heatmap',
    source: `prob-${ev.id}`,
    paint: {
      // Quantile-stretch: spread the color range across actual data distribution
      'heatmap-weight': ['interpolate', ['linear'], ['get', 'probability'],
        probStats.min, 0,
        probStats.p10, 0.15,
        probStats.p50, 0.5,
        probStats.p90, 0.85,
        probStats.max, 1,
      ],
      'heatmap-intensity': ['interpolate', ['linear'], ['zoom'],
        4,  0.6,
        10, 1.5,
        12, 2.5,
        14, 2.0,
        16, 1.2,
      ],
      'heatmap-color': light ? HEATMAP_COLORS_LIGHT : HEATMAP_COLORS_DARK,
      'heatmap-radius': ['interpolate', ['linear'], ['zoom'],
        3,  3,
        8,  5,
        10, 8,
        11, 12,
        12, 18,
        13, 30,
        14, 50,
        15, 80,
        17, 160,
      ],
      'heatmap-opacity': ['interpolate', ['linear'], ['zoom'],
        10, 0.85,
        15, 0.6,
      ],
    },
    layout: { visibility: 'visible' },
  })

  // ── Invisible pixel hit layer for hover interaction ──
  map.addLayer({
    id: `prob-hit-${ev.id}`,
    type: 'circle',
    source: `prob-${ev.id}`,
    paint: {
      'circle-radius': ['interpolate', ['linear'], ['zoom'], 8, 3, 12, 8, 14, 14],
      'circle-opacity': 0,
      'circle-stroke-width': 0,
    },
    layout: { visibility: 'visible' },
  })

  // ── Buffer zones source + layer ──
  const bufColor = light ? '#0066aa' : '#00d4ff'
  map.addSource(`buffers-${ev.id}`, { type: 'geojson', data: bufferGeoJSON })
  map.addLayer({
    id: `buffers-fill-${ev.id}`,
    type: 'fill',
    source: `buffers-${ev.id}`,
    paint: {
      'fill-color': bufColor,
      'fill-opacity': light ? 0.06 : 0.08,
    },
    layout: { visibility: 'none' },
  })
  map.addLayer({
    id: `buffers-line-${ev.id}`,
    type: 'line',
    source: `buffers-${ev.id}`,
    paint: {
      'line-color': bufColor,
      'line-width': 1,
      'line-opacity': light ? 0.6 : 0.5,
      'line-dasharray': [4, 3],
    },
    layout: { visibility: 'none' },
  })

  // ── Facility points source + layer (icon symbols) ──
  map.addSource(`facilities-${ev.id}`, { type: 'geojson', data: facilityGeoJSON })
  map.addLayer({
    id: `facilities-${ev.id}`,
    type: 'symbol',
    source: `facilities-${ev.id}`,
    layout: {
      'icon-image': [
        'match', ['get', 'type'],
        'hospital',    'fac-hospital',
        'airport',     'fac-airport',
        'fire',        'fac-fire',
        'power',       'fac-power',
        'police',      'fac-police',
        'government',  'fac-government',
        'substation',  'fac-substation',
        'water_works', 'fac-water_works',
        'fac-_default',
      ],
      'icon-size': ['interpolate', ['linear'], ['zoom'], 8, 0.4, 12, 0.7, 14, 1],
      'icon-allow-overlap': true,
      visibility: 'visible',
    },
  })

  // Facility labels
  map.addLayer({
    id: `facilities-label-${ev.id}`,
    type: 'symbol',
    source: `facilities-${ev.id}`,
    minzoom: 12,
    layout: {
      'text-field': ['get', 'name'],
      'text-size': 10,
      'text-offset': [0, 1.3],
      'text-anchor': 'top',
      'text-font': ['Noto Sans Regular'],
      visibility: 'visible',
    },
    paint: {
      'text-color': light ? '#1a1a2e' : '#c8dff0',
      'text-halo-color': light ? 'rgba(255,255,255,0.85)' : 'rgba(3,13,26,0.8)',
      'text-halo-width': 1.5,
    },
  })
}

// ── Toggle a layer group on/off ──
function toggleLayer(layer) {
  layer.visible = !layer.visible
  if (!map) return

  EVENTS.forEach(ev => {
    const vis = layer.visible ? 'visible' : 'none'
    if (layer.id === 'heatmap') {
      safeSetVisibility(`heatmap-${ev.id}`, vis)
      safeSetVisibility(`prob-hit-${ev.id}`, vis)
    } else if (layer.id === 'buffers') {
      safeSetVisibility(`buffers-fill-${ev.id}`, vis)
      safeSetVisibility(`buffers-line-${ev.id}`, vis)
    } else if (layer.id === 'facilities') {
      safeSetVisibility(`facilities-${ev.id}`, vis)
      safeSetVisibility(`facilities-label-${ev.id}`, vis)
    }
  })
}

function safeSetVisibility(layerId, vis) {
  if (map.getLayer(layerId)) {
    map.setLayoutProperty(layerId, 'visibility', vis)
  }
}

// ── Toggle facility labels ──
function toggleLabels() {
  showLabels.value = !showLabels.value
  if (!map) return
  const vis = showLabels.value ? 'visible' : 'none'
  EVENTS.forEach(ev => safeSetVisibility(`facilities-label-${ev.id}`, vis))
}

// ── Switch basemap ──
const mapFading = ref(false)

function switchBasemap(id) {
  if (!map || activeBasemap.value === id || mapFading.value) return
  activeBasemap.value = id
  const bm = basemaps.find(b => b.id === id)
  if (!bm) return

  const center = map.getCenter()
  const zoom = map.getZoom()
  const container = mapContainer.value

  // Fade out
  mapFading.value = true

  setTimeout(() => {
    // Remove old map and recreate
    map.remove()

    map = new maplibregl.Map({
      container,
      style: bm.url,
      center: [center.lng, center.lat],
      zoom,
      attributionControl: false,
    })
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    map.addControl(new maplibregl.AttributionControl({ compact: true }), 'bottom-left')

    map.on('load', async () => {
      addFacilityIcons(map)
      await Promise.all(EVENTS.map(ev => addEventLayers(ev)))

      // Restore layer visibility
      EVENTS.forEach(ev => {
        for (const layer of layers.value) {
          const vis = layer.visible ? 'visible' : 'none'
          if (layer.id === 'heatmap') {
            safeSetVisibility(`heatmap-${ev.id}`, vis)
            safeSetVisibility(`prob-hit-${ev.id}`, vis)
          } else if (layer.id === 'buffers') {
            safeSetVisibility(`buffers-fill-${ev.id}`, vis)
            safeSetVisibility(`buffers-line-${ev.id}`, vis)
          }
          else if (layer.id === 'facilities') safeSetVisibility(`facilities-${ev.id}`, vis)
        }
        safeSetVisibility(`facilities-label-${ev.id}`, showLabels.value ? 'visible' : 'none')
      })

      // Fade in
      mapFading.value = false
    })

    // Re-add click/hover handlers
    map.on('click', (e) => {
      const ids = EVENTS.map(ev => `facilities-${ev.id}`).filter(lid => map.getLayer(lid))
      if (!ids.length) return
      const features = map.queryRenderedFeatures(e.point, { layers: ids })
      if (!features.length) { popup.value.visible = false; return }
      const f = features[0]
      const sp = map.project(f.geometry.coordinates)
      popup.value = {
        visible: true, x: sp.x + 16, y: sp.y - 60,
        name: f.properties.name, facilityType: f.properties.type, probability: f.properties.probability,
      }
    })
    map.on('mousemove', (e) => {
      const ids = EVENTS.map(ev => `facilities-${ev.id}`).filter(lid => map.getLayer(lid))
      const facFeatures = ids.length ? map.queryRenderedFeatures(e.point, { layers: ids }) : []
      map.getCanvas().style.cursor = facFeatures.length ? 'pointer' : ''

      if (!facFeatures.length) {
        const hitIds = EVENTS.map(ev => `prob-hit-${ev.id}`).filter(lid => map.getLayer(lid))
        if (hitIds.length) {
          const px = map.queryRenderedFeatures(e.point, { layers: hitIds })
          if (px.length) {
            pixelTip.value = {
              visible: true, x: e.point.x + 16, y: e.point.y - 40,
              prob: px[0].properties.probability, inBuffer: !!px[0].properties.inBuffer,
            }
            map.getCanvas().style.cursor = 'crosshair'
            return
          }
        }
      }
      pixelTip.value.visible = false
    })
  }, 350) // wait for fade-out to finish
}

// ── Fly to event ──
function flyToEvent(ev) {
  activeEventId.value = ev.id
  activeEvent.value   = ev
  popup.value.visible = false

  map?.flyTo({
    center: ev.center,
    zoom:   ev.zoom,
    duration: 1800,
    essential: true,
  })
}
</script>

<style scoped>
.map-page {
  position: relative;
  width: 100%;
  height: calc(100vh - var(--nav-h));
  overflow: hidden;
}

/* Map takes full space */
.map-container {
  width: 100%;
  height: 100%;
  opacity: 1;
  transition: opacity 0.35s ease;
}
.map-container--fading {
  opacity: 0;
}

/* ── Sidebar ── */
.sidebar {
  position: absolute;
  top: 16px;
  left: 16px;
  z-index: 10;
  width: 230px;
  background: rgba(3, 13, 26, 0.92);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  transition: width var(--t-med), height var(--t-med);
}
.sidebar.collapsed {
  width: 42px;
  height: 42px;
}
.sidebar__toggle {
  position: absolute;
  top: 8px;
  left: 8px;
  z-index: 2;
  width: 26px;
  height: 26px;
  padding: 0;
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius);
  cursor: pointer;
}
.sidebar__inner {
  padding: 40px 14px 16px;
  overflow-y: auto;
  overflow-x: hidden;
  max-height: calc(100vh - var(--nav-h) - 100px);
}
.sidebar__section-title {
  font-family: var(--font-head);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin-bottom: 10px;
}

/* Layer toggle */
.layer-toggle {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  padding: 5px 0;
  font-size: 12px;
  color: var(--text);
  user-select: none;
}
.layer-toggle input { display: none; }
.layer-toggle__box {
  width: 14px; height: 14px;
  border-radius: 2px;
  border: 1.5px solid var(--lc, var(--border-2));
  background: transparent;
  position: relative;
  flex-shrink: 0;
  transition: background var(--t-fast);
}
.layer-toggle input:checked ~ .layer-toggle__box {
  background: var(--lc, var(--cyan));
  border-color: var(--lc, var(--cyan));
}
.layer-toggle input:checked ~ .layer-toggle__box::after {
  content: '✓';
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 9px;
  color: var(--bg);
}

/* Probability legend */
.legend { display: flex; flex-direction: column; gap: 4px; }
.legend__bar {
  height: 10px;
  border-radius: 3px;
  background: linear-gradient(90deg, #410082, #b4003c, #ff5000, #ffaa00, #ffff32, #b4ff64, #ffffff);
  opacity: 0.85;
}
.legend__bar--light {
  background: linear-gradient(90deg, #000078, #003cb4, #0082c8, #c80050, #dc3200, #c80000, #8c0000);
}
.legend__labels {
  display: flex;
  justify-content: space-between;
  font-family: var(--font-mono);
  font-size: 10px;
  color: var(--text-muted);
}
.legend__caption {
  font-size: 10px;
  color: var(--text-dim);
  margin-top: 2px;
}

/* Facility legend */
.fac-legend {
  display: flex;
  flex-direction: column;
  gap: 5px;
}
.fac-legend__item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  color: var(--text);
}
.fac-legend__icon {
  width: 18px;
  text-align: center;
  font-size: 13px;
  flex-shrink: 0;
}

/* Basemap selector */
.basemap-list {
  display: flex;
  flex-direction: column;
  gap: 3px;
}
.basemap-btn {
  display: block;
  width: 100%;
  padding: 5px 10px;
  background: none;
  border: 1px solid transparent;
  border-radius: var(--radius);
  cursor: pointer;
  font-family: var(--font-body);
  font-size: 11px;
  color: var(--text-muted);
  text-align: left;
  transition: all var(--t-fast);
}
.basemap-btn:hover {
  color: var(--text-bright);
  background: var(--bg-3);
  border-color: var(--border);
}
.basemap-btn.active {
  color: var(--cyan);
  background: var(--cyan-dim);
  border-color: rgba(0,212,255,.2);
}

/* Event info in sidebar */
.event-info { display: flex; flex-direction: column; gap: 6px; }
.event-info__name { font-family: var(--font-head); font-size: 13px; font-weight: 600; color: var(--text-bright); }
.event-info__sub  { font-size: 11px; color: var(--text-muted); }
.event-info__row  { display: flex; align-items: center; gap: 8px; }
.event-info__stats { display: flex; flex-direction: column; gap: 4px; margin-top: 4px; }
.event-info__stats > div { display: flex; justify-content: space-between; align-items: center; }
.lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-dim); }
.val { font-size: 12px; color: var(--text-bright); }

/* ── Event panel (right sidebar) ── */
.event-panel {
  position: absolute;
  top: 16px;
  right: 16px;
  z-index: 10;
  width: 280px;
  background: rgba(3, 13, 26, 0.92);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  transition: width var(--t-med), height var(--t-med);
}
.event-panel.collapsed {
  width: 42px;
  height: 42px;
}
.event-panel__toggle {
  position: absolute;
  top: 8px;
  left: 8px;
  z-index: 2;
  width: 26px;
  height: 26px;
  padding: 0;
  font-size: 16px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: var(--radius);
  cursor: pointer;
}
.event-panel__inner {
  padding: 40px 12px 12px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  overflow-y: auto;
  max-height: calc(100vh - var(--nav-h) - 80px);
}
.event-panel__label {
  font-family: var(--font-head);
  font-size: 10px;
  font-weight: 600;
  letter-spacing: 0.14em;
  color: var(--text-dim);
  margin-bottom: 6px;
}
.event-pill {
  display: flex;
  align-items: center;
  gap: 7px;
  padding: 6px 10px;
  border-radius: var(--radius);
  background: transparent;
  border: 1px solid transparent;
  cursor: pointer;
  transition: all var(--t-fast);
  color: var(--text);
  font-family: var(--font-body);
  font-size: 12px;
  width: 100%;
  text-align: left;
}
.event-pill:hover {
  border-color: var(--border);
  background: var(--bg-3);
  color: var(--text-bright);
}
.event-pill.active {
  border-color: var(--ec, var(--cyan));
  background: rgba(0,212,255,.08);
  color: var(--text-bright);
}
.event-pill__dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  background: var(--ec, var(--cyan));
  flex-shrink: 0;
}
.event-pill__text {
  flex: 1;
  font-size: 12px;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.event-pill__type {
  font-size: 9px;
  letter-spacing: 0.06em;
  color: var(--text-muted);
  white-space: nowrap;
}

/* ── Popup ── */
.map-popup {
  position: fixed;
  z-index: 200;
  width: 220px;
  background: rgba(7, 21, 37, 0.97);
  backdrop-filter: blur(12px);
  border: 1px solid var(--border-2);
  border-radius: var(--radius-lg);
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  pointer-events: none;
  animation: fadeUp 0.15s ease both;
}
.map-popup__close {
  position: absolute;
  top: 8px; right: 10px;
  background: none; border: none;
  color: var(--text-muted); font-size: 16px;
  cursor: pointer; pointer-events: all;
  transition: color var(--t-fast);
}
.map-popup__close:hover { color: var(--text-bright); }
.map-popup__name {
  font-family: var(--font-head);
  font-size: 13px;
  font-weight: 600;
  color: var(--text-bright);
  padding-right: 20px;
}
.map-popup__prob { display: flex; flex-direction: column; gap: 4px; }
.prob-bar-wrap {
  height: 6px;
  background: var(--bg-4);
  border-radius: 3px;
  overflow: hidden;
}
.prob-bar {
  height: 100%;
  border-radius: 3px;
  transition: width 0.3s ease;
}
.prob-val { font-size: 14px; font-weight: 600; }

/* ── Pixel tooltip ── */
.pixel-tooltip {
  position: absolute;
  z-index: 20;
  background: rgba(7, 21, 37, 0.95);
  backdrop-filter: blur(8px);
  border: 1px solid var(--border-2);
  border-radius: var(--radius);
  padding: 8px 12px;
  pointer-events: none;
  min-width: 120px;
  animation: fadeIn 0.1s ease;
}
.pixel-tooltip__row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 5px;
}
.pixel-tooltip__lbl {
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
}
.pixel-tooltip__val {
  font-size: 15px;
  font-weight: 700;
}
.pixel-tooltip__bar {
  height: 4px;
  background: var(--bg-4);
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 5px;
}
.pixel-tooltip__fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.1s ease;
}
.pixel-tooltip__meta {
  font-size: 9px;
  letter-spacing: 0.08em;
  color: var(--text-dim);
}

/* ── Status bar ── */
.status-bar {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  padding: 6px 16px;
  background: rgba(3, 13, 26, 0.85);
  border-top: 1px solid var(--border);
  backdrop-filter: blur(8px);
}

/* Override MapLibre controls styling */
:deep(.maplibregl-ctrl-group) {
  background: rgba(7, 21, 37, 0.95) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: none !important;
}
:deep(.maplibregl-ctrl-group button) {
  color: var(--text) !important;
  background: transparent !important;
}
:deep(.maplibregl-ctrl-group button:hover) {
  background: var(--bg-3) !important;
  color: var(--text-bright) !important;
}
:deep(.maplibregl-ctrl-attrib) {
  background: rgba(3,13,26,0.7) !important;
  color: var(--text-dim) !important;
  font-size: 10px !important;
}
:deep(.maplibregl-ctrl-attrib a) {
  color: var(--text-muted) !important;
}

/* ── Chart panel ── */
.chart-panel {
  position: absolute;
  bottom: 32px;   /* above status bar */
  right: 16px;
  z-index: 10;
  width: 440px;
  max-width: calc(100vw - 32px);
  display: flex;
  flex-direction: column;
  gap: 8px;
  align-items: flex-end;
}
.chart-panel__toggle {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 11px;
  padding: 6px 14px;
  background: rgba(3, 13, 26, 0.92);
  backdrop-filter: blur(10px);
  border-color: var(--border-2);
}
.chart-panel__toggle:hover {
  border-color: var(--cyan);
  color: var(--cyan);
}
.chart-panel__body {
  width: 100%;
  background: rgba(3, 13, 26, 0.95);
  backdrop-filter: blur(12px);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

</style>
