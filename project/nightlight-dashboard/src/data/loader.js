// =====================================================
// Data loader — tries real exported files first,
// falls back to mock generators if files not found.
// =====================================================

import { generateMockProbabilityGeoJSON, generateBufferGeoJSON, generateFacilityGeoJSON } from './events.js'
import { generateRecoverySeries } from './timeseries.js'

const BASE = import.meta.env.BASE_URL

export async function loadProbabilityGeoJSON(event) {
  try {
    const res = await fetch(`${BASE}data/prob_${event.id}.geojson`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  } catch {
    return generateMockProbabilityGeoJSON(event)
  }
}

export async function loadFacilityGeoJSON(event) {
  try {
    const res = await fetch(`${BASE}data/facilities_${event.id}.json`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const facilities = await res.json()
    return {
      type: 'FeatureCollection',
      features: facilities.map(f => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: f.coords },
        properties: { name: f.name, type: f.type, probability: f.probability },
      })),
    }
  } catch {
    return generateFacilityGeoJSON(event)
  }
}

export async function loadTimeSeries(event) {
  try {
    const res = await fetch(`${BASE}data/ts_${event.id}.json`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  } catch {
    return generateRecoverySeries(event.id, 30, 60)
  }
}

export async function loadLoeoResults() {
  try {
    const res = await fetch(`${BASE}data/loeo_results.json`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  } catch {
    return null
  }
}
