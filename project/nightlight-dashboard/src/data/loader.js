// =====================================================
// Data loader — tries real exported files first,
// falls back to mock generators if files not found.
// =====================================================

import { generateMockProbabilityGeoJSON, generateBufferGeoJSON, generateFacilityGeoJSON } from './events.js'
import { generateRecoverySeries } from './timeseries.js'

/**
 * Load probability GeoJSON for an event.
 * Tries /data/prob_<id>.geojson first; falls back to mock.
 */
export async function loadProbabilityGeoJSON(event) {
  try {
    const res = await fetch(`/data/prob_${event.id}.geojson`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    console.log(`[loader] Loaded real prob GeoJSON for ${event.id} (${data.features.length} px)`)
    return data
  } catch {
    console.log(`[loader] Using mock prob GeoJSON for ${event.id}`)
    return generateMockProbabilityGeoJSON(event)
  }
}

/**
 * Load facility GeoJSON for an event.
 * Tries /data/facilities_<id>.json; falls back to mock.
 */
export async function loadFacilityGeoJSON(event) {
  try {
    const res = await fetch(`/data/facilities_${event.id}.json`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const facilities = await res.json()
    // Convert to GeoJSON FeatureCollection
    return {
      type: 'FeatureCollection',
      features: facilities.map(f => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: f.coords },
        properties: { name: f.name, type: f.type, probability: f.probability },
      })),
    }
  } catch {
    console.log(`[loader] Using mock facility GeoJSON for ${event.id}`)
    return generateFacilityGeoJSON(event)
  }
}

/**
 * Load daily time-series for an event.
 * Tries /data/ts_<id>.json; falls back to mock.
 */
export async function loadTimeSeries(event) {
  try {
    const res = await fetch(`/data/ts_${event.id}.json`)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    const data = await res.json()
    console.log(`[loader] Loaded real time-series for ${event.id} (${data.length} days)`)
    return data
  } catch {
    console.log(`[loader] Using mock time-series for ${event.id}`)
    return generateRecoverySeries(event.id, 30, 60)
  }
}

/**
 * Load LOEO results summary.
 */
export async function loadLoeoResults() {
  try {
    const res = await fetch('/data/loeo_results.json')
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  } catch {
    return null
  }
}
