// =====================================================
// Mock time-series recovery data generator
// Replace generateRecoverySeries() with your actual
// daily NTL ratio outputs per event.
//
// Expected output format (one entry per day):
// { day: number, R_buffer: number, R_nonBuffer: number, date: string }
// =====================================================

import { EVENTS } from './events.js'

function seededRandom(seed) {
  let s = seed
  return function () {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5
    return (s >>> 0) / 4294967296
  }
}

/**
 * Generate a plausible NTL recovery curve for one event.
 * Pre-disaster: R ≈ 1.0 ± noise
 * Post-disaster: sudden drop, then gradual recovery
 * Buffer zone recovers faster than non-buffer
 */
export function generateRecoverySeries(eventId, preDays = 30, postDays = 60) {
  const ev  = EVENTS.find(e => e.id === eventId)
  if (!ev) return []

  const rng = seededRandom(eventId.split('').reduce((a, c) => a + c.charCodeAt(0), 0))

  // Event-specific parameters (simulate realistic diversity)
  const params = {
    'maria':   { drop: 0.72, bufferAdvantage: 0.22, recoveryDays: 55, noiseLevel: 0.04 },
    'michael': { drop: 0.55, bufferAdvantage: 0.14, recoveryDays: 22, noiseLevel: 0.03 },
    'eq-pr':   { drop: 0.48, bufferAdvantage: 0.18, recoveryDays: 32, noiseLevel: 0.05 },
    'ida':     { drop: 0.61, bufferAdvantage: 0.17, recoveryDays: 30, noiseLevel: 0.03 },
    'laura':   { drop: 0.58, bufferAdvantage: 0.16, recoveryDays: 40, noiseLevel: 0.04 },
    'irma':    { drop: 0.42, bufferAdvantage: 0.11, recoveryDays: 14, noiseLevel: 0.03 },
  }
  const p = params[eventId] ?? { drop: 0.5, bufferAdvantage: 0.15, recoveryDays: 30, noiseLevel: 0.04 }

  const series = []

  for (let day = -preDays; day <= postDays; day++) {
    const t        = day                       // relative day (0 = disaster)
    const noise    = () => (rng() - 0.5) * 2 * p.noiseLevel

    // Recovery curve: logistic-ish, faster for buffer
    const recFrac  = (d, k) => d <= 0 ? 1 : Math.min(1, (1 - p.drop) * (1 - Math.exp(-k * d / p.recoveryDays)) + (1 - p.drop))
    const nonBufR  = t < 0 ? 1 + noise() : (1 - p.drop) * recFrac(t, 1.8) + noise()
    const bufR     = t < 0 ? 1 + noise() : nonBufR + (t >= 0 ? p.bufferAdvantage * recFrac(t, 2.5) : 0) + noise() * 0.5

    series.push({
      day,
      date:        `Day ${day >= 0 ? '+' : ''}${day}`,
      R_buffer:    Math.max(0.01, Math.min(1.4, parseFloat(bufR.toFixed(3)))),
      R_nonBuffer: Math.max(0.01, Math.min(1.4, parseFloat(nonBufR.toFixed(3)))),
      isPostDisaster: day >= 0,
    })
  }

  return series
}

/**
 * Compute summary stats from a recovery series
 */
export function computeResilienceStats(series) {
  const post = series.filter(d => d.isPostDisaster)
  if (!post.length) return null

  const meanBuf    = post.reduce((s, d) => s + d.R_buffer, 0) / post.length
  const meanNonBuf = post.reduce((s, d) => s + d.R_nonBuffer, 0) / post.length
  const minR       = Math.min(...post.map(d => d.R_nonBuffer))
  const ra         = parseFloat((meanBuf - meanNonBuf).toFixed(3))

  // Recovery day = first day non-buffer R > 0.9
  const recoveryDay = post.find(d => d.R_nonBuffer > 0.9)?.day ?? null

  return { meanBuf, meanNonBuf, minR, ra, recoveryDay }
}
