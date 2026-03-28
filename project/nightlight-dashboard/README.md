# NightLight Dashboard

Interactive research dashboard for the **Critical Infrastructure Resilience Detection** project.
Built with **Vite + Vue 3 + MapLibre GL JS**.

---

## Quick Start

```bash
# Install dependencies
npm install

# Start dev server (http://localhost:5173)
npm run dev

# Production build
npm run build
npm run preview
```

---

## Project Structure

```
src/
├── main.js              # App entry point
├── App.vue              # Root layout
├── router/index.js      # Vue Router routes
├── assets/global.css    # Design tokens + global styles
├── data/
│   └── events.js        # ← Plug in your real data here
├── components/
│   └── NavBar.vue
└── views/
    ├── HomeView.vue     # Landing page
    ├── MapView.vue      # Interactive map
    └── DocsView.vue     # Technical documentation
```

---

## Replacing Mock Data with Real Predictions

### Option A: GeoJSON (point cloud)

In `src/data/events.js`, replace `generateMockProbabilityGeoJSON()` with a function
that returns your real model predictions as a GeoJSON FeatureCollection:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "Point", "coordinates": [-66.1057, 18.4655] },
      "properties": { "probability": 0.87 }
    }
  ]
}
```

### Option B: Raster Tiles (recommended for large datasets)

1. Export predictions as GeoTIFF from Python/R
2. Convert to MBTiles: `gdal2tiles.py` or `rio mbtiles`
3. Serve tiles via a tile server (e.g., TileServer-GL, Martin)
4. In `MapView.vue`, replace the heatmap layer with a raster source:

```javascript
map.addSource('prob-raster', {
  type: 'raster',
  tiles: ['https://your-tile-server/{z}/{x}/{y}.png'],
  tileSize: 256,
})
map.addLayer({
  id: 'prob-raster-layer',
  type: 'raster',
  source: 'prob-raster',
  paint: { 'raster-opacity': 0.75 },
})
```

---

## Adding New Events

In `src/data/events.js`, add a new entry to the `EVENTS` array:

```javascript
{
  id: 'ian',
  name: 'Hurricane Ian',
  subtitle: 'Fort Myers, Florida',
  year: 2022,
  date: 'Sep 28, 2022',
  type: 'hurricane',
  center: [-81.8723, 26.6406],
  zoom: 11.5,
  color: '#7ec8e3',
  affectedUsers: '2.6M+',
  outageDuration: '2–4 weeks',
  description: '...',
  facilities: [
    { name: 'Lee Health Cape Coral Hospital', type: 'hospital', coords: [-81.9901, 26.6194], probability: 0.84 },
    // ...
  ],
}
```

---

## Map Base Style

The dashboard uses **CARTO Dark Matter** tiles (free, no API key):
```
https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json
```

To switch to Mapbox (higher quality):
1. Get a token at https://account.mapbox.com
2. In `MapView.vue`, replace the style URL and add `transformRequest`:

```javascript
map = new maplibregl.Map({
  container: mapContainer.value,
  style: 'mapbox://styles/mapbox/dark-v11',
  // MapLibre + Mapbox token
  transformRequest: (url) => {
    if (url.startsWith('https://api.mapbox.com')) {
      return { url: url + '?access_token=YOUR_TOKEN' }
    }
  },
})
```

---

## Tech Stack

| Package       | Version | Purpose            |
|---------------|---------|--------------------|
| vue           | 3.4+    | UI framework       |
| vue-router    | 4.3+    | Client-side routing |
| maplibre-gl   | 4.7+    | Interactive map    |
| vite          | 5.4+    | Build tool         |
