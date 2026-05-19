// NLCD export helper for six-event modeling
// CONUS: 2021 NLCD; Puerto Rico events: 2016 NLCD

var events = {
  maria_sanjuan: {bounds: [-66.20, 18.35, -65.95, 18.48], isPR: true},
  michael_panamacity: {bounds: [-85.80, 30.10, -85.55, 30.25], isPR: false},
  earthquake_sanjuan: {bounds: [-66.20, 18.35, -65.95, 18.48], isPR: true},
  ida_neworleans: {bounds: [-90.20, 29.87, -89.90, 30.08], isPR: false},
  laura_lakecharles: {bounds: [-93.35, 30.10, -93.05, 30.30], isPR: false},
  irma_miami: {bounds: [-80.35, 25.70, -80.10, 25.90], isPR: false}
};

var nlcdConus = ee.Image('USGS/NLCD_RELEASES/2021_REL/NLCD/2021').select('landcover');
var nlcdPR = ee.Image('USGS/NLCD_RELEASES/2016_REL/NLCD/2016').select('landcover');

Object.keys(events).forEach(function(eid) {
  var cfg = events[eid];
  var roi = ee.Geometry.Rectangle(cfg.bounds);
  var nlcd = cfg.isPR ? nlcdPR : nlcdConus;

  var nlcd500 = nlcd
    .reduceResolution({reducer: ee.Reducer.mode(), maxPixels: 4096})
    .reproject({crs: 'EPSG:4326', scale: 500});

  Export.image.toDrive({
    image: nlcd500,
    description: 'nlcd_' + eid,
    folder: 'Practicum_NLCD',
    fileNamePrefix: 'nlcd_' + eid,
    region: roi,
    scale: 500,
    crs: 'EPSG:4326',
    maxPixels: 1e13
  });
});
