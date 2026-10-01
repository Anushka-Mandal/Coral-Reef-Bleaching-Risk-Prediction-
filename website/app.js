/* CoralWatch - live reef map
 *
 * Data:
 *   window.GBR        Marine Park boundary + reef outlines (data/gbr.js, GBRMPA open data)
 *   window.SNAPSHOTS  saved NOAA Coral Reef Watch days (data/snapshots.js)
 *   Other dates are fetched live from NOAA ERDDAP in the browser.
 * Grid format: { date, lats[] north->south, lons[] west->east, step, vars: { baa, dhw, ssta, sst, hotspot } }
 * Each vars array is flattened row by row; null = land or outside the Marine Park.
 */

const ERDDAP = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km.json";
const LIVE_STRIDE = 2;

const BAA_LEVELS = [
  { name: "No Stress",     color: "#8ec9f2", desc: "Temperatures are normal for the time of year." },
  { name: "Watch",         color: "#f8d74a", desc: "Water is warmer than the usual summer maximum." },
  { name: "Warning",       color: "#fb953a", desc: "Heat stress is starting to build up (DHW above 0)." },
  { name: "Alert Level 1", color: "#ff5236", desc: "DHW 4–8: significant bleaching is likely." },
  { name: "Alert Level 2", color: "#d4145a", desc: "DHW 8+: severe bleaching and significant coral death are likely." },
];

// Dark-theme ramps: dark = low, bright = high
const HEAT = ["#132a45", "#4a2c6e", "#9b2f67", "#dd4b4b", "#f98a3c", "#fdd26b", "#fff6c2"];
const DIVERGING = ["#8fd0ff", "#3d86d6", "#1d4577", "#27313d", "#7a2f36", "#d24b3c", "#ff9a70"];
const TEAL = ["#0b2436", "#0d4659", "#127679", "#20a89c", "#5eead4", "#c8fff5"];

const LAYERS = {
  baa: {
    label: "Bleaching Alert Area",
    desc: "NOAA's heat-stress level, from No Stress to Alert Level 2. This is the scale our model forecasts.",
  },
  dhw: {
    label: "Degree Heating Weeks", unit: "°C-weeks", min: 0, max: 16, ramp: HEAT,
    desc: "Heat stress accumulated over 12 weeks. 4+ means significant bleaching is likely; 8+ means severe bleaching.",
  },
  hotspot: {
    label: "HotSpot", unit: "°C", min: 0, max: 3, ramp: HEAT,
    desc: "How far today's temperature is above the usual warmest-month average.",
  },
  ssta: {
    label: "Temperature anomaly", unit: "°C", min: -3, max: 3, ramp: DIVERGING,
    desc: "Today's sea surface temperature compared with the long-term average for this day.",
  },
  sst: {
    label: "Sea surface temperature", unit: "°C", min: 22, max: 32, ramp: TEAL,
    desc: "Daily satellite sea surface temperature (CoralTemp, 5 km).",
  },
  forecast: {
    label: "Forecast (model)",
    desc: "The model's forecast of NOAA's 7-day maximum alert: the highest alert level expected over the 7 days ending on the chosen lead day. Built from NOAA data up to the selected date plus the weather forecast.",
  },
};

// Peak alert day of each recent bleaching summer in the study area
const EVENT_LABELS = {
  "2025-03-09": "Mar 2025",
  "2024-03-03": "Mar 2024",
  "2022-03-12": "Mar 2022",
  "2020-03-08": "Mar 2020",
  "2017-03-21": "Mar 2017",
  "2016-03-23": "Mar 2016",
};
const SHOWCASE_DATE = "2024-03-03";

const GEO = window.GBR || { boundary: [], reefs: [], study: { north: -10.4, south: -21, split: -16.5 } };
const STUDY = GEO.study;
const ZONES = [
  { id: "N", name: "Northern GBR", north: STUDY.north, south: STUDY.split },
  { id: "C", name: "Central GBR",  north: STUDY.split, south: STUDY.south },
];

const state = {
  grids: Object.assign({}, window.SNAPSHOTS || {}),
  latest: window.SNAPSHOT_LATEST || null,
  date: null,
  layer: "baa",
  live: false,
  hoverReef: null,
  api: false,          // true when the CoralWatch server is running (enables live data, forecast, AI)
  forecasts: {},       // date -> forecast from /api/forecast
  forecastOK: false,
  aiZone: "N",
  lead: 7,             // forecast lead time in days (1, 3, 7 or 14)
};

// ── Shared helpers (also used by landing.js) ────────────
const $ = (id) => document.getElementById(id);

const rgbCache = new Map();
function rgb(hex) {
  if (!rgbCache.has(hex)) {
    const n = parseInt(hex.slice(1), 16);
    rgbCache.set(hex, [(n >> 16) & 255, (n >> 8) & 255, n & 255]);
  }
  return rgbCache.get(hex);
}
function rampColor(ramp, t) {
  t = Math.min(1, Math.max(0, t));
  const x = t * (ramp.length - 1);
  const i = Math.min(ramp.length - 2, Math.floor(x));
  const f = x - i, a = rgb(ramp[i]), b = rgb(ramp[i + 1]);
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
function colorFor(layer, v) {
  if (v === null || v === undefined) return null;
  if (layer === "baa") return rgb(BAA_LEVELS[Math.max(0, Math.min(4, v))].color);
  const L = LAYERS[layer];
  return rampColor(L.ramp, (v - L.min) / (L.max - L.min));
}
function fmtDate(iso) {
  return new Date(iso + "T00:00:00Z").toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });
}
const fmt = (v, d, unit = "") => (v === null || v === undefined ? "–" : `${v.toFixed(d)}${unit ? " " + unit : ""}`);

function pointInRing(lat, lon, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [yi, xi] = ring[i], [yj, xj] = ring[j];
    if ((yi > lat) !== (yj > lat) && lon < ((xj - xi) * (lat - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// Clip a [lat, lon] ring to latitudes between north and south (Sutherland–Hodgman on two edges)
function clipRing(ring, north, south) {
  const clipEdge = (pts, keep, cut) => {
    const out = [];
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      const ina = keep(a[0]), inb = keep(b[0]);
      if (ina) out.push(a);
      if (ina !== inb) {
        const t = (cut - a[0]) / (b[0] - a[0]);
        out.push([cut, a[1] + t * (b[1] - a[1])]);
      }
    }
    return out;
  };
  return clipEdge(clipEdge(ring, (lat) => lat <= north, north), (lat) => lat >= south, south);
}

// Min / max longitude of the park at a given latitude
function parkSpanAt(lat) {
  const ring = GEO.boundary, xs = [];
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [yi, xi] = ring[i], [yj, xj] = ring[j];
    if ((yi > lat) !== (yj > lat)) xs.push(xi + ((lat - yi) / (yj - yi)) * (xj - xi));
  }
  return xs.length ? [Math.min(...xs), Math.max(...xs)] : null;
}

// Value at a reef: nearest data cell to its centre (searching up to 2 cells away)
const reefIndexCache = new Map();
function reefCellIndex(g) {
  const key = `${g.lats.length}x${g.lons.length}x${g.lats[0]}`;
  if (reefIndexCache.has(key)) return reefIndexCache.get(key);
  const nLat = g.lats.length, nLon = g.lons.length, baa = g.vars.baa;
  const idx = GEO.reefs.map((r) => {
    const i0 = Math.round((g.lats[0] - r.c[0]) / g.step), j0 = Math.round((r.c[1] - g.lons[0]) / g.step);
    for (let rad = 0; rad <= 2; rad++) {
      for (let di = -rad; di <= rad; di++) for (let dj = -rad; dj <= rad; dj++) {
        const i = i0 + di, j = j0 + dj;
        if (i >= 0 && j >= 0 && i < nLat && j < nLon && baa[i * nLon + j] !== null) return i * nLon + j;
      }
    }
    return -1;
  });
  reefIndexCache.set(key, idx);
  return idx;
}

// ── Map ─────────────────────────────────────────────────
let map = null, overlay = null, reefLayer = null, infoBox = null;
const studyRing = clipRing(GEO.boundary, STUDY.north, STUDY.south);

function initMap() {
  if (!window.L || !$("leaflet-map")) return;
  map = L.map("leaflet-map", { zoomSnap: 0.25, minZoom: 5, maxZoom: 12, scrollWheelZoom: false, attributionControl: true });
  map.on("focus", () => map.scrollWheelZoom.enable());
  map.on("blur", () => map.scrollWheelZoom.disable());

  map.createPane("labels"); map.getPane("labels").style.zIndex = 450; map.getPane("labels").style.pointerEvents = "none";
  map.createPane("mask"); map.getPane("mask").style.zIndex = 420; map.getPane("mask").style.pointerEvents = "none";
  map.createPane("reefs"); map.getPane("reefs").style.zIndex = 430;
  map.createPane("outline"); map.getPane("outline").style.zIndex = 440; map.getPane("outline").style.pointerEvents = "none";

  const esri = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas";
  L.tileLayer(`${esri}/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}`, {
    maxNativeZoom: 16,
    attribution: 'Tiles &copy; Esri · Data: <a href="https://coralreefwatch.noaa.gov/">NOAA Coral Reef Watch</a> · Reefs &copy; GBRMPA',
  }).addTo(map);
  L.tileLayer(`${esri}/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}`, { pane: "labels", maxNativeZoom: 16 }).addTo(map);

  if (studyRing.length) {
    // Spotlight: dim everything outside the study area of the Marine Park
    const world = [[-89, -179], [-89, 179], [89, 179], [89, -179]];
    L.polygon([world, studyRing], { pane: "mask", stroke: false, fillColor: "#02060c", fillOpacity: 0.55, interactive: false }).addTo(map);

    const svg = L.svg({ pane: "outline" });
    L.polygon(GEO.boundary, { renderer: svg, color: "#5eead4", weight: 1, opacity: 0.35, dashArray: "3 5", fill: false, interactive: false }).addTo(map);
    L.polygon(studyRing, { renderer: svg, color: "#5eead4", weight: 1.6, fill: false, className: "park-glow", interactive: false }).addTo(map);

    // Northern / central split
    const span = parkSpanAt(STUDY.split);
    if (span) L.polyline([[STUDY.split, span[0]], [STUDY.split, span[1]]], { renderer: svg, color: "#a9bfd8", weight: 1, opacity: 0.6, dashArray: "5 6", interactive: false }).addTo(map);

    ZONES.forEach((z) => {
      const lat = (z.north + z.south) / 2 + (z.id === "N" ? 0.8 : 0);
      const s = parkSpanAt(lat);
      if (s) L.tooltip({ permanent: true, direction: "right", className: "zone-tag", offset: [8, 0] }).setLatLng([lat, s[1]]).setContent(z.name).addTo(map);
    });

    const b = L.latLngBounds(studyRing);
    map.fitBounds(b, { padding: [20, 20] });
    map.setMaxBounds(b.pad(0.6));
  }

  // Reef outlines (drawn on canvas for speed, shown when zoomed in)
  const canvas = L.canvas({ pane: "reefs", padding: 0.3 });
  reefLayer = L.layerGroup(GEO.reefs.map((r, k) => {
    const poly = L.polygon(r.r.map((flat) => {
      const pts = [];
      for (let i = 0; i < flat.length; i += 2) pts.push([flat[i], flat[i + 1]]);
      return pts;
    }), { renderer: canvas, color: "#e8f3ff", weight: 0.8, opacity: 0.7, fillOpacity: 0, bubblingMouseEvents: true });
    poly.on("mouseover", () => { state.hoverReef = k; poly.setStyle({ color: "#5eead4", weight: 2, opacity: 1 }); });
    poly.on("mouseout", () => { state.hoverReef = null; poly.setStyle({ color: "#e8f3ff", weight: 0.8, opacity: 0.7 }); });
    return poly;
  }));
  const toggleReefs = () => (map.getZoom() >= 7.5 ? map.addLayer(reefLayer) : map.removeLayer(reefLayer));
  map.on("zoomend", toggleReefs);
  toggleReefs();

  // Hover info
  const InfoBox = L.Control.extend({
    onAdd() { this._div = L.DomUtil.create("div", "info-box glass"); this.update(null); return this._div; },
    update(cell) {
      if (!cell) { this._div.innerHTML = '<p class="empty">Hover over the reef to see values</p>'; return; }
      const v = cell.values, lvl = v.baa === null ? null : BAA_LEVELS[v.baa];
      const title = state.hoverReef !== null ? GEO.reefs[state.hoverReef].n : "Marine Park waters";
      this._div.innerHTML = `
        <h4>${title}</h4>
        <div class="coords">${Math.abs(cell.lat).toFixed(2)}°S, ${cell.lon.toFixed(2)}°E · ${fmtDate(state.date)}</div>
        <table>
          <tr><td>Alert level</td><td>${lvl ? `<span class="info-level"><i style="background:${lvl.color};box-shadow:0 0 8px ${lvl.color}"></i>${lvl.name}</span>` : "–"}</td></tr>
          <tr><td>Degree Heating Weeks</td><td>${fmt(v.dhw, 1)}</td></tr>
          <tr><td>HotSpot</td><td>${fmt(v.hotspot, 2, "°C")}</td></tr>
          <tr><td>Anomaly</td><td>${v.ssta === null ? "–" : (v.ssta > 0 ? "+" : "") + v.ssta.toFixed(2) + " °C"}</td></tr>
          <tr><td>Sea temperature</td><td>${fmt(v.sst, 1, "°C")}</td></tr>
          ${v.forecast !== undefined && v.forecast !== null ? `<tr><td>Forecast (+${state.lead} d)</td><td><span class="info-level"><i style="background:${BAA_LEVELS[v.forecast].color}"></i>${BAA_LEVELS[v.forecast].name}</span></td></tr>` : ""}
          ${v.p_alert !== undefined && v.p_alert !== null ? `<tr><td>Chance of Alert</td><td>${Math.round(v.p_alert * 100)}%</td></tr>` : ""}
        </table>`;
    },
  });
  infoBox = new InfoBox({ position: "topright" });
  infoBox.addTo(map);
  L.DomEvent.disableClickPropagation(infoBox.getContainer());

  map.on("mousemove", (e) => infoBox.update(cellAt(e.latlng)));
  map.on("mouseout", () => infoBox.update(null));
  map.on("click", (e) => infoBox.update(cellAt(e.latlng)));
}

function cellAt(latlng) {
  const g = state.grids[state.date];
  if (!g) return null;
  const i = Math.round((g.lats[0] - latlng.lat) / g.step), j = Math.round((latlng.lng - g.lons[0]) / g.step);
  if (i < 0 || j < 0 || i >= g.lats.length || j >= g.lons.length) return null;
  const k = i * g.lons.length + j;
  if (g.vars.baa[k] === null) return null;
  const values = {};
  for (const key of Object.keys(g.vars)) values[key] = g.vars[key][k];
  const f = state.forecasts[state.date];
  if (f) {
    const fi = Math.round((f.lats[0] - latlng.lat) / f.step), fj = Math.round((latlng.lng - f.lons[0]) / f.step);
    if (fi >= 0 && fj >= 0 && fi < f.lats.length && fj < f.lons.length) {
      values.forecast = forecastValues(f)[fi * f.lons.length + fj];
      const pa = f.by_lead_p_alert ? f.by_lead_p_alert[String(state.lead)] : f.vars.p_alert;
      if (pa) values.p_alert = pa[fi * f.lons.length + fj];
    }
  }
  return { lat: g.lats[i], lon: g.lons[j], values };
}

// Rows are resampled on a Mercator scale so every cell sits at its true latitude
const mercY = (lat) => Math.log(Math.tan(Math.PI / 4 + (lat * Math.PI) / 360));
const invMercY = (y) => ((Math.atan(Math.exp(y)) - Math.PI / 4) * 360) / Math.PI;

function layerSource() {
  if (state.layer === "forecast") {
    const f = state.forecasts[state.date];
    return f ? { g: f, values: forecastValues(f), colorKey: "baa" } : null;
  }
  const g = state.grids[state.date];
  return g ? { g, values: g.vars[state.layer], colorKey: state.layer } : null;
}

function forecastValues(f) {
  return (f.by_lead_levels && f.by_lead_levels[String(state.lead)]) || f.vars.forecast;
}

function forecastTarget(f) {
  if (!f.by_lead_levels) return f.target_date;
  const d = new Date(f.issue_date + "T00:00:00Z");
  d.setUTCDate(d.getUTCDate() + state.lead);
  return d.toISOString().slice(0, 10);
}

function drawOverlay() {
  const src = layerSource();
  if (!src || !map) {
    if (overlay && state.layer === "forecast") overlay.setUrl("data:image/gif;base64,R0lGODlhAQABAAAAACw=");
    return;
  }
  const { g, values, colorKey } = src;
  const nLat = g.lats.length, nLon = g.lons.length, half = g.step / 2;
  const north = g.lats[0] + half, south = g.lats[nLat - 1] - half, west = g.lons[0] - half, east = g.lons[nLon - 1] + half;
  const H = nLat * 3, yN = mercY(north), yS = mercY(south);
  const cv = document.createElement("canvas");
  cv.width = nLon; cv.height = H;
  const ctx = cv.getContext("2d"), img = ctx.createImageData(nLon, H);
  for (let r = 0; r < H; r++) {
    const lat = invMercY(yN + ((r + 0.5) / H) * (yS - yN));
    const i = Math.min(nLat - 1, Math.max(0, Math.round((g.lats[0] - lat) / g.step)));
    for (let j = 0; j < nLon; j++) {
      const c = colorFor(colorKey, values[i * nLon + j]);
      if (!c) continue;
      const p = (r * nLon + j) * 4;
      img.data[p] = c[0]; img.data[p + 1] = c[1]; img.data[p + 2] = c[2]; img.data[p + 3] = 235;
    }
  }
  ctx.putImageData(img, 0, 0);
  const bounds = L.latLngBounds([[south, west], [north, east]]);
  if (overlay) { overlay.setUrl(cv.toDataURL()); overlay.setBounds(bounds); }
  else overlay = L.imageOverlay(cv.toDataURL(), bounds, { className: "data-overlay", interactive: false }).addTo(map);
}

function drawLegend() {
  const el = $("map-legend");
  $("layer-desc").textContent = LAYERS[state.layer].desc;
  if (state.layer === "forecast") {
    const f = state.forecasts[state.date];
    const leads = f && f.by_lead_levels ? [1, 3, 7, 14] : [];
    const title = !f ? "Forecast loading…" : f.target === "max7" ? `Highest alert, week to ${fmtDate(forecastTarget(f))}` : `Forecast for ${fmtDate(forecastTarget(f))}`;
    el.innerHTML = `<h4>${title}</h4>
      ${leads.length ? `<div class="lead-row">${leads.map((k) => `<button type="button" class="lead-btn" data-lead="${k}" aria-pressed="${k === state.lead}">${k}d</button>`).join("")}</div>` : ""}
      <div class="legend-steps">${BAA_LEVELS.map((l) =>
      `<div class="legend-step"><i style="--c:${l.color}"></i>${l.name}</div>`).join("")}</div>`;
    el.querySelectorAll(".lead-btn").forEach((b) => b.addEventListener("click", () => {
      state.lead = Number(b.dataset.lead);
      drawOverlay(); drawLegend(); drawZones();
    }));
    return;
  }
  if (state.layer === "baa") {
    el.innerHTML = `<h4>Bleaching Alert Area</h4><div class="legend-steps">${BAA_LEVELS.map((l) =>
      `<div class="legend-step"><i style="--c:${l.color}"></i>${l.name}</div>`).join("")}</div>`;
    return;
  }
  const L_ = LAYERS[state.layer];
  const stops = L_.ramp.map((c, i) => `${c} ${(i / (L_.ramp.length - 1)) * 100}%`).join(",");
  el.innerHTML = `<h4>${L_.label} (${L_.unit})</h4>
    <div class="legend-bar" style="background:linear-gradient(90deg,${stops})"></div>
    <div class="legend-ticks"><span>${L_.min}</span><span>${(L_.min + L_.max) / 2}</span><span>${L_.max}+</span></div>`;
}

// ── Zone statistics ─────────────────────────────────────
function zoneStats(g, z) {
  const counts = [0, 0, 0, 0, 0];
  let n = 0, dhwMax = 0, sstaSum = 0;
  for (let i = 0; i < g.lats.length; i++) {
    const lat = g.lats[i];
    if (lat > z.north || lat <= z.south) continue;
    for (let j = 0; j < g.lons.length; j++) {
      const k = i * g.lons.length + j, baa = g.vars.baa[k];
      if (baa === null) continue;
      n++; counts[baa]++;
      const d = g.vars.dhw[k] ?? 0;
      if (d > dhwMax) dhwMax = d;
      sstaSum += g.vars.ssta[k] ?? 0;
    }
  }
  const idx = reefCellIndex(g);
  let reefs = 0, reefsAlert = 0;
  GEO.reefs.forEach((r, k) => {
    if (r.z !== z.id || idx[k] < 0) return;
    reefs++;
    if (g.vars.baa[idx[k]] >= 3) reefsAlert++;
  });
  const worst = counts.reduce((w, c, lvl) => (c > 0 ? lvl : w), 0);
  return { n, counts, pctAlert: n ? (100 * (counts[3] + counts[4])) / n : 0, worst, dhwMax, sstaMean: n ? sstaSum / n : 0, reefs, reefsAlert };
}

function allStats(g) { return ZONES.map((z) => ({ z, s: zoneStats(g, z) })); }

function drawZones() {
  const g = state.grids[state.date], stats = allStats(g);
  $("zones").innerHTML = stats.map(({ z, s }) => `
    <button class="zone" type="button" data-zone="${z.id}">
      <div class="zone-head">
        <span class="zone-name">${z.name}</span>
        <span class="zone-level"><i style="background:${BAA_LEVELS[s.worst].color};color:${BAA_LEVELS[s.worst].color}"></i>Worst: ${BAA_LEVELS[s.worst].name}</span>
      </div>
      <div class="zone-big"><b>${s.reefsAlert.toLocaleString()}</b><span>of ${s.reefs.toLocaleString()} reefs at<br>Alert Level 1 or higher</span></div>
      <div class="zone-meter" title="Share of zone area at each alert level">${s.counts.map((c, l) =>
        c ? `<i style="width:${(100 * c) / s.n}%;background:${BAA_LEVELS[l].color}" title="${BAA_LEVELS[l].name}: ${Math.round((100 * c) / s.n)}%"></i>` : "").join("")}</div>
      ${forecastLine(z.id)}
      <div class="zone-meta">
        <div><b>${Math.round(s.pctAlert)}%</b><span>area at Alert</span></div>
        <div><b>${s.dhwMax.toFixed(1)}</b><span>max DHW</span></div>
        <div><b>${s.sstaMean >= 0 ? "+" : ""}${s.sstaMean.toFixed(1)}°</b><span>mean anomaly</span></div>
      </div>
    </button>`).join("");
  document.querySelectorAll(".zone").forEach((btn) => btn.addEventListener("click", () => {
    const z = ZONES.find((x) => x.id === btn.dataset.zone);
    const ring = clipRing(GEO.boundary, z.north, z.south);
    if (ring.length) map.flyToBounds(L.latLngBounds(ring), { padding: [30, 30], duration: 0.8 });
  }));
  $("summary").textContent = buildSummary(stats);
}

function forecastLine(zoneId) {
  const f = state.forecasts[state.date];
  if (!f) return "";
  const byLead = f.forecast_zones_by_lead && f.forecast_zones_by_lead[String(state.lead)];
  const lead = byLead ? state.lead : 7;
  const z = (byLead || f.forecast_zones).find((x) => x.zone === zoneId);
  const prob = z.mean_alert_probability !== undefined ? ` · avg. chance of Alert ${Math.round(z.mean_alert_probability * 100)}%` : "";
  const tag = f.in_training_data ? " (training period)" : f.in_validation_data ? " (validation period)" : "";
  const when = fmtDate(byLead ? forecastTarget(f) : f.target_date);
  const what = f.target === "max7" ? `Week to ${when} (+${lead} d)` : `In ${lead} days (${when})`;
  return `<div class="forecast-line">${what}: <b>${z.reefs_at_alert.toLocaleString()}</b> reefs reach Alert Level 1+${prob}${tag}</div>`;
}

function buildSummary(stats) {
  const when = fmtDate(state.date), month = Number(state.date.slice(5, 7));
  const stressed = stats.some(({ s }) => s.counts[1] + s.counts[2] + s.counts[3] + s.counts[4] > s.n * 0.01);
  if (!stressed) {
    const season = month >= 5 && month <= 10 ? " This is normal for the southern winter (May–October), when bleaching risk is low." : "";
    return `On ${when}, the northern and central Great Barrier Reef show no significant bleaching heat stress.${season}`;
  }
  const parts = stats.map(({ z, s }) => {
    if (s.reefsAlert > 0) return `in the ${z.name}, ${s.reefsAlert.toLocaleString()} of ${s.reefs.toLocaleString()} reefs (${Math.round((100 * s.reefsAlert) / s.reefs)}%) are at Alert Level 1 or higher, with DHW up to ${s.dhwMax.toFixed(1)}`;
    const watch = Math.round((100 * (s.counts[1] + s.counts[2])) / s.n);
    return watch ? `the ${z.name} is ${watch}% under a Watch or Warning` : `the ${z.name} shows no significant stress`;
  });
  const maxDhw = Math.max(...stats.map(({ s }) => s.dhwMax));
  const risk = maxDhw >= 8 ? " DHW above 8 means severe bleaching and significant coral death are likely."
    : maxDhw >= 4 ? " DHW above 4 means significant bleaching is likely." : "";
  const txt = `On ${when}, ${parts.join("; ")}.${risk}`;
  return txt.charAt(0).toUpperCase() + txt.slice(1);
}

// ── Loading data ────────────────────────────────────────
function liveUrl(date) {
  const dims = `[(${date})][(${STUDY.north}):${LIVE_STRIDE}:(${STUDY.south})][(${STUDY.west}):${LIVE_STRIDE}:(${STUDY.east})]`;
  const q = ["CRW_BAA", "CRW_DHW", "CRW_SSTANOMALY", "CRW_SST", "CRW_HOTSPOT"].map((v) => v + dims).join(",");
  return ERDDAP + "?" + q.replace(/\[/g, "%5B").replace(/\]/g, "%5D");
}

function toGrid(table) {
  const cols = table.columnNames, rows = table.rows;
  const lats = [...new Set(rows.map((r) => r[1]))].sort((a, b) => b - a);
  const lons = [...new Set(rows.map((r) => r[2]))].sort((a, b) => a - b);
  const inside = rows.map((r) => pointInRing(r[1], r[2], GEO.boundary)); // clip to the Marine Park
  const keys = { baa: "CRW_BAA", dhw: "CRW_DHW", ssta: "CRW_SSTANOMALY", sst: "CRW_SST", hotspot: "CRW_HOTSPOT" };
  const vars = {};
  for (const [k, col] of Object.entries(keys)) {
    const c = cols.indexOf(col);
    vars[k] = rows.map((r, i) => (inside[i] ? r[c] : null));
  }
  return { date: rows[0][0].slice(0, 10), lats, lons, step: Math.abs(lats[1] - lats[0]), vars };
}

async function loadDate(date) {
  if (state.grids[date]) { state.live = false; return show(date); }
  $("map-loading").hidden = false;
  $("date-msg").textContent = "Fetching from NOAA… this takes about 10 seconds.";
  try {
    let grid;
    if (state.api) {
      const res = await fetch(`/api/grid/${date}`);
      if (!res.ok) throw new Error((await res.json()).detail || `Server replied ${res.status}`);
      grid = await res.json();
    } else {
      const res = await fetch(liveUrl(date));
      if (!res.ok) throw new Error(`NOAA replied ${res.status}`);
      grid = toGrid((await res.json()).table);
    }
    state.grids[grid.date] = grid;
    state.live = true;
    $("date-msg").textContent = "Loaded live from NOAA Coral Reef Watch.";
    show(grid.date);
  } catch (err) {
    console.error(err);
    $("date-msg").textContent = "Couldn't reach NOAA for that date. Check the connection, or use a saved date above.";
  } finally {
    $("map-loading").hidden = true;
  }
}

function show(date) {
  state.date = date;
  $("date-input").value = date;
  document.querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", String(c.dataset.date === date)));
  const src = $("source-line");
  src.classList.toggle("saved", !state.live);
  src.innerHTML = `<span class="dot"></span>NOAA Coral Reef Watch · ${fmtDate(date)} · ${state.live ? "live" : "saved"}`;
  drawOverlay();
  drawLegend();
  drawZones();
  if (state.forecastOK) loadForecast(date);
}

async function loadForecast(date) {
  if (state.forecasts[date] !== undefined) return;
  state.forecasts[date] = null; // in flight
  const url = date === state.latest && state.live ? "/api/forecast" : `/api/forecast?date=${date}`;
  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`forecast ${res.status}`);
    state.forecasts[date] = await res.json();
  } catch (e) {
    console.warn("Forecast unavailable for", date, e);
    delete state.forecasts[date];
    if (state.layer === "forecast" && state.date === date) $("layer-desc").textContent = "No forecast is available for this date (NOAA data for the week before is incomplete).";
    return;
  }
  if (state.date === date) { drawOverlay(); drawLegend(); drawZones(); }
}

function renderChips() {
  const chips = [];
  if (state.latest) chips.push({ date: state.latest, label: state.live ? "Today (live)" : "Latest" });
  Object.entries(EVENT_LABELS).forEach(([d, label]) => { if (state.grids[d]) chips.push({ date: d, label }); });
  $("date-chips").innerHTML = chips.map((c) =>
    `<button type="button" class="chip" data-date="${c.date}" aria-pressed="${c.date === state.date}">${c.label}</button>`).join("");
  document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => loadDate(c.dataset.date)));
}

function initControls() {
  const select = $("layer-select");
  if (!select) return;
  select.innerHTML = Object.entries(LAYERS).map(([k, l]) => `<option value="${k}">${l.label}</option>`).join("");
  select.addEventListener("change", () => { state.layer = select.value; drawOverlay(); drawLegend(); });

  renderChips();

  $("date-input").max = new Date().toISOString().slice(0, 10);
  $("date-form").addEventListener("submit", (e) => { e.preventDefault(); if ($("date-input").value) loadDate($("date-input").value); });
}

// ── AI briefing ─────────────────────────────────────────
function escapeHtml(t) {
  return t.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

// Tiny markdown renderer for the LLM output: ### headings, **bold**, *italic*, - bullets, [S#] citations
function renderMarkdown(md) {
  const out = [];
  let list = false;
  for (const raw of md.split("\n")) {
    const line = raw.trim();
    const inline = (t) => escapeHtml(t)
      .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
      .replace(/(^|[^*])\*(?!\s)(.+?)\*/g, "$1<i>$2</i>")
      .replace(/\[\s*S\s*(\d+)\s*\]/gi, '<span class="cite" title="Report passage S$1">S$1</span>')
      .replace(/\[\s*data\s*\]/gi, '<span class="cite cite-data" title="CoralWatch data: NOAA satellite values and the model forecast">Data</span>');
    if (/^[-•]\s+/.test(line)) {
      if (!list) { out.push("<ul>"); list = true; }
      out.push(`<li>${inline(line.replace(/^[-•]\s+/, ""))}</li>`);
      continue;
    }
    if (list) { out.push("</ul>"); list = false; }
    if (!line) continue;
    const h = line.match(/^#{1,4}\s+(.*)$/);
    out.push(h ? `<h4>${inline(h[1])}</h4>` : `<p>${inline(line)}</p>`);
  }
  if (list) out.push("</ul>");
  return out.join("");
}

function renderSources(sources) {
  const dataItem = '<li class="data-item"><b>CoralWatch data</b><span class="meta">NOAA satellite values · model forecast · measured skill</span></li>';
  $("ai-sources").innerHTML = dataItem + (sources.length ? sources.map((s) => {
    const meta = [s.source, s.published, s.page ? `p. ${s.page}` : null].filter(Boolean).join(" · ");
    const title = s.url ? `<a href="${s.url}" target="_blank" rel="noopener">${escapeHtml(s.title)}</a>` : escapeHtml(s.title);
    return `<li><b>${title}</b><span class="meta">${escapeHtml(meta)}</span><span class="snip">${escapeHtml(s.snippet)}</span></li>`;
  }).join("") : '<li class="muted-item">No sufficiently relevant report passages.</li>');
}

// Fact-check result: badge under the answer, unverified numbers highlighted in the text
function renderVerification(out, text, v) {
  let html = renderMarkdown(text);
  for (const n of v.unverified_numbers) {
    const esc = n.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    html = html.replace(new RegExp(`(^|[^\\w.])(${esc})(?![\\w])`, "g"), '$1<mark class="unverified" title="Not found in the data or the reports">$2</mark>');
  }
  const ok = v.grounded;
  const parts = [`${v.numbers_verified}/${v.numbers_checked} numbers match the data or reports`,
                 `${v.cited_factual_sentences}/${v.factual_sentences} factual sentences cited`];
  let extra = "";
  if (v.unverified_numbers.length) extra += `<div>Couldn't verify: ${v.unverified_numbers.map(escapeHtml).join(", ")}</div>`;
  if (v.uncited_claims.length) extra += `<div>Uncited: “${escapeHtml(v.uncited_claims[0])}”</div>`;
  if (v.removed_sentences) extra += `<div>Strict mode removed ${v.removed_sentences} sentence${v.removed_sentences > 1 ? "s" : ""} that couldn't be verified.</div>`;
  out.innerHTML = html + `<div class="verify ${ok ? "ok" : "warn"}"><b>${ok ? "✓ Fact-checked" : "⚠ Check before relying on this"}</b> · ${parts.join(" · ")}${extra}</div>`;
}

let aiBusy = false;
async function runBriefing(question) {
  if (aiBusy || !state.date) return;
  if (!state.api) {
    $("ai-output").innerHTML = '<p class="note">AI briefings need the CoralWatch server. Start it with <b>./run.sh</b> and open <b>http://127.0.0.1:8000</b>.</p>';
    return;
  }
  aiBusy = true;
  $("explain-btn").disabled = true;
  const out = $("ai-output");
  out.innerHTML = `<p class="note">Reading the numbers and searching the reef reports for ${fmtDate(state.date)}…</p>`;
  out.classList.add("streaming");
  let text = "", verification = null;
  try {
    const res = await fetch("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ date: state.date, zone: state.aiZone, question: question || null }),
    });
    if (!res.ok || !res.body) throw new Error((await res.json().catch(() => ({}))).detail || `Server replied ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let cut;
      while ((cut = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, cut);
        buf = buf.slice(cut + 2);
        const event = (chunk.match(/^event: (.*)$/m) || [])[1];
        const data = JSON.parse((chunk.match(/^data: (.*)$/m) || [])[1] || "{}");
        if (event === "context") renderSources(data.sources);
        else if (event === "delta") { text += data.text; out.innerHTML = renderMarkdown(text); }
        else if (event === "verify") { verification = data; }
        else if (event === "replace") { text = data.text; out.innerHTML = renderMarkdown(text); }
        else if (event === "done") {
          const badge = $("ai-mode");
          if (data.mode === "llm") { badge.textContent = `${data.model} · live`; badge.className = "ai-badge on"; }
          else { badge.textContent = "Offline briefing"; badge.className = "ai-badge off"; out.insertAdjacentHTML("beforeend", `<p class="note">${escapeHtml(data.reason)}</p>`); }
        } else if (event === "error") throw new Error(data.message);
      }
    }
  } catch (e) {
    out.insertAdjacentHTML("beforeend", `<p class="note">Something went wrong: ${escapeHtml(e.message)}</p>`);
  } finally {
    out.classList.remove("streaming");
    if (verification) {
      const notes = [...out.querySelectorAll(".note")].map((n) => n.outerHTML).join("");
      renderVerification(out, text, verification);
      out.insertAdjacentHTML("beforeend", notes);
    }
    aiBusy = false;
    $("explain-btn").disabled = false;
  }
}

function initAI() {
  if (!$("ai-panel")) return;
  document.querySelectorAll(".seg-btn").forEach((b) => b.addEventListener("click", () => {
    state.aiZone = b.dataset.zone;
    document.querySelectorAll(".seg-btn").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  }));
  $("explain-btn").addEventListener("click", () => runBriefing(null));
  $("ask-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("ask-input").value.trim();
    if (q) runBriefing(q);
  });
}

// ── Server detection + real-time data ───────────────────
async function connectServer() {
  const badge = $("ai-mode");
  if (!location.protocol.startsWith("http")) {
    if (badge) { badge.textContent = "Server not running"; badge.className = "ai-badge off"; }
    disableForecastLayer();
    return;
  }
  try {
    const health = await (await fetch("/api/health")).json();
    state.api = true;
    state.forecastOK = !!health.forecast_model;
    if (!state.forecastOK) disableForecastLayer();
    if (badge) {
      badge.textContent = !health.llm.configured ? "Offline mode"
        : health.llm.provider === "local" ? `${health.llm.model} · free, local` : `${health.llm.model} ready`;
      badge.className = `ai-badge ${health.llm.configured ? "on" : "off"}`;
    }
    if (state.forecastOK && state.date) loadForecast(state.date);
  } catch {
    if (badge) { badge.textContent = "Server not running"; badge.className = "ai-badge off"; }
    disableForecastLayer();
    return;
  }
  // Today's NOAA data, live
  try {
    const res = await fetch("/api/grid/latest");
    if (!res.ok) throw new Error(res.status);
    const g = await res.json();
    state.grids[g.date] = g;
    if (g.source === "live") { state.latest = g.date; state.live = true; }
    renderChips();
    if (window.heroStatus) window.heroStatus(g, g.source === "live");
  } catch (e) {
    console.warn("Live NOAA data unavailable, using saved snapshot", e);
  }
}

function disableForecastLayer() {
  const opt = document.querySelector('#layer-select option[value="forecast"]');
  if (opt) { opt.disabled = true; opt.textContent += " - needs the server"; }
}

initMap();
initControls();
initAI();
if (map) {
  if (state.grids[SHOWCASE_DATE]) loadDate(SHOWCASE_DATE);
  else if (state.latest) loadDate(state.latest);
}
connectServer();
