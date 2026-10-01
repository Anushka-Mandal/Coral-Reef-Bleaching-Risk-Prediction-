/* CoralWatch - landing page: hero visual, particles, sections, dataset chart.
 * Uses BAA_LEVELS, GEO, STUDY, clipRing, reefCellIndex, fmtDate, $ from app.js. */

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

// ── Nav background on scroll ────────────────────────────
const nav = $("nav");
const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 30);
window.addEventListener("scroll", onScroll, { passive: true });
onScroll();

// ── Reveal sections as they scroll into view ────────────
const io = new IntersectionObserver((entries) => {
  entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("visible"); io.unobserve(e.target); } });
}, { threshold: 0.12 });
function observeReveals() { document.querySelectorAll(".reveal:not(.visible)").forEach((el) => io.observe(el)); }

// ── Bioluminescent plankton background ──────────────────
(function plankton() {
  const cv = $("plankton");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  let w, h, dots;
  const colors = ["94,234,212", "56,189,248", "125,211,252", "255,122,89"];
  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = cv.clientWidth; h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round((w * h) / 9000);
    dots = Array.from({ length: n }, () => ({
      x: Math.random() * w, y: Math.random() * h,
      r: Math.random() * 1.6 + 0.3,
      vx: (Math.random() - 0.5) * 0.15, vy: -Math.random() * 0.25 - 0.05,
      a: Math.random() * Math.PI * 2,
      c: colors[Math.random() < 0.08 ? 3 : Math.floor(Math.random() * 3)],
    }));
  }
  function frame(t) {
    ctx.clearRect(0, 0, w, h);
    for (const d of dots) {
      d.x += d.vx; d.y += d.vy;
      if (d.y < -5) { d.y = h + 5; d.x = Math.random() * w; }
      if (d.x < -5) d.x = w + 5; else if (d.x > w + 5) d.x = -5;
      const alpha = 0.25 + 0.45 * (0.5 + 0.5 * Math.sin(t / 900 + d.a));
      ctx.beginPath();
      ctx.fillStyle = `rgba(${d.c},${alpha})`;
      ctx.shadowColor = `rgba(${d.c},0.9)`;
      ctx.shadowBlur = 8;
      ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
      ctx.fill();
    }
    if (!reduceMotion) requestAnimationFrame(frame);
  }
  resize();
  window.addEventListener("resize", resize);
  requestAnimationFrame(frame);
})();

// ── Hero: glowing map of every reef in the study area ───
function heroMap() {
  const svg = $("hero-map");
  if (!svg || !GEO.boundary.length) return;
  const ring = clipRing(GEO.boundary, STUDY.north, STUDY.south);
  const lats = ring.map((p) => p[0]), lons = ring.map((p) => p[1]);
  const north = Math.max(...lats), south = Math.min(...lats), west = Math.min(...lons) - 0.2, east = Math.max(...lons) + 1.6;
  const kx = Math.cos((((north + south) / 2) * Math.PI) / 180), S = 48;
  const X = (lon) => ((lon - west) * kx * S).toFixed(1), Y = (lat) => ((north - lat) * S).toFixed(1);
  const W = (east - west) * kx * S, H = (north - south) * S;
  svg.setAttribute("viewBox", `-10 -10 ${(W + 20).toFixed(0)} ${(H + 20).toFixed(0)}`);

  const path = "M" + ring.map((p) => `${X(p[1])},${Y(p[0])}`).join("L") + "Z";
  const [x0, x1] = (() => { const s = parkSpanAt(STUDY.split); return s ? [X(s[0]), X(s[1])] : [0, 0]; })();

  // Colour reefs by alert level on the showcase date
  const showcase = window.SNAPSHOTS && (window.SNAPSHOTS[SHOWCASE_DATE] || window.SNAPSHOTS[window.SNAPSHOT_LATEST]);
  const idx = showcase ? reefCellIndex(showcase) : [];
  const reefs = GEO.reefs.map((r, k) => {
    const lvl = showcase && idx[k] >= 0 ? showcase.vars.baa[idx[k]] : 0;
    const delay = (1.2 + ((north - r.c[0]) / (north - south)) * 2.2).toFixed(2);
    return `<circle class="reef" cx="${X(r.c[1])}" cy="${Y(r.c[0])}" r="${lvl >= 3 ? 2.6 : 2}" fill="${BAA_LEVELS[lvl].color}" style="animation-delay:${delay}s"/>`;
  }).join("");

  svg.innerHTML = `
    <defs>
      <filter id="reef-glow" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="2.4" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
    </defs>
    <path class="coast" d="${path}"/>
    <line class="split" x1="${x0}" y1="${Y(STUDY.split)}" x2="${(+x1 + 70).toFixed(1)}" y2="${Y(STUDY.split)}"/>
    <text class="zone-text" x="${(+x1 + 18).toFixed(1)}" y="${(+Y(STUDY.split) - 12).toFixed(1)}">Northern GBR</text>
    <text class="zone-text" x="${(+x1 + 18).toFixed(1)}" y="${(+Y(STUDY.split) + 22).toFixed(1)}">Central GBR</text>
    <g filter="url(#reef-glow)">${reefs}</g>`;
  const coast = svg.querySelector(".coast");
  coast.style.setProperty("--len", Math.ceil(coast.getTotalLength()));

  if (showcase) {
    $("hero-caption").textContent = `All ${GEO.reefs.length.toLocaleString()} reefs in our study area, coloured by NOAA bleaching alert level on ${fmtDate(showcase.date)}.`;
  }
  $("hero-legend").innerHTML = BAA_LEVELS.map((l) => `<span><i style="background:${l.color};color:${l.color}"></i>${l.name}</span>`).join("");
}

// ── Hero status line + stats ────────────────────────────
function heroStatus(g = window.SNAPSHOTS && window.SNAPSHOTS[window.SNAPSHOT_LATEST], live = false) {
  if ($("stat-reefs") && GEO.reefs.length) $("stat-reefs").textContent = GEO.reefs.length.toLocaleString();
  if (!g) { $("hero-status").textContent = "open the live map"; return; }
  const stats = allStats(g);
  const alert = stats.reduce((a, { s }) => a + s.reefsAlert, 0);
  const worst = Math.max(...stats.map(({ s }) => s.worst));
  const lvl = BAA_LEVELS[worst];
  $("hero-status").innerHTML = alert
    ? `<b style="color:${lvl.color}">${alert.toLocaleString()} reefs at Alert Level 1+</b> · ${fmtDate(g.date)}`
    : `<b style="color:${lvl.color}">${lvl.name}</b> across the northern &amp; central reef · ${fmtDate(g.date)}`;
  if (live) $("hero-status").innerHTML += ' <span class="live-pill">LIVE</span>';
}
window.heroStatus = heroStatus;

// ── Alert level cards ───────────────────────────────────
function levels() {
  const el = $("levels-list");
  if (!el) return;
  el.innerHTML = BAA_LEVELS.map((l, i) => `
    <article class="level-card reveal" style="--c:${l.color}">
      <span class="level-num">${i}</span>
      <h3>${l.name}</h3>
      <p>${l.desc}</p>
    </article>`).join("");
}

// ── Dataset chart: days per alert level per year ────────
// Counts from CNN/baa_labels.csv (NOAA Bleaching Alert Area, daily, 2018-2025)
const YEAR_COUNTS = {
  2018: [138, 78, 31, 62, 56],
  2019: [177, 65, 56, 67, 0],
  2020: [104, 129, 55, 30, 44],
  2021: [100, 61, 25, 42, 137],
  2022: [0, 90, 30, 16, 227],
  2023: [126, 65, 30, 0, 144],
  2024: [86, 116, 25, 32, 104],
  2025: [39, 121, 27, 21, 157],
};

function yearChart() {
  const box = $("year-chart");
  if (!box) return;
  const years = Object.keys(YEAR_COUNTS);
  const W = 1000, H = 360, m = { t: 36, r: 8, b: 34, l: 40 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b, maxY = 366;
  const band = iw / years.length, bw = Math.min(64, band * 0.56);
  const y = (v) => m.t + ih - (v / maxY) * ih;
  const order = [4, 3, 2, 1, 0]; // most severe sits on the baseline

  let bars = "";
  years.forEach((yr, i) => {
    const cx = m.l + band * i + band / 2;
    let acc = 0, segs = "";
    order.forEach((lvl, s) => {
      const v = YEAR_COUNTS[yr][lvl];
      if (!v) return;
      const y0 = y(acc), y1 = y(acc + v);
      const top = s === order.length - 1 || order.slice(s + 1).every((l) => !YEAR_COUNTS[yr][l]);
      const hgt = Math.max(0, y0 - y1 - 2); // 2px gap between segments
      segs += `<rect class="bar-seg" x="${(cx - bw / 2).toFixed(1)}" y="${y1.toFixed(1)}" width="${bw.toFixed(1)}" height="${hgt.toFixed(1)}" rx="${top ? 4 : 1.5}" fill="${BAA_LEVELS[lvl].color}"/>`;
      acc += v;
    });
    bars += `<g class="bar-group" data-year="${yr}">
      <rect x="${(cx - band / 2).toFixed(1)}" y="${m.t}" width="${band.toFixed(1)}" height="${ih}" fill="transparent"/>
      ${segs}
      <text class="year-label" x="${cx.toFixed(1)}" y="${H - 10}" text-anchor="middle">${yr}</text>
    </g>`;
  });

  const ticks = [0, 100, 200, 300].map((v) =>
    `<g class="grid"><line x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/></g><g class="axis"><text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text></g>`).join("");
  const testX = m.l + band * 6 + 4;

  box.innerHTML = `<svg viewBox="0 0 ${W} ${H}">
    <rect class="test-band" x="${testX}" y="${m.t - 30}" width="${band * 2 - 8}" height="${ih + 30}" rx="10"/>
    <text class="band-label" x="${testX + band - 4}" y="${m.t - 14}" text-anchor="middle">Test years</text>
    ${ticks}${bars}
    <g class="axis"><text x="${m.l - 8}" y="${m.t - 12}" text-anchor="end">days</text></g>
  </svg>`;

  // Hover tooltip
  const tip = document.createElement("div");
  tip.className = "tooltip"; tip.hidden = true;
  box.appendChild(tip);
  box.querySelectorAll(".bar-group").forEach((gEl) => {
    gEl.addEventListener("mouseenter", () => {
      const yr = gEl.dataset.year, c = YEAR_COUNTS[yr], total = c.reduce((a, b) => a + b, 0);
      tip.innerHTML = `<h5>${yr}</h5>` + [4, 3, 2, 1, 0].map((l) =>
        `<div><span><i style="background:${BAA_LEVELS[l].color}"></i>${BAA_LEVELS[l].name}</span><b>${c[l]}</b></div>`).join("") +
        `<div style="margin-top:4px"><span>Total days</span><b>${total}</b></div>`;
      tip.hidden = false;
      box.classList.add("hovering");
    });
    gEl.addEventListener("mousemove", (e) => {
      const r = box.getBoundingClientRect();
      let left = e.clientX - r.left + 16;
      if (left + 190 > r.width) left = e.clientX - r.left - 200;
      tip.style.left = `${left}px`; tip.style.top = `${e.clientY - r.top - 20}px`;
    });
    gEl.addEventListener("mouseleave", () => { tip.hidden = true; box.classList.remove("hovering"); });
  });

  $("chart-legend").innerHTML = [4, 3, 2, 1, 0].map((l) => `<span><i style="background:${BAA_LEVELS[l].color}"></i>${BAA_LEVELS[l].name}</span>`).join("");
  $("year-table").innerHTML = `<table><thead><tr><th>Year</th>${BAA_LEVELS.map((l) => `<th>${l.name}</th>`).join("")}</tr></thead><tbody>${
    years.map((yr) => `<tr><td>${yr}</td>${YEAR_COUNTS[yr].map((v) => `<td>${v}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}

heroMap();
heroStatus();
levels();
yearChart();
observeReveals();
