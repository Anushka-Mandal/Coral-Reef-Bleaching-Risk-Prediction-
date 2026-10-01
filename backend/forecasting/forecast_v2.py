"""Live 1-14 day forecast with the physics-guided ensemble (forecast/models/v2).

For an issue day t it gathers, with caching:
  * NOAA SST anomaly, HotSpot, DHW, SST for t-111..t  (84 days for the DHW drop-off,
    plus 4 weeks of model input) from PacIOOS
  * weather at 74 points for t-40..t+14: Open-Meteo forecast API for recent dates
    (a real forecast for the days ahead), the ERA5 archive for past dates
  * ENSO and MJO indices
then runs every seed of the selected model and averages the probabilities.
"""

import datetime as dt
import json
import logging
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "forecast" / "2_model"))

from coralwatch import geo, noaa, stats, weather  # noqa: E402
from paths import MODELS_V2 as MODEL_DIR, RESULTS_DAILY, RESULTS_FINAL  # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
log = logging.getLogger("coralwatch.forecast_v2")
_lock = threading.Lock()


class LiveCube:
    """The parts of forecast.dataset.Cube that PhysNet's Builder uses."""

    def __init__(self, dates, lats, lons, x, valid):
        self.dates, self.lats, self.lons, self.valid = list(dates), np.asarray(lats), np.asarray(lons), valid
        self.x = x.astype(np.float16)
        self.x[:, :, ~valid] = np.nan
        self.index = {d: i for i, d in enumerate(self.dates)}
        self.baa = geo.baa_from(self.x[:, 1].astype(np.float32), self.x[:, 2].astype(np.float32)).astype(np.float32)


@lru_cache(maxsize=1)
def meta():
    f = MODEL_DIR / "meta.json"
    if not f.exists():
        return None
    m = json.loads(f.read_text())
    m["valid"] = np.array(m["valid"], bool)
    return m


def available():
    m = meta()
    return bool(m) and all((MODEL_DIR / f"{k}.pt").exists() for k in m["ensemble"])


@lru_cache(maxsize=1)
def skill():
    """Measured test-season skill of the live forecast, for the UI and the LLM."""
    m = meta() or {}
    max7 = m.get("target") == "max7"
    folder = RESULTS_FINAL if max7 else RESULTS_DAILY
    res = folder / "results.json"
    if not res.exists():
        return None
    r = json.loads(res.read_text())
    sel = r["selected"]
    hyb_f = folder / "hybrid.json"
    hyb = json.loads(hyb_f.read_text())["leads"] if hyb_f.exists() else {}
    pick = lambda n, k: {x: round(r["leads"][str(k)][n][x][0], 3) for x in ("accuracy", "macro_f1", "alert_f1", "alert_recall")}
    out = {"model": sel, "target": r.get("target"), "test_seasons": r["test_seasons"],
           "model_test_7d": pick(sel, 7), "persistence_test_7d": pick("Persistence", 7),
           "hybrid_accuracy_by_lead": {k: round(v["test"]["accuracy"][0], 3) for k, v in hyb.items()}}
    real = folder / "real_forecast_check.json"
    if real.exists():
        rr = json.loads(real.read_text())["results"]
        out["with_real_weather_forecasts_7d"] = {
            "model_accuracy": round(rr["Forecast-weather model + real archived forecasts"]["accuracy"][0], 3),
            "persistence_accuracy": round(rr["Persistence"]["accuracy"][0], 3),
            "period": "Jan-Apr peaks of 2024 and 2025"}
    return out


def physics_levels(cube, t, max7):
    """Physics persistence for leads 1..14: HotSpot stays at today's value, DHW follows NOAA's
    formula (today's DHW + new HotSpots >= 1 - the known days leaving the 84-day window).
    For the 7-day-maximum target, each lead takes the highest level over its 7-day window,
    including the days already observed."""
    from physnet import LEADS
    day = cube.dates[t]
    hs = cube.x[t, 1].astype(np.float32)
    hs_plus = np.where(hs >= 1, hs, 0)
    drop = np.zeros_like(hs)
    daily = []
    for j in range(1, LEADS + 1):
        i = cube.index.get(day - dt.timedelta(days=84 - j))
        if i is not None:
            old = cube.x[i, 1].astype(np.float32)
            drop = drop + np.nan_to_num(np.where(old >= 1, old, 0)) / 7
        dhw_j = np.maximum(0, cube.x[t, 2].astype(np.float32) + j * hs_plus / 7 - drop)
        daily.append(geo.baa_from(hs, dhw_j))
    daily = np.stack(daily)                                              # [LEADS, H, W]
    if not max7:
        return daily
    out = np.empty_like(daily)
    for k in range(1, LEADS + 1):
        parts = [daily[j] for j in range(max(0, k - 7), k)]
        parts += [cube.baa[cube.index[day - dt.timedelta(days=j)]] for j in range(0, 7 - k)
                  if day - dt.timedelta(days=j) in cube.index]
        with np.errstate(all="ignore"):
            out[k - 1] = np.nanmax(np.stack(parts), axis=0)
    return out


_indices = {}


def _indices_for(dates):
    today = dt.date.today()
    if _indices.get("day") != today:
        _indices.update(day=today, nino=weather.fetch_nino34(), mjo=weather.fetch_mjo())
    return weather.daily_series(_indices["nino"], dates), weather.daily_series(_indices["mjo"], dates)


def _weather(start, end):
    """Weather for start..end at the 74 points: forecast API near today, archive for the past."""
    pts = weather.weather_points()
    recent = start >= dt.date.today() - dt.timedelta(days=85)
    f = CACHE / f"wx_{start}_{end}_{'fc' if recent else 'era5'}.npz"
    if f.exists() and not (recent and dt.date.fromtimestamp(f.stat().st_mtime) != dt.date.today()):
        z = np.load(f)
        return [dt.date.fromisoformat(s) for s in z["dates"]], pts, z["vals"], "forecast" if recent else "reanalysis"
    if recent:
        dates, vals = weather.fetch_points(pts, start, end, archive=False, forecast_days=16)
    else:
        dates, vals = weather.fetch_points(pts, start, min(end, dt.date.today() - dt.timedelta(days=6)))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, dates=np.array([d.isoformat() for d in dates]), vals=vals)
    return dates, pts, vals, "forecast" if recent else "reanalysis"


def forecast(issue=None):
    from physnet import Drivers, LEADS, PhysNet

    m = meta()
    if not m:
        raise RuntimeError("No v2 model - run forecast/train_v2.py")
    with _lock:
        if issue is None:
            issue = noaa.latest_model_date()
        dates, lats, lons, x = noaa.fetch_days_cached(issue - dt.timedelta(days=111), issue, CACHE / "days")
        if not dates or dates[-1] != issue:
            raise RuntimeError(f"NOAA data for {issue} isn't available yet")
        if not (np.allclose(lats, m["lats"]) and np.allclose(lons, m["lons"])):
            raise RuntimeError("NOAA grid differs from the model's training grid")
        cube = LiveCube(dates, lats, lons, x, m["valid"])
        max7 = m.get("target") == "max7"
        members, label = m["ensemble"], m["label"]
        try:
            wd, pts, wv, wx_kind = _weather(issue - dt.timedelta(days=40), issue + dt.timedelta(days=LEADS))
            nino, mjo = _indices_for(wd)
            drivers = Drivers.from_arrays(wd, pts, wv, nino, mjo, cube.lats, cube.lons)
        except Exception as e:
            # Weather service down: the network needs the weather forecast, so fall back to
            # physics persistence for every lead (needs only NOAA data).
            log.warning("Weather unavailable (%s) - using physics persistence for all leads", e)
            members, label, wx_kind, drivers = [], m.get("fallback_label", "Physics persistence"), "none", None
        t = cube.index[issue]
        probs = None
        for key in members:
            model = PhysNet.load(MODEL_DIR / f"{key}.pt", cube, drivers)
            p, _ = model.predict_all([t])
            probs = p[0].astype(np.float32) if probs is None else probs + p[0].astype(np.float32)

    valid = cube.valid
    phys = physics_levels(cube, t, max7)
    if probs is not None:
        probs /= len(members)                                        # [LEADS, 5, H, W]
        levels = probs.argmax(1).astype(float)
        p_alert = probs[:, 3] + probs[:, 4]
        # Hybrid: for the short leads chosen on validation, physics persistence beats the network.
        physics_leads = list(m.get("hybrid_physics_leads", []))
    else:
        levels, p_alert = phys.copy(), (phys >= 3).astype(np.float32)
        physics_leads = list(range(1, LEADS + 1))
    for k in physics_leads:
        levels[k - 1] = phys[k - 1]
        p_alert[k - 1] = (phys[k - 1] >= 3).astype(np.float32)
    levels[:, ~valid] = np.nan
    p_alert[:, ~valid] = np.nan
    now = cube.x[t].astype(np.float32)
    week_ago = cube.x[cube.index.get(issue - dt.timedelta(days=6), t)].astype(np.float32)
    observed = stats.zone_stats(lats, lons, cube.baa[t], dhw=now[2], ssta=now[0], hotspot=now[1])
    rows = geo.zone_rows(lats)
    for item in observed:
        mm = valid & rows[item["zone"]][:, None]
        item["dhw_change_7d"] = round(float(np.nanmean(now[2][mm] - week_ago[2][mm])), 2)
        item["ssta_change_7d"] = round(float(np.nanmean(now[0][mm] - week_ago[0][mm])), 2)
    by_lead = {}
    for k in (1, 3, 7, 14):
        zs = stats.zone_stats(lats, lons, levels[k - 1])
        for z in zs:
            mm = valid & rows[z["zone"]][:, None]
            z["mean_alert_probability"] = round(float(np.nanmean(p_alert[k - 1][mm])), 3)
        by_lead[k] = zs

    season = lambda d: d.year + 1 if d.month >= 7 else d.year
    to_list = lambda a, nd: [None if np.isnan(v) else (int(v) if nd == 0 else round(float(v), nd)) for v in a.ravel()]
    target = issue + dt.timedelta(days=7)
    return {
        "issue_date": issue.isoformat(),
        "target_date": target.isoformat(),
        "model": label,
        "target": "max7" if max7 else "daily",
        "target_description": ("NOAA 7-day maximum alert: the highest daily alert level over the 7 days "
                               "ending on the forecast day" if max7 else "NOAA daily alert level"),
        "physics_persistence_leads": physics_leads,
        "weather": wx_kind,
        "in_training_data": season(target) <= 2021,
        "in_validation_data": season(target) in (2022, 2023),
        "off_season": not (target.month >= 11 or target.month <= 4),
        "lats": [float(v) for v in lats], "lons": [float(v) for v in lons],
        "step": round(abs(float(lats[1] - lats[0])), 4),
        "leads": list(range(1, LEADS + 1)),
        "vars": {"forecast": to_list(levels[6], 0), "p_alert": to_list(p_alert[6], 2)},
        "by_lead_levels": {str(k): to_list(levels[k - 1], 0) for k in range(1, LEADS + 1)},
        "by_lead_p_alert": {str(k): to_list(p_alert[k - 1], 2) for k in range(1, LEADS + 1)},
        "observed_zones": observed,
        "forecast_zones": by_lead[7],
        "forecast_zones_by_lead": {str(k): v for k, v in by_lead.items()},
        "skill": skill(),
    }
