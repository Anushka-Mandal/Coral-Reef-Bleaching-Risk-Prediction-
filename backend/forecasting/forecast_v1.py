"""Runs the trained 7-day forecast on live NOAA data.

For an issue day t it downloads t-6..t from NOAA CoastWatch (cached on disk), feeds
the chosen model from forecast/models/, and returns the forecast grid plus zone
statistics for the observed day and the forecast day.
"""

import datetime as dt
import json
import logging
import pickle
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "forecast" / "v1_first_model"), str(ROOT / "forecast" / "2_model")]

from coralwatch import geo, noaa, stats  # noqa: E402
from paths import MODELS_V1 as MODEL_DIR, RESULTS_V1  # noqa: E402

RESULTS = RESULTS_V1 / "results.json"
CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"
log = logging.getLogger("coralwatch.forecast")
_lock = threading.Lock()


class LiveCube:
    """The subset of forecast.dataset.Cube that the models use, built from live days."""

    def __init__(self, dates, lats, lons, x, valid):
        self.dates, self.lats, self.lons, self.valid = list(dates), lats, lons, valid
        self.x = x.copy()
        self.x[:, :, ~valid] = np.nan
        self.index = {d: i for i, d in enumerate(self.dates)}
        self.baa = geo.baa_from(self.x[:, 1], self.x[:, 2])

    def window(self, t):
        day = self.dates[t]
        return np.stack([self.x[self.index[day - dt.timedelta(days=k)]] for k in range(6, -1, -1)])


@lru_cache(maxsize=1)
def load_model():
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        return None, None
    meta = json.loads(meta_path.read_text())
    path = MODEL_DIR / meta["file"]
    if meta["file"].endswith(".pt"):
        from models import CNNConvLSTM
        model = CNNConvLSTM.load(path)
    else:
        import models  # noqa: F401  (needed so pickle can find the class)
        with open(path, "rb") as f:
            model = pickle.load(f)
    meta["valid"] = np.array(meta["valid"], bool)
    return model, meta


@lru_cache(maxsize=1)
def skill_summary():
    """Test-season skill of the chosen model vs persistence, for the UI and the LLM."""
    if not RESULTS.exists():
        return None
    r = json.loads(RESULTS.read_text())
    best = r["selected_model"]
    pick = lambda m: {k: round(r["models"][m]["test"][k], 3) for k in ("accuracy", "macro_f1", "alert_f1", "alert_recall", "change_accuracy")}
    return {"model": best, "test_seasons": r["splits"]["test"], "model_test": pick(best),
            "persistence_test": pick("Persistence")}


def _week(issue):
    """7 days of raw model inputs ending on `issue`, cached per day on disk."""
    f = CACHE / f"model_{issue.isoformat()}.npz"
    if f.exists():
        d = np.load(f)
        return [dt.date.fromisoformat(s) for s in d["dates"]], d["lats"], d["lons"], d["x"]
    dates, lats, lons, x = noaa.fetch_recent_model_days(issue, 7)
    if len(dates) == 7 and issue < dt.date.today() - dt.timedelta(days=1):
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(f, dates=np.array([d.isoformat() for d in dates]), lats=lats, lons=lons, x=x)
    return dates, lats, lons, x


def available():
    return load_model()[0] is not None


def forecast(issue=None):
    """Forecast from issue day `issue` (default: latest NOAA day) to issue + 7 days."""
    model, meta = load_model()
    if model is None:
        raise RuntimeError("No trained model - run forecast/train.py")
    with _lock:
        if issue is None:
            issue = noaa.latest_model_date()
        dates, lats, lons, x = _week(issue)
    if len(dates) < 7 or dates[-1] != issue:
        raise RuntimeError(f"NOAA data incomplete for the week ending {issue}")
    if not (np.allclose(lats, meta["lats"]) and np.allclose(lons, meta["lons"])):
        raise RuntimeError("NOAA grid differs from the model's training grid")

    cube = LiveCube(dates, lats, lons, x, meta["valid"])
    t = cube.index[issue]
    pred = model.predict(cube, t)
    p_alert = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(cube, t)
        p_alert = np.nansum(proba[..., 3:], axis=-1)
        p_alert[~cube.valid] = np.nan

    now = cube.x[t]
    week_ago = cube.x[cube.index[issue - dt.timedelta(days=6)]]
    observed = stats.zone_stats(lats, lons, cube.baa[t], dhw=now[2], ssta=now[0], hotspot=now[1])
    forecast_stats = stats.zone_stats(lats, lons, pred)
    rows = geo.zone_rows(lats)
    for item in observed:
        m = cube.valid & rows[item["zone"]][:, None]
        item["dhw_change_7d"] = round(float(np.nanmean(now[2][m] - week_ago[2][m])), 2)
        item["ssta_change_7d"] = round(float(np.nanmean(now[0][m] - week_ago[0][m])), 2)

    target = issue + dt.timedelta(days=meta["horizon"])
    season = target.year + 1 if target.month >= 7 else target.year
    to_list = lambda a, nd=0: [None if np.isnan(v) else (int(v) if nd == 0 else round(float(v), nd)) for v in a.ravel()]
    return {
        "issue_date": issue.isoformat(),
        "target_date": target.isoformat(),
        "model": meta["name"],
        "in_training_data": season <= 2021,
        "off_season": not (target.month >= 11 or target.month <= 4),
        "lats": [float(v) for v in lats], "lons": [float(v) for v in lons],
        "step": round(abs(float(lats[1] - lats[0])), 4),
        "vars": {"forecast": to_list(pred), **({"p_alert": to_list(p_alert, 2)} if p_alert is not None else {})},
        "observed_zones": observed,
        "forecast_zones": forecast_stats,
        "skill": skill_summary(),
    }
