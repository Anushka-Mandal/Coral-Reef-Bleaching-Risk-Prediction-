"""
eval_real_forecasts.py
======================
Honest check of the weather-forecast model: in training and in evaluate_v2.py the
"next two weeks" of weather are the weather that actually happened (ERA5). A live
system only has a forecast. Here the Jan-Apr peak of the 2024 and 2025 test seasons
is re-scored using the weather forecasts that were really issued on each day
(Open-Meteo archive of past forecasts, lead 1-7 days; days 8-14 use the week-1 mean
because longer-lead archives don't exist).

    python forecast/eval_real_forecasts.py                   # -> results/v2/real_forecast_check.json
    python forecast/eval_real_forecasts.py --download-only   # just fetch the archived forecasts

Compared on exactly the same forecast days:
  persistence, the observed-weather-only model, and the forecast-weather model fed
  (a) observed weather and (b) real archived forecasts.
"""

import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]

from coralwatch import weather  # noqa: E402
from dataset import Cube  # noqa: E402
from evaluate_v2 import boot, day_cm, paired_p  # noqa: E402
from physnet import Drivers, PhysNet  # noqa: E402

from paths import MODELS_V2 as MODELS, PREVRUNS_CACHE as CACHE, RESULTS_DAILY, RESULTS_FINAL, WEATHER  # noqa: E402
PERIODS = [(dt.date(2023, 12, 20), dt.date(2024, 5, 8)), (dt.date(2024, 12, 20), dt.date(2025, 5, 8))]
SCALE_SHIFT = np.array([20, 50, 20, 27], np.float32)
SCALE_DIV = np.array([10, 40, 8, 3], np.float32)


def download(points):
    CACHE.mkdir(parents=True, exist_ok=True)
    per_lead = {j: {} for j in range(1, 8)}
    for (s, e) in PERIODS:
        for b in range(0, len(points), 8):
            f = CACHE / f"{s}_{b:03d}.npz"
            if not f.exists():
                for attempt in range(12):
                    try:
                        r = weather.fetch_previous_runs(points[b:b + 8], s, e)
                        np.savez(f, dates=np.array([d.isoformat() for d in r[1][0]]),
                                 **{f"lead{j}": r[j][1] for j in r})
                        break
                    except Exception as err:
                        wait = min(900, 60 * (attempt + 1))
                        print(f"  {s} batch {b}: {err} - retrying in {wait}s", flush=True)
                        time.sleep(wait)
                else:
                    raise SystemExit("Open-Meteo unavailable - run again later")
                time.sleep(3)
            z = np.load(f)
            for j in range(1, 8):
                for d, row in zip(z["dates"], z[f"lead{j}"]):
                    per_lead[j].setdefault(dt.date.fromisoformat(str(d)), np.full((len(points), 4), np.nan, np.float32))[b:b + 8] = row
        print(f"  forecasts for {s}..{e} ready", flush=True)
    return per_lead


def main():
    cube = Cube()
    drivers = Drivers(WEATHER, cube.lats, cube.lons)
    points = weather.weather_points()
    per_lead = download(points)
    log = json.loads((MODELS / "log.json").read_text())

    days = [t for t in cube.samples("test")
            if any(s <= cube.dates[t] + dt.timedelta(days=7) <= e - dt.timedelta(days=8) and (cube.dates[t] + dt.timedelta(days=7)).month <= 4
                   for s, e in PERIODS)]
    # future weather from forecasts issued on day t: day t+j comes from the lead-j forecast
    future = {}
    for t in days:
        rows = []
        for j in range(1, 8):
            v = per_lead[j].get(cube.dates[t] + dt.timedelta(days=j))
            if v is not None:
                rows.append(v)
        if len(rows) < 5:
            continue
        wk1 = np.nanmean(np.stack(rows), 0)                              # [P, 4]
        wk1 = (wk1 - SCALE_SHIFT) / SCALE_DIV
        grid = (drivers.M @ np.nan_to_num(wk1)).T.reshape((4,) + drivers.shape)
        future[t] = np.concatenate([grid, grid]).astype(np.float32)      # week 2 := week-1 forecast
    days = [t for t in days if t in future]
    print(f"{len(days)} forecast days with archived weather forecasts", flush=True)

    max7 = "--target" in sys.argv and sys.argv[sys.argv.index("--target") + 1] == "max7"
    if max7:   # 7-day-maximum alert models; "past" slot = the one trained with forecast error
        ens = [k for k in log if k.startswith("max7_s")]
        past = [k for k in log if k.startswith("max7_noise_s")][:1]
    else:
        ens = [k for k in log if k.startswith("full_s")]
        past = [k for k in log if k.startswith("past_wx_s")][:1]
    models = {k: PhysNet.load(MODELS / f"{k}.pt", cube, drivers) for k in ens + past}
    valid = cube.valid
    tgt = cube.baa7 if max7 else cube.baa
    target = lambda t: tgt[cube.index[cube.dates[t] + dt.timedelta(days=7)]][valid]
    alt = "Trained with forecast error + real archived forecasts (seed 0)" if max7 else "Observed weather only (seed 0)"
    cms = {n: [] for n in ("Persistence", alt, "Forecast-weather model + observed future weather",
                           "Forecast-weather model + real archived forecasts")}
    for s in range(0, len(days), 16):
        chunk = days[s:s + 16]
        p_obs = np.mean([models[k].predict_all(chunk)[0][:, 6].astype(np.float32) for k in ens], 0)
        p_fc = np.mean([models[k].predict_all(chunk, future_weather={t: future[t] for t in chunk})[0][:, 6].astype(np.float32) for k in ens], 0)
        if past:
            fw = {t: future[t] for t in chunk} if max7 else None
            p_past = models[past[0]].predict_all(chunk, future_weather=fw)[0][:, 6].astype(np.float32)
        else:
            p_past = None
        for i, t in enumerate(chunk):
            y = target(t)
            cms["Persistence"].append(day_cm(y, (cube.baa7[t] if max7 else cube.baa[t])[valid]))
            cms["Forecast-weather model + observed future weather"].append(day_cm(y, p_obs[i].argmax(0)[valid]))
            cms["Forecast-weather model + real archived forecasts"].append(day_cm(y, p_fc[i].argmax(0)[valid]))
            if p_past is not None:
                cms[alt].append(day_cm(y, p_past[i].argmax(0)[valid]))
    cms = {n: np.array(v) for n, v in cms.items() if v}
    out = {"forecast_days": len(days), "periods": [f"{s}..{e}" for s, e in PERIODS], "lead_days": 7,
           "results": {n: boot(v)[0] for n, v in cms.items()},
           "p_vs_persistence": {n: {k: paired_p(v, cms["Persistence"], k) for k in ("accuracy", "macro_f1", "alert_f1")}
                                for n, v in cms.items() if n != "Persistence"}}
    out["target"] = "max7" if max7 else "daily"
    ((RESULTS_FINAL if max7 else RESULTS_DAILY) / "real_forecast_check.json").write_text(json.dumps(out, indent=1))
    for n, m in out["results"].items():
        print(f"{n:52s} acc {m['accuracy'][0]:.3f}  macro-F1 {m['macro_f1'][0]:.3f}  alert-F1 {m['alert_f1'][0]:.3f}")


if __name__ == "__main__":
    if "--download-only" in sys.argv:
        download(weather.weather_points())
    else:
        main()
