"""
download_weather.py
===================
Downloads the weather and climate drivers for 2017-09 .. 2025-12:
  * daily wind, cloud cover, shortwave radiation, air temperature (Open-Meteo archive,
    ERA5-based) at 74 points on a 0.75 deg grid around the Marine Park
  * ENSO (Nino 3.4) and MJO (RMM1, RMM2) indices

    python forecast/1_data_preparation/download_weather.py      # -> forecast/data/weather.npz

Resumable (one cache file per batch of points and year) and polite to the free API.
"""

import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]
from coralwatch import weather  # noqa: E402
from paths import WEATHER as OUT, WEATHER_CACHE as CACHE  # noqa: E402
START, END = dt.date(2017, 9, 1), dt.date(2025, 12, 31)
BATCH = 8


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    points = weather.weather_points()
    dates = [START + dt.timedelta(days=i) for i in range((END - START).days + 1)]
    index = {d: i for i, d in enumerate(dates)}
    vals = np.full((len(dates), len(points), len(weather.WEATHER_VARS)), np.nan, np.float32)
    jobs = [(b, y) for b in range(0, len(points), BATCH) for y in range(START.year, END.year + 1)]
    print(f"{len(points)} points, {len(jobs)} requests", flush=True)
    for n, (b, year) in enumerate(jobs, 1):
        f = CACHE / f"b{b:03d}_{year}.npz"
        pts = points[b:b + BATCH]
        if not f.exists():
            s, e = max(START, dt.date(year, 1, 1)), min(END, dt.date(year, 12, 31))
            for attempt in range(12):
                try:
                    d, v = weather.fetch_points(pts, s, e)
                    np.savez(f, dates=np.array([x.isoformat() for x in d]), vals=v)
                    break
                except Exception as err:
                    wait = min(900, 60 * (attempt + 1))   # the free API limits calls per minute and per hour
                    print(f"  batch {b} {year}: {err} - retrying in {wait}s", flush=True)
                    time.sleep(wait)
            else:
                raise SystemExit("Open-Meteo unavailable - run again later to resume")
            time.sleep(2.0)
        z = np.load(f)
        for d, row in zip(z["dates"], z["vals"]):
            i = index.get(dt.date.fromisoformat(str(d)))
            if i is not None:
                vals[i, b:b + len(pts)] = row
        if n % 10 == 0:
            print(f"  {n}/{len(jobs)}", flush=True)

    print("Climate indices ...", flush=True)
    nino = weather.daily_series(weather.fetch_nino34(), dates)
    mjo = weather.daily_series(weather.fetch_mjo(), dates)
    np.savez_compressed(OUT, dates=np.array([d.isoformat() for d in dates]), points=np.array(points),
                        vals=vals, names=np.array(weather.WEATHER_NAMES), nino34=nino, mjo=mjo)
    print(f"Saved {OUT}: {len(dates)} days, {len(points)} points, "
          f"{np.isnan(vals).mean():.1%} missing", flush=True)


if __name__ == "__main__":
    main()
