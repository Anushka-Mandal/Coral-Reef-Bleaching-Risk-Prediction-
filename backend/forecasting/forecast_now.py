"""
forecast_now.py
===============
Run the live forecast from the terminal - the same pipeline the website uses.

    .venv/bin/python -m backend.forecasting.forecast_now                   # from the latest NOAA day
    .venv/bin/python -m backend.forecasting.forecast_now --date 2024-03-03 # issued on a past day
    .venv/bin/python -m backend.forecasting.forecast_now --save today.json # also save the full grid

What it feeds the model automatically (no manual input needed):
  1. NOAA Coral Reef Watch 5 km daily data (SST anomaly, HotSpot, DHW, SST) for the 112 days
     up to the issue day, from NOAA's PacIOOS server (cached per day; only new days download)
  2. Open-Meteo weather at 74 points: the past 40 days + the next 16 days of real forecast
     (for past dates: ERA5 reanalysis)
  3. ENSO (NOAA CPC Nino 3.4) and MJO (Bureau of Meteorology RMM) indices
"""

import argparse
import datetime as dt
import json
import time

from . import forecast_v2


def main():
    ap = argparse.ArgumentParser(description="Run the CoralWatch forecast on real-time data")
    ap.add_argument("--date", help="issue date YYYY-MM-DD (default: latest NOAA day)")
    ap.add_argument("--save", help="write the full forecast (all cells, all leads) to this JSON file")
    args = ap.parse_args()

    t0 = time.time()
    f = forecast_v2.forecast(dt.date.fromisoformat(args.date) if args.date else None)
    print(f"\nCoralWatch forecast issued {f['issue_date']}  ({time.time() - t0:.0f}s)")
    print(f"Model:    {f['model']}")
    print(f"Target:   {f['target_description']}")
    print(f"Weather:  {f['weather']}  |  physics persistence for leads {f['physics_persistence_leads']}")
    if f["off_season"]:
        print("Note:     outside the Nov-Apr bleaching season - risk is normally low")
    print("\nToday (observed):")
    for z in f["observed_zones"]:
        print(f"  {z['name']:13s} {z['reefs_at_alert']:5d} of {z['reefs_total']} reefs at Alert Level 1+ | "
              f"worst {z['worst_level']} | max DHW {z.get('dhw_max', 0):.1f}")
    print("\nForecast (reefs at Alert Level 1+, average chance of Alert):")
    for k, zones in f["forecast_zones_by_lead"].items():
        day = dt.date.fromisoformat(f["issue_date"]) + dt.timedelta(days=int(k))
        cells = "  ".join(f"{z['name']}: {z['reefs_at_alert']:4d} ({z['mean_alert_probability']:.0%})" for z in zones)
        print(f"  +{int(k):2d} days ({day}):  {cells}")
    if f.get("skill"):
        s = f["skill"]
        print(f"\nMeasured skill (test {s['test_seasons']}): 7-day accuracy {s['model_test_7d']['accuracy']:.1%} "
              f"vs {s['persistence_test_7d']['accuracy']:.1%} for 'same as today'")
    if args.save:
        with open(args.save, "w") as fh:
            json.dump(f, fh)
        print(f"\nSaved the full forecast to {args.save}")


if __name__ == "__main__":
    main()
