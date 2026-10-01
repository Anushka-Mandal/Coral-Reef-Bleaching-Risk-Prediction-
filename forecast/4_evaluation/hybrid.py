"""
hybrid.py
=========
Picks, for each lead time 1-14 days, the better of two forecasters using the
validation seasons only (never the test seasons):
    * physics persistence - today's HotSpot held steady, DHW by NOAA's formula
    * the PhysNet ensemble
The PhysNet ensemble was tuned for the 7-day lead, and at 1 day it is worse than
simply carrying today's state forward, so the operational forecast is a hybrid.

    python forecast/hybrid.py    # -> results/v2/hybrid.json (choice per lead + test scores)
"""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(Path(__file__).resolve().parent), str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]

from dataset import SPLITS, Cube, is_scored, season_of  # noqa: E402
from evaluate_v2 import day_cm, metrics_from_cm, physics_persistence  # noqa: E402
from physnet import LEADS, Drivers, PhysNet  # noqa: E402

from paths import MODELS_V2 as MODELS, RESULTS_DAILY, RESULTS_FINAL, WEATHER  # noqa: E402


def score(cm):
    m = metrics_from_cm(cm)
    return 0.5 * (m["accuracy"] + m["macro_f1"])


def main():
    cube = Cube()
    drivers = Drivers(WEATHER, cube.lats, cube.lons)
    meta = json.loads((MODELS / "meta.json").read_text())
    max7 = "--target" in sys.argv and sys.argv[sys.argv.index("--target") + 1] == "max7"
    log_keys = json.loads((MODELS / "log.json").read_text())
    keys = sorted(k for k in log_keys if (k.startswith("max7_s") if max7 else k.startswith("full_s")))
    models = [PhysNet.load(MODELS / f"{k}.pt", cube, drivers) for k in keys]
    tgt = cube.baa7 if max7 else cube.baa
    from evaluate_v2 import window_max
    val = cube.samples("val")
    valid = cube.valid
    cm = {"physics": np.zeros((LEADS, 5, 5), np.int64), "ensemble": np.zeros((LEADS, 5, 5), np.int64)}
    b = models[0].builder
    for s in range(0, len(val), 24):
        chunk = val[s:s + 24]
        probs = np.mean([m.predict_all(chunk)[0].astype(np.float32) for m in models], 0)
        for i, t in enumerate(chunk):
            for k in range(1, LEADS + 1):
                day = cube.dates[t] + dt.timedelta(days=k)
                j = cube.index.get(day)
                if j is None or not is_scored(day) or season_of(day) not in SPLITS["val"]:
                    continue
                y = tgt[j][valid]
                phys = (window_max(cube, t, k, lambda q: physics_persistence(b, cube, t, q)) if max7
                        else physics_persistence(b, cube, t, k))
                cm["physics"][k - 1] += day_cm(y, phys[valid])
                cm["ensemble"][k - 1] += day_cm(y, probs[i, k - 1].argmax(0)[valid])
        print(f"  validation {min(s + 24, len(val))}/{len(val)}", flush=True)

    out_dir = RESULTS_FINAL if max7 else RESULTS_DAILY
    test = json.loads((out_dir / "results.json").read_text())
    ens_name = test["selected"]
    choice, rows = {}, {}
    for k in range(1, LEADS + 1):
        sp, se = score(cm["physics"][k - 1]), score(cm["ensemble"][k - 1])
        choice[k] = "physics" if sp > se else "ensemble"
        name = "Physics persistence" if choice[k] == "physics" else ens_name
        rows[k] = {"choice": choice[k], "val_score_physics": round(sp, 4), "val_score_ensemble": round(se, 4),
                   "test": {m: test["leads"][str(k)][name][m] for m in ("accuracy", "macro_f1", "alert_f1", "within_one")},
                   "test_persistence": {m: test["leads"][str(k)]["Persistence"][m] for m in ("accuracy", "macro_f1", "alert_f1")}}
        print(f"lead {k:2d}: val physics {sp:.3f} vs ensemble {se:.3f} -> {choice[k]:8s} | test acc "
              f"{rows[k]['test']['accuracy'][0]:.3f} (persistence {rows[k]['test_persistence']['accuracy'][0]:.3f})")
    (out_dir / "hybrid.json").write_text(json.dumps({"selected_on": "validation seasons", "leads": rows}, indent=1))
    # store the choice in the config of the model family it belongs to (meta.json = live model)
    live_is_max7 = meta.get("target") == "max7"
    cfg_file = MODELS / ("meta.json" if live_is_max7 == max7 else ("meta_daily.json" if not max7 else "meta_max7.json"))
    if cfg_file.exists():
        cfg = json.loads(cfg_file.read_text())
        cfg["hybrid_physics_leads"] = [k for k, c in choice.items() if c == "physics"]
        cfg_file.write_text(json.dumps(cfg))
    print(f"Saved hybrid choice and updated {cfg_file.name}")


if __name__ == "__main__":
    main()
