"""
train_v2.py
===========
Trains the physics-guided forecaster and its ablations.

    python forecast/train_v2.py                    # all ablations, then extra seeds of the best
    python forecast/train_v2.py --only full        # one experiment

Experiments (same data, same split, one change at a time):
    ocean        PhysNet, ocean inputs only (SST anomaly, HotSpot, DHW, SST)
    past_wx      + observed weather (wind, cloud, radiation, air temp) and ENSO/MJO
    full         + weather for the next 14 days (ERA5 in training; a real forecast when live)
    no_physics   everything as 'full' but without the HotSpot head / DHW formula layer

The best experiment is chosen on the validation seasons (mean of accuracy and macro-F1
at the 7-day lead) and retrained with two more seeds for an ensemble.
Models -> forecast/models/v2/<experiment>_s<seed>.pt, histories -> models/v2/log.json
"""

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]

from dataset import Cube  # noqa: E402
from physnet import Config, Drivers, PhysNet  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

from paths import MODELS_V2 as OUT, WEATHER  # noqa: E402
LEAD = 7

EXPERIMENTS = {
    "ocean": Config(weather_past=False, weather_future=False, indices=False, name="PhysNet (ocean only)"),
    "past_wx": Config(weather_future=False, name="PhysNet + observed weather & climate indices"),
    "full": Config(name="PhysNet + weather forecast"),
    "no_physics": Config(physics=False, name="Same inputs, no DHW formula layer"),
}
# Leakage check, run after the main experiments: ERA5 air temperature over the sea is computed
# from the observed SST, so future air temperature may leak future sea temperature.
EXTRA = {"full_no_airtemp": Config(future_air_temp=False, name="PhysNet + weather forecast without air temperature"),
         # improvement round 2
         "max7": Config(target="max7", name="PhysNet + weather forecast, 7-day maximum alert"),
         "max7_noise": Config(target="max7", forecast_noise=True, name="PhysNet 7-day maximum alert, trained with forecast error")}


def lead_score(cube, model, idx):
    """Validation score: mean of accuracy and macro-F1 at the 7-day lead (for the model's own target)."""
    probs, _ = model.predict_all(idx)
    tgt = cube.baa7 if model.cfg.target == "max7" else cube.baa
    ys, ps = [], []
    for k, t in enumerate(idx):
        y = tgt[cube.index[cube.dates[t] + dt.timedelta(days=LEAD)]][cube.valid]
        p = probs[k, LEAD - 1].astype(np.float32).argmax(0)[cube.valid]
        ok = ~np.isnan(y)
        ys.append(y[ok].astype(int)); ps.append(p[ok])
    y, p = np.concatenate(ys), np.concatenate(ps)
    return 0.5 * (float((y == p).mean()) + float(f1_score(y, p, labels=range(5), average="macro", zero_division=0)))


def run(name, cfg, cube, drivers, idx, log):
    path = OUT / f"{name}_s{cfg.seed}.pt"
    if path.exists():
        print(f"== {name} seed {cfg.seed}: already trained", flush=True)
        return log.get(f"{name}_s{cfg.seed}", {}).get("best_val")
    print(f"\n== {cfg.name} (seed {cfg.seed})", flush=True)
    t0 = time.time()
    model = PhysNet(cfg).fit(cube, drivers, idx["train"], idx["val"], lambda m, v: lead_score(cube, m, v))
    model.save(path)
    log[f"{name}_s{cfg.seed}"] = {"experiment": name, "seed": cfg.seed, "label": cfg.name, "best_val": model.best_val,
                                  "minutes": round((time.time() - t0) / 60, 1), "history": model.history}
    (OUT / "log.json").write_text(json.dumps(log, indent=1))
    print(f"   best validation score {model.best_val:.4f} ({(time.time() - t0) / 60:.0f} min)", flush=True)
    return model.best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(EXPERIMENTS) + list(EXTRA))
    ap.add_argument("--seeds", type=int, default=3, help="ensemble size for the best experiment")
    ap.add_argument("--seed", type=int, default=0, help="seed for --only")
    ap.add_argument("--weather", default=str(WEATHER), help="weather file (the 'ocean' experiment doesn't use it)")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    log = json.loads((OUT / "log.json").read_text()) if (OUT / "log.json").exists() else {}
    cube = Cube()
    drivers = Drivers(args.weather, cube.lats, cube.lons)
    idx = {s: cube.samples(s) for s in ("train", "val", "test")}
    print({s: len(v) for s, v in idx.items()}, "issue days", flush=True)

    names = [args.only] if args.only else list(EXPERIMENTS)
    pick = lambda n: Config(**{**{**EXPERIMENTS, **EXTRA}[n].__dict__, "seed": args.seed}) if args.only else EXPERIMENTS[n]
    scores = {n: run(n, pick(n), cube, drivers, idx, log) for n in names}
    if args.only:
        return
    best = max(scores, key=lambda n: scores[n])
    print(f"\nBest on validation: {best} ({scores[best]:.4f}) - training {args.seeds - 1} more seeds", flush=True)
    for seed in range(1, args.seeds):
        cfg = Config(**{**EXPERIMENTS[best].__dict__, "seed": seed})
        run(best, cfg, cube, drivers, idx, log)
    log["_selected"] = best
    (OUT / "log.json").write_text(json.dumps(log, indent=1))
    (OUT / "meta.json").write_text(json.dumps({
        "label": EXPERIMENTS[best].name, "experiment": best,
        "ensemble": [f"{best}_s{s}" for s in range(args.seeds)],
        "fallback": "ocean_s0" if (OUT / "ocean_s0.pt").exists() else None,
        "fallback_label": EXPERIMENTS["ocean"].name,
        "lats": cube.lats.tolist(), "lons": cube.lons.tolist(), "valid": cube.valid.astype(int).tolist()}))


if __name__ == "__main__":
    main()
