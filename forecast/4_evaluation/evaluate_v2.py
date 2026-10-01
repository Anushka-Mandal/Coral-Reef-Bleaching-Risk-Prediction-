"""
evaluate_v2.py
==============
Full evaluation of the forecasters on the held-out test seasons (2024-2025).

    python forecast/evaluate_v2.py

For every model and every lead time 1-14 days:
  accuracy, macro-F1, alert precision / recall / F1, within-one-level accuracy
  95% confidence intervals by bootstrap over forecast days (1,000 resamples)
  paired bootstrap test against persistence (one-sided p-value)
Plus, for the chosen ensemble at the 7-day lead: reliability of the alert probability
(Brier score, skill vs climatology) and the 2024 event maps.

Outputs: forecast/results/v2/{results.json, RESULTS.md, lead_curve.png, reliability.png,
                              case_2024.png, ablation.png}
"""

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]

from coralwatch import geo  # noqa: E402
from dataset import SPLITS, Cube, is_scored, season_of  # noqa: E402
from physnet import LEADS, Drivers, PhysNet  # noqa: E402

from paths import MODELS_V2 as MODELS, RESULTS_DAILY as OUT, RESULTS_FINAL, WEATHER  # noqa: E402
N_BOOT = 1000
BLOCK = 7   # moving-block bootstrap: consecutive forecast days are correlated, so resample whole weeks
CHUNK = 24
rng = np.random.default_rng(0)


# ── Confusion-matrix metrics (so bootstrap can resample days cheaply) ─
def metrics_from_cm(cm):
    total = cm.sum()
    acc = np.trace(cm) / total
    tp = np.diag(cm); fp = cm.sum(0) - tp; fn = cm.sum(1) - tp
    f1 = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0)
    present = cm.sum(1) > 0
    a_tp = cm[3:, 3:].sum(); a_fp = cm[:3, 3:].sum(); a_fn = cm[3:, :3].sum()
    prec = a_tp / max(a_tp + a_fp, 1); rec = a_tp / max(a_tp + a_fn, 1)
    i, j = np.indices(cm.shape)
    return {"accuracy": acc, "macro_f1": f1[present].mean(), "alert_precision": prec, "alert_recall": rec,
            "alert_f1": 2 * prec * rec / max(prec + rec, 1e-9), "within_one": cm[np.abs(i - j) <= 1].sum() / total}


def day_cm(y, p):
    ok = ~np.isnan(y) & ~np.isnan(p)
    cm = np.zeros((5, 5), np.int64)
    np.add.at(cm, (y[ok].astype(int), p[ok].astype(int)), 1)
    return cm


def block_sample(n):
    """Indices for one moving-block bootstrap resample of n chronologically ordered days."""
    starts = rng.integers(0, max(1, n - BLOCK + 1), int(np.ceil(n / BLOCK)))
    return np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in starts])[:n]


def boot(cms, keys=("accuracy", "macro_f1", "alert_f1", "alert_recall", "within_one")):
    """cms: [days, 5, 5] in date order -> {metric: (value, lo, hi)} and the bootstrap samples."""
    point = metrics_from_cm(cms.sum(0))
    samples = {k: [] for k in keys}
    n = len(cms)
    for _ in range(N_BOOT):
        m = metrics_from_cm(cms[block_sample(n)].sum(0))
        for k in keys:
            samples[k].append(m[k])
    return {k: (float(point[k]), float(np.percentile(samples[k], 2.5)), float(np.percentile(samples[k], 97.5))) for k in keys}, \
           {k: np.array(v) for k, v in samples.items()}


def paired_p(cms_a, cms_b, key):
    """One-sided p-value that model a is NOT better than b (paired bootstrap over days)."""
    n = len(cms_a); worse = 0
    for _ in range(N_BOOT):
        s = block_sample(n)
        worse += metrics_from_cm(cms_a[s].sum(0))[key] <= metrics_from_cm(cms_b[s].sum(0))[key]
    return worse / N_BOOT


# ── Baselines for any lead ────────────────────────────────────────────
def persistence(cube, t, k):
    return cube.baa[t]


def trend_rule(cube, t, k):
    w = cube.window(t)
    hs = w[-1, 1] + k * (w[-1, 1] - w[0, 1]) / 6
    dhw = np.maximum(0, w[-1, 2] + k * (w[-1, 2] - w[0, 2]) / 6)
    return geo.baa_from(hs, dhw)


def physics_persistence(builder, cube, t, k):
    """HotSpot stays at today's value; DHW follows NOAA's formula exactly."""
    hs = cube.x[t, 1].astype(np.float32)
    dhw = cube.x[t, 2].astype(np.float32) + k * np.where(hs >= 1, hs, 0) / 7 - builder.dropoff(t)[k - 1]
    return geo.baa_from(hs, np.maximum(dhw, 0))


def window_max(cube, t, k, daily_pred):
    """7-day-maximum forecast for lead k: max of observed levels already inside the window
    (days t+k-6 .. t) and the predicted daily levels for days t+1 .. t+k."""
    day = cube.dates[t]
    parts = [daily_pred(j) for j in range(1, k + 1)]
    parts += [cube.baa[cube.index[day - dt.timedelta(days=j)]] for j in range(0, 7 - k)
              if day - dt.timedelta(days=j) in cube.index]
    with np.errstate(all="ignore"):
        return np.nanmax(np.stack(parts), axis=0)


def main():
    global OUT
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=["daily", "max7"], default="daily")
    args = ap.parse_args()
    out_dir = OUT if args.target == "daily" else RESULTS_FINAL
    out_dir.mkdir(parents=True, exist_ok=True)
    cube = Cube()
    drivers = Drivers(WEATHER, cube.lats, cube.lons)
    log = json.loads((MODELS / "log.json").read_text())
    test = cube.samples("test")
    test_seasons = SPLITS["test"]
    valid = cube.valid
    tgt = cube.baa7 if args.target == "max7" else cube.baa

    def target(t, k):
        day = cube.dates[t] + dt.timedelta(days=k)
        i = cube.index.get(day)
        if i is None or not is_scored(day) or season_of(day) not in test_seasons:
            return None
        return tgt[i]

    # models trained for this target: single-seed runs of every experiment + the selected ensemble
    runs = {k: v for k, v in log.items() if not k.startswith("_")}
    loaded = {key: PhysNet.load(MODELS / f"{key}.pt", cube, drivers) for key in runs}
    loaded = {k: m for k, m in loaded.items() if m.cfg.target == args.target}
    runs = {k: v for k, v in runs.items() if k in loaded}
    by_exp = {}
    for key, info in runs.items():
        by_exp.setdefault(info["experiment"], []).append(key)
    if args.target == "daily":
        selected = log.get("_selected")
    else:   # choose among this target's experiments by mean validation score
        selected = max(by_exp, key=lambda e: np.mean([runs[k]["best_val"] for k in by_exp[e]]))
    labels = {info["experiment"]: info["label"] for info in runs.values()}

    names = ["Persistence", "Trend + NOAA rule", "Physics persistence"]
    names += [labels[e] + " (seed 0)" for e in by_exp]
    ens_name = f"{labels[selected]} - ensemble of {len(by_exp[selected])}" if selected else None
    if ens_name:
        names.append(ens_name)
    cms = {n: np.zeros((len(test), LEADS, 5, 5), np.int64) for n in names}
    has = np.zeros((len(test), LEADS), bool)
    calib_p, calib_y = [], []
    case = {}
    case_day = dt.date(2024, 3, 3)

    b0 = next(iter(loaded.values())).builder
    for s in range(0, len(test), CHUNK):
        chunk = test[s:s + CHUNK]
        preds = {}
        for e, keys in by_exp.items():
            seed0 = [k for k in keys if k.endswith("_s0")][0]
            preds[e] = {kk: loaded[kk].predict_all(chunk)[0] for kk in (keys if e == selected else [seed0])}
        for ci, t in enumerate(chunk):
            n = s + ci
            for k in range(1, LEADS + 1):
                y = target(t, k)
                if y is None:
                    continue
                has[n, k - 1] = True
                ym = y[valid]
                if args.target == "daily":
                    cms["Persistence"][n, k - 1] = day_cm(ym, persistence(cube, t, k)[valid])
                    cms["Trend + NOAA rule"][n, k - 1] = day_cm(ym, trend_rule(cube, t, k)[valid])
                    cms["Physics persistence"][n, k - 1] = day_cm(ym, physics_persistence(b0, cube, t, k)[valid])
                else:
                    cms["Persistence"][n, k - 1] = day_cm(ym, cube.baa7[t][valid])
                    cms["Trend + NOAA rule"][n, k - 1] = day_cm(ym, window_max(cube, t, k, lambda j: trend_rule(cube, t, j))[valid])
                    cms["Physics persistence"][n, k - 1] = day_cm(ym, window_max(cube, t, k, lambda j: physics_persistence(b0, cube, t, j))[valid])
                for e, pr in preds.items():
                    seed0 = [kk for kk in pr if kk.endswith("_s0")][0]
                    p = pr[seed0][ci, k - 1].astype(np.float32).argmax(0)
                    cms[labels[e] + " (seed 0)"][n, k - 1] = day_cm(ym, p[valid])
                    if e == selected:
                        avg = np.mean([pr[kk][ci, k - 1].astype(np.float32) for kk in pr], 0)
                        cms[ens_name][n, k - 1] = day_cm(ym, avg.argmax(0)[valid])
                        if k == 7:
                            calib_p.append((avg[3] + avg[4])[valid]); calib_y.append(ym >= 3)
                        if cube.dates[t] + dt.timedelta(days=k) == case_day and k in (3, 7, 14):
                            case[k] = (avg.argmax(0).astype(float), y)
        print(f"  evaluated {min(s + CHUNK, len(test))}/{len(test)} forecast days", flush=True)

    # ── Tables with confidence intervals ─────────────────────────────
    results = {"test_seasons": f"{min(test_seasons)}-{max(test_seasons)}", "n_forecast_days": len(test),
               "selected": ens_name, "leads": {}, "significance_vs_persistence_7d": {}}
    for k in range(1, LEADS + 1):
        rows = has[:, k - 1]
        results["leads"][k] = {n: boot(cms[n][rows, k - 1])[0] for n in names}
        print(f"lead {k:2d}: " + "  ".join(f"{n[:18]} {results['leads'][k][n]['accuracy'][0]:.3f}" for n in names), flush=True)
    rows7 = has[:, 6]
    for n in names[1:]:
        results["significance_vs_persistence_7d"][n] = {
            key: paired_p(cms[n][rows7, 6], cms["Persistence"][rows7, 6], key) for key in ("accuracy", "macro_f1", "alert_f1")}

    # ── Reliability of the alert probability (7-day lead) ─────────────
    if calib_p:
        pa, ya = np.concatenate(calib_p), np.concatenate(calib_y).astype(float)
        ok = ~np.isnan(pa)
        pa, ya = pa[ok], ya[ok]
        bins = np.linspace(0, 1, 11)
        which = np.clip(np.digitize(pa, bins) - 1, 0, 9)
        rel = [{"bin": f"{bins[i]:.1f}-{bins[i + 1]:.1f}", "forecast": float(pa[which == i].mean()),
                "observed": float(ya[which == i].mean()), "n": int((which == i).sum())} for i in range(10) if (which == i).any()]
        brier = float(np.mean((pa - ya) ** 2))
        train_idx = cube.samples("train")
        base = float(np.mean([np.nanmean(tgt[cube.index[cube.dates[t] + dt.timedelta(days=7)]][valid] >= 3) for t in train_idx]))
        brier_clim = float(np.mean((base - ya) ** 2))
        results["reliability_7d"] = {"brier": brier, "brier_climatology": brier_clim,
                                     "brier_skill_score": 1 - brier / brier_clim, "bins": rel}

    results["validation_scores"] = {info["label"] + f" (seed {info['seed']})": info["best_val"] for info in runs.values()}
    results["target"] = "NOAA 7-day maximum alert" if args.target == "max7" else "daily alert level"
    OUT = out_dir
    (OUT / "results.json").write_text(json.dumps(results, indent=1))
    write_markdown(results, names)
    figures(results, names, ens_name, case, cube)
    print(f"Saved {OUT}")


def write_markdown(r, names):
    L = [f"# Forecast results v2 - physics-guided model ({r.get('target', 'daily alert level')})", "",
         f"Held-out test seasons {r['test_seasons']} ({r['n_forecast_days']} forecast days, 1,927 reef cells each). "
         "Brackets: 95% confidence intervals from a moving-block bootstrap (7-day blocks) over forecast days.", "",
         f"**Selected on validation:** {r['selected']}", ""]
    cols = [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("alert_f1", "Alert F1"), ("alert_recall", "Alert recall"), ("within_one", "Within one level")]
    for k in (7, 3, 1, 14):
        L += [f"## {k}-day lead", "", "| Model | " + " | ".join(c for _, c in cols) + " |", "|---|" + "---|" * len(cols)]
        for n in names:
            m = r["leads"][k][n]
            L.append(f"| {n} | " + " | ".join(f"{m[c][0]:.3f} [{m[c][1]:.3f}-{m[c][2]:.3f}]" for c, _ in cols) + " |")
        L.append("")
    L += ["## Better than persistence at 7 days? (paired bootstrap, one-sided p-values)", "", "| Model | Accuracy | Macro-F1 | Alert F1 |", "|---|---|---|---|"]
    for n, p in r["significance_vs_persistence_7d"].items():
        L.append(f"| {n} | {p['accuracy']:.3f} | {p['macro_f1']:.3f} | {p['alert_f1']:.3f} |")
    L += ["", "p < 0.05 means the model is better than persistence with 95% confidence.", ""]
    if "reliability_7d" in r:
        rl = r["reliability_7d"]
        L += ["## Reliability of the 7-day alert probability", "",
              f"Brier score {rl['brier']:.4f} vs {rl['brier_climatology']:.4f} for climatology "
              f"(skill score {rl['brier_skill_score']:.3f}; above 0 = more useful than the long-term average).", "",
              "| Forecast probability | Mean forecast | Observed frequency | Cells |", "|---|---|---|---|"]
        L += [f"| {b['bin']} | {b['forecast']:.2f} | {b['observed']:.2f} | {b['n']:,} |" for b in rl["bins"]]
    L += ["", "## Validation scores used for model choice (mean of accuracy and macro-F1, 7-day lead)", ""]
    L += [f"- {n}: {v:.4f}" for n, v in r["validation_scores"].items()]
    (OUT / "RESULTS.md").write_text("\n".join(L) + "\n")


def figures(r, names, ens_name, case, cube):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    leads = list(range(1, LEADS + 1))
    show = ["Persistence", "Trend + NOAA rule", "Physics persistence"] + ([ens_name] if ens_name else [])
    colors = ["#898781", "#eda100", "#1baf7a", "#2a78d6"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (key, title) in zip(axes, [("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("alert_f1", "Alert F1")]):
        for n, c in zip(show, colors):
            v = np.array([r["leads"][k][n][key] for k in leads])
            ax.plot(leads, v[:, 0], color=c, lw=2, label=n if key == "accuracy" else None)
            ax.fill_between(leads, v[:, 1], v[:, 2], color=c, alpha=0.15, lw=0)
        ax.set_title(title); ax.set_xlabel("Lead time (days)"); ax.grid(alpha=0.25); ax.set_xticks([1, 3, 5, 7, 10, 14])
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Skill vs lead time on held-out 2024-2025 seasons (shaded: 95% CI)")
    fig.tight_layout(); fig.savefig(OUT / "lead_curve.png", dpi=150); plt.close(fig)

    if "reliability_7d" in r:
        b = r["reliability_7d"]["bins"]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot([0, 1], [0, 1], "--", color="#898781", lw=1)
        ax.plot([x["forecast"] for x in b], [x["observed"] for x in b], "o-", color="#2a78d6", lw=2)
        ax.set_xlabel("Forecast probability of Alert (7 days)"); ax.set_ylabel("Observed frequency")
        ax.set_title(f"Reliability (Brier skill {r['reliability_7d']['brier_skill_score']:.2f})"); ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(OUT / "reliability.png", dpi=150); plt.close(fig)

    if case:
        cmap = ListedColormap(["#8ec9f2", "#f8d74a", "#fb953a", "#ff5236", "#d4145a"])
        ext = [cube.lons[0], cube.lons[-1], cube.lats[-1], cube.lats[0]]
        ks = sorted(case)
        fig, axes = plt.subplots(1, len(ks) + 1, figsize=(4.2 * (len(ks) + 1), 5))
        axes[0].imshow(np.ma.masked_invalid(case[ks[0]][1]), cmap=cmap, vmin=-0.5, vmax=4.5, extent=ext); axes[0].set_title("Observed 3 Mar 2024")
        for ax, k in zip(axes[1:], ks):
            arr = case[k][0].copy(); arr[~cube.valid] = np.nan
            ax.imshow(np.ma.masked_invalid(arr), cmap=cmap, vmin=-0.5, vmax=4.5, extent=ext); ax.set_title(f"Forecast issued {k} days before")
        fig.tight_layout(); fig.savefig(OUT / "case_2024.png", dpi=150); plt.close(fig)

    abl = [n for n in names if n.endswith("(seed 0)")] + ["Persistence"]
    v = np.array([r["leads"][7][n]["macro_f1"] for n in abl])
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.barh(range(len(abl)), v[:, 0], xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], color="#2a78d6", height=0.6)
    ax.set_yticks(range(len(abl)), [n.replace(" (seed 0)", "") for n in abl]); ax.invert_yaxis()
    ax.set_xlabel("Macro-F1, 7-day lead, test seasons (95% CI)"); ax.set_title("Ablation: what each part adds")
    ax.set_xlim(max(0, v[:, 1].min() - 0.05), min(1, v[:, 2].max() + 0.03)); ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(); fig.savefig(OUT / "ablation.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
