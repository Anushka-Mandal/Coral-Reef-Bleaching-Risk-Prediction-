"""
recover_from_images.py
======================
Rebuilds the raw NOAA values for 2018-2025 from the project's own map images
(CNN/2_images), so the forecast can be trained without re-downloading 19 GB.

Why this works: CNN/SCRIPTS/nc_to_png.py painted every pixel with a fixed colour scale
(viridis 0-12 for DHW, viridis 0-6 for HotSpot, RdBu_r -3..3 for SST anomaly) with no
blending, so each colour maps back to one value. The first pipeline threw this
information away by averaging the colours to grey (for RdBu_r, +3 degC and -3 degC
then look the same).

Precision: one of 256 colour steps - 0.047 degC-weeks for DHW, 0.023 degC for HotSpot,
0.023 degC for SST anomaly. Values outside the scale were clipped when the images were
made (DHW above 12 reads as 12, which doesn't change the Bleaching Alert level).
Checked against real NOAA values at the end of this script.

    python forecast/1_data_preparation/recover_from_images.py      # writes forecast/data/cube.npz
"""

import datetime as dt
import json
import sys
from pathlib import Path

import matplotlib.cm as cm
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

FORECAST_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(FORECAST_DIR / "2_model"), str(FORECAST_DIR.parent)]
from coralwatch import geo  # noqa: E402
from paths import CUBE, DATA_CHECKS, IMAGES, SNAPSHOTS_JS  # noqa: E402

# channel -> (folder, file prefix, colormap, vmin, vmax), exactly as in nc_to_png.py
SCALES = {
    "sst": ("sst", "sst", cm.viridis, 24.0, 32.0),
    "ssta": ("sst_anomaly", "sst_anomaly", cm.RdBu_r, -3.0, 3.0),
    "hotspot": ("hotspot", "hotspot", cm.viridis, 0.0, 6.0),
    "dhw": ("dhw", "dhw", cm.viridis, 0.0, 12.0),
}
CHANNELS = ["ssta", "hotspot", "dhw", "sst"]   # first three match dataset.CHANNELS

# Images cover 10-24 S, 142-154 E at 0.05 deg; row 0 is centred on 10.025 S, column 0 on 142.025 E.
# The study grid (0.1 deg, 10.375-20.975 S, 142.325-152.825 E) is every second pixel from row 7, column 6.
ROWS = slice(7, 220, 2)
COLS = slice(6, 217, 2)
LATS = np.round(-10.375 - 0.1 * np.arange(107), 3)
LONS = np.round(142.325 + 0.1 * np.arange(106), 3)

_trees = {}


def invert(path, channel):
    folder, prefix, cmap, vmin, vmax = SCALES[channel]
    if channel not in _trees:
        _trees[channel] = cKDTree(cmap(np.linspace(0, 1, 256))[:, :3] * 255)
    img = np.asarray(Image.open(path).convert("RGB"))[ROWS, COLS].astype(float)
    flat = img.reshape(-1, 3)
    _, idx = _trees[channel].query(flat)
    val = np.where(idx == 0, vmin, vmin + (idx + 0.5) / 256 * (vmax - vmin))
    val[flat.sum(1) == 0] = np.nan          # black = land / missing
    return val.reshape(img.shape[:2]).astype(np.float32)


def image_path(channel, day):
    folder, prefix, *_ = SCALES[channel]
    return IMAGES / folder / f"{day.year}" / f"{day.month:02d}" / f"{prefix}_{day:%Y_%m_%d}.png"


def fill_ssta_from_sst(x, dates):
    """SST anomaly = SST - a fixed daily climatology. Where the anomaly image is missing
    (Jan-Aug 2023) but SST exists, rebuild it with the climatology learned from other years."""
    doy = np.array([min(d.timetuple().tm_yday, 365) for d in dates])
    sst, ssta = x[:, 3], x[:, 0]
    usable = ~np.isnan(sst) & ~np.isnan(ssta) & (sst > 24.05) & (sst < 31.95) & (np.abs(ssta) < 2.95)
    clim_diff = np.where(usable, sst - ssta, np.nan)
    missing = np.isnan(ssta).all(axis=(1, 2)) & ~np.isnan(sst).all(axis=(1, 2))
    filled = 0
    for i in np.where(missing)[0]:
        near = (np.abs(doy - doy[i]) <= 3) | (np.abs(doy - doy[i]) >= 362)
        near[i] = False
        clim = np.nanmedian(clim_diff[near], axis=0)
        x[i, 0] = np.clip(sst[i] - clim, -3, 3)
        filled += 1
    return filled


def build():
    day, end = dt.date(2018, 1, 1), dt.date(2025, 12, 31)
    dates, arrays, skipped = [], [], 0
    while day <= end:
        paths = [image_path(c, day) for c in CHANNELS]
        if all(image_path(c, day).exists() for c in ("hotspot", "dhw", "sst")):
            arrays.append(np.stack([invert(p, c) if p.exists() else np.full((len(LATS), len(LONS)), np.nan, np.float32)
                                    for p, c in zip(paths, CHANNELS)]))
            dates.append(day)
        else:
            skipped += 1
        day += dt.timedelta(days=1)
    x = np.stack(arrays)
    filled = fill_ssta_from_sst(x, dates)
    print(f"  rebuilt SST anomaly for {filled} days from SST and climatology")
    ok_days = ~np.isnan(x[:, :3]).all(axis=(2, 3)).any(axis=1)
    x, dates = x[ok_days], [d for d, k in zip(dates, ok_days) if k]
    valid = geo.park_mask(LATS, LONS) & (np.isnan(x[:, :3]).mean(axis=(0, 1)) < 0.02)
    x[:, :, ~valid] = np.nan
    CUBE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CUBE, x=x.astype(np.float16), dates=np.array([d.isoformat() for d in dates]),
                        lats=LATS, lons=LONS, valid=valid,
                        source="CNN/2_images recovered by colour-scale inversion")
    print(f"Saved {CUBE}: {len(dates)} days ({dates[0]} .. {dates[-1]}), "
          f"{skipped} days skipped for a missing channel, {valid.sum()} park cells")


def validate():
    """Compare recovered values with real NOAA values saved by website/build_data.py."""
    text = SNAPSHOTS_JS.read_text()
    snaps = json.loads(text[text.index("window.SNAPSHOTS = ") + 19:].strip().rstrip(";"))
    report = {}
    for date in ("2024-03-03", "2022-03-12", "2020-03-08", "2025-03-09"):
        if date not in snaps:
            continue
        g, day = snaps[date], dt.date.fromisoformat(date)
        lats, lons = np.array(g["lats"]), np.array(g["lons"])
        r0 = int(round((-10.025 - lats[0]) / 0.05)); c0 = int(round((lons[0] - 142.025) / 0.05))
        res = {}
        for c in CHANNELS:
            p = image_path(c, day)
            if not p.exists():
                continue
            full = SCALES[c]
            img = np.asarray(Image.open(p).convert("RGB")).astype(float)
            flat = img.reshape(-1, 3)
            _, idx = _trees.setdefault(c, cKDTree(full[2](np.linspace(0, 1, 256))[:, :3] * 255)).query(flat)
            val = np.where(idx == 0, full[3], full[3] + (idx + 0.5) / 256 * (full[4] - full[3])).reshape(img.shape[:2])
            val[flat.reshape(img.shape).sum(-1) == 0] = np.nan
            sub = val[r0:r0 + len(lats), c0:c0 + len(lons)]
            ref = np.array([np.nan if v is None else v for v in g["vars"][c]], float).reshape(len(lats), len(lons))
            ok = ~np.isnan(ref) & ~np.isnan(sub) & (ref > full[3]) & (ref < full[4])
            res[c] = round(float(np.abs(sub[ok] - ref[ok]).mean()), 4)
            if c == "hotspot":
                hs = sub
            if c == "dhw":
                baa_ref = np.array([np.nan if v is None else v for v in g["vars"]["baa"]], float).reshape(hs.shape)
                baa = geo.baa_from(hs, sub)
                m = ~np.isnan(baa) & ~np.isnan(baa_ref)
                res["baa_agreement"] = round(float((baa[m] == baa_ref[m]).mean()), 4)
        report[date] = res
        print(f"  {date}: mean abs error {res}")
    DATA_CHECKS.mkdir(parents=True, exist_ok=True)
    (DATA_CHECKS / "recovery_check.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    print("Checking recovered values against real NOAA data ...")
    validate()
    print("Building the dataset ...")
    build()
