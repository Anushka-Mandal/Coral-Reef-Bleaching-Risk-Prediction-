"""
dataset.py
==========
Builds the forecasting dataset from the raw daily NOAA files.

Task: on issue day t, using the 7 days t-6..t, predict the Bleaching Alert level of
every Marine Park cell on day t+7.

Fixes compared with the first (image-based) pipeline:
  * inputs are the raw physical values (degC, degC-weeks), not colour-mapped PNGs
  * samples are built by calendar date; a sample is dropped if any day it needs is missing
  * split by season: train 2018-2021, validation 2022-2023, test 2024-2025 (see SPLITS)
  * the model is chosen on validation only; test is used once for the final numbers
  * only warm-season target days (1 Nov - 30 Apr) are scored

The dataset (forecast/data/cube.npz) is built by one of:
    python forecast/recover_from_images.py   # from CNN/2_images, 2018-2025 (used for the results)
    python forecast/dataset.py               # from raw files fetched by download_noaa.py
"""

import datetime as dt
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from coralwatch import geo, noaa  # noqa: E402
from paths import CUBE, DATA, RAW_NOAA as RAW  # noqa: E402

SEQ_LEN = 7
HORIZON = 7
CHANNELS = ["ssta", "hotspot", "dhw"]  # order used everywhere (matches coralwatch.noaa.MODEL_VARIABLES)

SPLITS = {  # by warm season, named after the year it ends in (season 2024 = Nov 2023 - Apr 2024)
    "train": range(2018, 2022),   # 2018-2021 (includes the 2020 mass bleaching)
    "val": range(2022, 2024),     # 2022-2023 (includes the 2022 mass bleaching) - model choice only
    "test": range(2024, 2027),    # 2024-2025 (+ Nov-Dec 2025) - held out, used once
}


def season_of(day):
    """The warm season a date belongs to, named by the year it ends (Nov 2023 -> 2024)."""
    return day.year + 1 if day.month >= 7 else day.year


def is_scored(day):
    return day.month >= 11 or day.month <= 4


def build_cube():
    months = sorted(p.stem for p in (RAW / "ssta").glob("*.nc"))
    if not months:
        raise SystemExit("No raw data - run forecast/download_noaa.py first")
    days = {}
    lats = lons = None
    for m in months:
        try:
            parts = [noaa.read_nc(RAW / c / f"{m}.nc", c) for c in CHANNELS]
        except Exception as e:
            print(f"  skip {m}: {e}")
            continue
        lats, lons = parts[0][1], parts[0][2]
        by_var = [dict(zip(p[0], p[3])) for p in parts]
        for day in set.intersection(*(set(b) for b in by_var)):
            days[day] = np.stack([b[day] for b in by_var])
    dates = sorted(days)
    x = np.stack([days[d] for d in dates])                  # [T, 3, H, W]
    park = geo.park_mask(lats, lons)
    valid = park & ~np.isnan(x).any(axis=(0, 1))             # sea cells inside the park
    x[:, :, ~valid] = np.nan
    np.savez_compressed(CUBE, x=x.astype(np.float32), dates=np.array([d.isoformat() for d in dates]),
                        lats=lats, lons=lons, valid=valid)
    print(f"Saved {CUBE} - {len(dates)} days ({dates[0]} .. {dates[-1]}), "
          f"grid {x.shape[2]}x{x.shape[3]}, {valid.sum()} park cells")


class Cube:
    """In-memory dataset with calendar-aware sample lookup."""

    def __init__(self, path=CUBE):
        d = np.load(path)
        self.x = d["x"]
        self.dates = [dt.date.fromisoformat(s) for s in d["dates"]]
        self.lats, self.lons, self.valid = d["lats"], d["lons"], d["valid"]
        self.index = {day: i for i, day in enumerate(self.dates)}
        self.baa = geo.baa_from(self.x[:, 1], self.x[:, 2]).astype(np.float32)  # [T, H, W]

    @property
    def baa7(self):
        """NOAA's operational product: the highest daily alert level over the 7 days ending on
        each day (NaN if fewer than 5 of those days exist). [T, H, W]"""
        if not hasattr(self, "_baa7"):
            out = np.full(self.baa.shape, np.nan, np.float32)
            for i, day in enumerate(self.dates):
                ids = [self.index[day - dt.timedelta(days=k)] for k in range(7) if day - dt.timedelta(days=k) in self.index]
                if len(ids) >= 5:
                    with np.errstate(all="ignore"):
                        out[i] = np.nanmax(self.baa[ids], axis=0)
            self._baa7 = out
        return self._baa7

    def samples(self, split):
        """Issue-day indices t for the split: all of t-6..t and t+7 present, target day scored."""
        seasons = SPLITS[split]
        out = []
        for i, day in enumerate(self.dates):
            target = day + dt.timedelta(days=HORIZON)
            if not is_scored(target) or season_of(target) not in seasons:
                continue
            needed = [day - dt.timedelta(days=k) for k in range(SEQ_LEN)] + [target]
            if all(n in self.index for n in needed):
                out.append(i)
        return out

    def window(self, t):
        """Input frames [SEQ_LEN, 3, H, W] ending on issue index t."""
        day = self.dates[t]
        return np.stack([self.x[self.index[day - dt.timedelta(days=k)]] for k in range(SEQ_LEN - 1, -1, -1)]).astype(np.float32)

    def target(self, t):
        return self.baa[self.index[self.dates[t] + dt.timedelta(days=HORIZON)]]


if __name__ == "__main__":
    build_cube()
