"""Study-area geography: Marine Park boundary, reefs, zones and grid masks.

The boundary and reef outlines come from website/data/gbr.js (GBRMPA open data,
built by website/build_data.py), so the model, the server and the map all use
exactly the same shapes.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GBR_JS = ROOT / "website" / "data" / "gbr.js"

BAA_NAMES = ["No Stress", "Watch", "Warning", "Alert Level 1", "Alert Level 2"]


@lru_cache(maxsize=1)
def load_geo():
    text = GBR_JS.read_text()
    return json.loads(text[text.index("=") + 1:].strip().rstrip(";"))


def study():
    return load_geo()["study"]


def zones():
    s = study()
    return [
        {"id": "N", "name": "Northern GBR", "north": s["north"], "south": s["split"]},
        {"id": "C", "name": "Central GBR", "north": s["split"], "south": s["south"]},
    ]


def zone_by_id(zone_id):
    for z in zones():
        if z["id"] == zone_id:
            return z
    raise KeyError(zone_id)


def points_in_polygon(lats, lons, ring):
    """Vectorised ray casting. lats/lons are arrays of the same shape; ring is [[lat, lon], ...]."""
    lats, lons = np.asarray(lats, float), np.asarray(lons, float)
    inside = np.zeros(lats.shape, bool)
    ring = np.asarray(ring, float)
    yi, xi = ring[:, 0], ring[:, 1]
    yj, xj = np.roll(yi, 1), np.roll(xi, 1)
    for a_y, a_x, b_y, b_x in zip(yi, xi, yj, xj):
        crosses = (a_y > lats) != (b_y > lats)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at = (b_x - a_x) * (lats - a_y) / (b_y - a_y) + a_x
        inside ^= crosses & (lons < x_at)
    return inside


def park_mask(lats, lons):
    """Boolean [len(lats), len(lons)] mask of grid cells inside the Marine Park."""
    lat2, lon2 = np.meshgrid(lats, lons, indexing="ij")
    return points_in_polygon(lat2, lon2, load_geo()["boundary"])


def zone_rows(lats):
    """Dict zone id -> boolean mask over the lat axis (north inclusive, south exclusive)."""
    lats = np.asarray(lats)
    return {z["id"]: (lats <= z["north"]) & (lats > z["south"]) for z in zones()}


def reef_cells(lats, lons, valid):
    """For every reef, the flat index of the nearest valid grid cell (within 2 cells), or -1."""
    lats, lons = np.asarray(lats), np.asarray(lons)
    step = abs(lats[1] - lats[0])
    n_lat, n_lon = len(lats), len(lons)
    out = []
    for r in load_geo()["reefs"]:
        i0 = int(round((lats[0] - r["c"][0]) / step))
        j0 = int(round((r["c"][1] - lons[0]) / step))
        found = -1
        for rad in range(3):
            for di in range(-rad, rad + 1):
                for dj in range(-rad, rad + 1):
                    i, j = i0 + di, j0 + dj
                    if found < 0 and 0 <= i < n_lat and 0 <= j < n_lon and valid[i, j]:
                        found = i * n_lon + j
            if found >= 0:
                break
        out.append(found)
    return np.array(out)


def baa_from(hotspot, dhw):
    """NOAA Coral Reef Watch v3.1 Bleaching Alert Area from HotSpot and DHW.
    Verified to match NOAA's published CRW_BAA on every pixel of Feb 2024 (240k pixels)."""
    hotspot, dhw = np.asarray(hotspot, float), np.asarray(dhw, float)
    baa = np.select([hotspot <= 0, hotspot < 1, dhw < 4, dhw < 8], [0, 1, 2, 3], 4).astype(float)
    baa[np.isnan(hotspot) | np.isnan(dhw)] = np.nan
    return baa
