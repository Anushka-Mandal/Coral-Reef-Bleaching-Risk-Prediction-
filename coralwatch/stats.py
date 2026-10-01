"""Zone statistics for a grid - the same numbers the website's zone cards show."""

import numpy as np

from . import geo


def zone_stats(lats, lons, baa, dhw=None, ssta=None, hotspot=None):
    """Summary per zone for one day. Arrays are [n_lat, n_lon]; NaN = land / outside park."""
    baa = np.asarray(baa, float)
    valid = ~np.isnan(baa)
    rows = geo.zone_rows(lats)
    reef_idx = geo.reef_cells(lats, lons, valid)
    reefs = geo.load_geo()["reefs"]
    flat_baa = baa.ravel()
    out = []
    for z in geo.zones():
        m = valid & rows[z["id"]][:, None]
        n = int(m.sum())
        levels = baa[m].astype(int) if n else np.array([], int)
        counts = np.bincount(levels, minlength=5).tolist()
        in_zone = [k for k, r in enumerate(reefs) if r["z"] == z["id"] and reef_idx[k] >= 0]
        reef_levels = flat_baa[reef_idx[in_zone]] if in_zone else np.array([])
        item = {
            "zone": z["id"],
            "name": z["name"],
            "cells": n,
            "area_share_by_level": {geo.BAA_NAMES[i]: round(c / n, 3) if n else 0 for i, c in enumerate(counts)},
            "worst_level": geo.BAA_NAMES[int(levels.max())] if n else None,
            "reefs_total": len(in_zone),
            "reefs_at_alert": int((reef_levels >= 3).sum()),
            "reefs_by_level": {geo.BAA_NAMES[i]: int((reef_levels == i).sum()) for i in range(5)},
        }
        for key, arr in (("dhw", dhw), ("ssta", ssta), ("hotspot", hotspot)):
            if arr is not None and n:
                vals = np.asarray(arr, float)[m]
                vals = vals[~np.isnan(vals)]
                if len(vals):
                    item[f"{key}_mean"] = round(float(vals.mean()), 2)
                    item[f"{key}_max"] = round(float(vals.max()), 2)
        out.append(item)
    return out
