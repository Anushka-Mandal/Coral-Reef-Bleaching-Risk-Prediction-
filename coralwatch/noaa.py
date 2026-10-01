"""NOAA Coral Reef Watch data access (used by the forecast pipeline and the web backend).

Two ERDDAP servers serve the same 5 km v3.1 product on the same grid:
  * NOAA CoastWatch  - one dataset per variable; fast for multi-day requests -> model input
  * PacIOOS mirror   - all variables in one dataset incl. BAA and SST     -> map layers
"""

import datetime as dt
import io
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

NORTH, SOUTH, WEST, EAST = -10.4, -21.0, 142.3, 152.9
MODEL_STRIDE = 2  # 0.05 deg pixels -> 0.1 deg grid

COASTWATCH = "https://coastwatch.noaa.gov/erddap/griddap"
MODEL_DATASETS = {  # our name -> (dataset id, variable)
    "ssta": ("noaacrwsstanomalyDaily", "sea_surface_temperature_anomaly"),
    "hotspot": ("noaacrwhotspotDaily", "hotspot"),
    "dhw": ("noaacrwdhwDaily", "degree_heating_week"),
}
MODEL_VARIABLES = list(MODEL_DATASETS)

PACIOOS = "https://pae-paha.pacioos.hawaii.edu/erddap/griddap/dhw_5km"
MAP_VARIABLES = {"baa": "CRW_BAA", "dhw": "CRW_DHW", "ssta": "CRW_SSTANOMALY",
                 "sst": "CRW_SST", "hotspot": "CRW_HOTSPOT"}

USER_AGENT = {"User-Agent": "CoralWatch-capstone/1.0"}


def _get(url, timeout):
    req = urllib.request.Request(url, headers=USER_AGENT)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def is_netcdf(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size < 1000:
        return False
    with open(path, "rb") as f:
        return f.read(3) == b"CDF"


# ── Model input: raw values on the 0.1 deg grid ──────────────────────
def nc_url(var, start, end):
    ds, name = MODEL_DATASETS[var]
    dims = (f"[({start.isoformat()}):1:({end.isoformat()})]"
            f"[({NORTH}):{MODEL_STRIDE}:({SOUTH})][({WEST}):{MODEL_STRIDE}:({EAST})]")
    return f"{COASTWATCH}/{ds}.nc?" + urllib.parse.quote(name + dims, safe="(),:")


def fetch_nc(var, start, end, timeout=300):
    data = _get(nc_url(var, start, end), timeout)
    if data[:3] != b"CDF":
        raise ValueError(f"NOAA returned an error for {var} {start}..{end}: {data[:200]!r}")
    return data


def read_nc(data_or_path, var):
    """-> (dates, lats, lons, values[T, H, W] float32 with NaN for missing)."""
    from scipy.io import netcdf_file

    src = io.BytesIO(data_or_path) if isinstance(data_or_path, (bytes, bytearray)) else str(data_or_path)
    with netcdf_file(src, "r", mmap=False) as f:
        name = MODEL_DATASETS[var][1]
        v = f.variables[name]
        vals = v.data.astype(np.float32).copy()
        fill = getattr(v, "_FillValue", None)
        if fill is not None:
            vals[np.isclose(vals, fill)] = np.nan
        secs = f.variables["time"].data.astype(float)
        dates = [(dt.datetime(1970, 1, 1) + dt.timedelta(seconds=s)).date() for s in secs]
        lats = f.variables["latitude"].data.astype(float).copy()
        lons = f.variables["longitude"].data.astype(float).copy()
    return dates, lats, lons, vals


PACIOOS_MODEL_NAMES = {"ssta": "CRW_SSTANOMALY", "hotspot": "CRW_HOTSPOT", "dhw": "CRW_DHW", "sst": "CRW_SST"}


def _pacioos_model_days(start, end, timeout, variables=None):
    """Model variables in one PacIOOS request -> (dates, lats, lons, x[T, V, H, W])."""
    variables = list(variables or MODEL_VARIABLES)
    from scipy.io import netcdf_file

    dims = (f"[({start.isoformat()}):1:({end.isoformat()})]"
            f"[({NORTH}):{MODEL_STRIDE}:({SOUTH})][({WEST}):{MODEL_STRIDE}:({EAST})]")
    query = ",".join(PACIOOS_MODEL_NAMES[v] + dims for v in variables)
    data = _get(f"{PACIOOS}.nc?" + urllib.parse.quote(query, safe="(),:"), timeout)
    if data[:3] != b"CDF":
        raise ValueError(f"PacIOOS returned an error: {data[:200]!r}")
    with netcdf_file(io.BytesIO(data), "r", mmap=False) as f:
        chans = []
        for v in variables:
            var = f.variables[PACIOOS_MODEL_NAMES[v]]
            a = var.data.astype(np.float32).copy()
            fill = getattr(var, "_FillValue", None)
            if fill is not None:
                a[np.isclose(a, fill)] = np.nan
            chans.append(a)
        secs = f.variables["time"].data.astype(float)
        dates = [(dt.datetime(1970, 1, 1) + dt.timedelta(seconds=s)).date() for s in secs]
        lats = f.variables["latitude"].data.astype(float).copy()
        lons = f.variables["longitude"].data.astype(float).copy()
    return dates, lats, lons, np.stack(chans, axis=1)


def fetch_recent_model_days(end, days, retries=2):
    """Raw model variables for the `days` days ending on `end` -> (dates, lats, lons, x[T, 3, H, W]).
    Tries PacIOOS (one request), then NOAA CoastWatch (one request per variable)."""
    start = end - dt.timedelta(days=days - 1)
    errors = []
    for attempt in range(retries):
        try:
            return _pacioos_model_days(start, end, timeout=90)
        except Exception as e:
            errors.append(f"PacIOOS: {e}")
            time.sleep(2 * (attempt + 1))
    try:
        per_var = {v: read_nc(fetch_nc(v, start, end, timeout=90), v) for v in MODEL_VARIABLES}
    except Exception as e:
        errors.append(f"CoastWatch: {e}")
        raise RuntimeError("NOAA data unavailable - " + "; ".join(errors))
    dates, lats, lons, _ = per_var[MODEL_VARIABLES[0]]
    return dates, lats, lons, np.stack([per_var[v][3] for v in MODEL_VARIABLES], axis=1)


def latest_model_date():
    """Most recent day available on PacIOOS (falls back to CoastWatch)."""
    try:
        text = _get(f"{PACIOOS}.csv?time%5B(last)%5D", 30).decode()
        return dt.date.fromisoformat(text.strip().splitlines()[-1][:10])
    except Exception:
        return _coastwatch_latest()


def _coastwatch_latest():
    ends = []
    for var in MODEL_VARIABLES:
        ds, _ = MODEL_DATASETS[var]
        text = _get(f"{COASTWATCH}/{ds}.csv?time%5B(last)%5D", 60).decode()
        ends.append(dt.date.fromisoformat(text.strip().splitlines()[-1][:10]))
    return min(ends)


# ── Map layers: all variables, JSON table -> compact grid ─────────────
def fetch_map_grid(date="latest", stride=2, timeout=120):
    """Map data for one day in the website's compact grid format (not yet park-clipped)."""
    import json

    t = "(last)" if date == "latest" else f"({date})"
    dims = f"[{t}][({NORTH}):{stride}:({SOUTH})][({WEST}):{stride}:({EAST})]"
    query = ",".join(v + dims for v in MAP_VARIABLES.values())
    url = f"{PACIOOS}.json?" + urllib.parse.quote(query, safe="(),:")
    table = json.loads(_get(url, timeout))["table"]
    cols, rows = table["columnNames"], table["rows"]
    lats = sorted({r[1] for r in rows}, reverse=True)
    lons = sorted({r[2] for r in rows})
    grid = {"date": rows[0][0][:10], "lats": lats, "lons": lons,
            "step": round(abs(lats[1] - lats[0]), 4), "vars": {}}
    for key, col in MAP_VARIABLES.items():
        i = cols.index(col)
        grid["vars"][key] = [r[i] for r in rows]
    return grid


def fetch_days_cached(start, end, cache_dir, variables=("ssta", "hotspot", "dhw", "sst"), chunk=31):
    """Days start..end of the model variables on the 0.1 deg grid, cached one file per day.
    Only missing days are downloaded (PacIOOS, in chunks of up to a month).
    -> (dates, lats, lons, x[T, V, H, W] float16); days NOAA doesn't have are skipped."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(variables)
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    path = lambda d: cache_dir / f"{tag}_{d.isoformat()}.npz"
    missing = [d for d in days if not path(d).exists()]
    grid = None
    while missing:
        s0 = missing[0]
        run = [d for d in missing if (d - s0).days < chunk and d in days]
        e0 = run[-1]
        got_dates, lats, lons, x = _pacioos_model_days(s0, e0, timeout=180, variables=variables)
        grid = (lats, lons)
        for d, arr in zip(got_dates, x):
            np.savez_compressed(path(d), x=arr.astype(np.float16), lats=lats, lons=lons)
        missing = [d for d in missing if d > e0]      # days NOAA doesn't have are simply skipped
    dates, arrays = [], []
    for d in days:
        if path(d).exists():
            z = np.load(path(d))
            arrays.append(z["x"]); dates.append(d)
            grid = grid or (z["lats"], z["lons"])
    return dates, grid[0], grid[1], np.stack(arrays)
