"""Weather and climate drivers of reef heat stress.

Heat builds up under calm winds, clear skies and warm air, so these are the physical
predictors a 7-day forecast needs beyond sea temperature itself:

  daily weather (Open-Meteo, free, no key)   - 10 m wind speed, cloud cover,
                                               shortwave radiation, 2 m air temperature
      archive API  : ERA5-based history, used for training
      forecast API : live 16-day forecast, used by the website
  climate indices                             - ENSO (Nino 3.4 weekly SST anomaly, NOAA CPC)
                                               - MJO (RMM1/RMM2 daily, Bureau of Meteorology)

Weather is fetched on a 0.75 deg grid of points around the Marine Park and
interpolated linearly onto the 0.1 deg forecast grid.
"""

import datetime as dt
import json
import time
import urllib.parse
import urllib.request

import numpy as np

from . import geo

WEATHER_VARS = ["wind_speed_10m_mean", "cloud_cover_mean", "shortwave_radiation_sum", "temperature_2m_mean"]
WEATHER_NAMES = ["wind", "cloud", "radiation", "air_temp"]
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
NINO_URL = "https://www.cpc.ncep.noaa.gov/data/indices/wksst9120.for"
MJO_URL = "https://www.bom.gov.au/clim_data/IDCKGEM000/rmm.74toRealtime.txt"
UA = {"User-Agent": "Mozilla/5.0 (CoralWatch capstone research)"}

GRID_LATS = np.round(-10.375 - 0.1 * np.arange(107), 3)
GRID_LONS = np.round(142.325 + 0.1 * np.arange(106), 3)


def _get(url, timeout=120):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def weather_points(res=0.75):
    """Points of a regular grid within one grid step of any Marine Park cell."""
    park = geo.park_mask(GRID_LATS, GRID_LONS)
    pl, pn = np.meshgrid(GRID_LATS, GRID_LONS, indexing="ij")
    pl, pn = pl[park], pn[park]
    pts = []
    for a in np.arange(-10.0, -21.01, -res):
        for b in np.arange(142.0, 153.01, res):
            if np.min(np.hypot(pl - a, pn - b)) <= res * 1.05:
                pts.append((round(float(a), 3), round(float(b), 3)))
    return pts


def fetch_points(points, start, end, archive=True, forecast_days=16, timeout=120):
    """Daily weather for points -> (dates, values[T, P, 4]). NaN where missing."""
    params = {
        "latitude": ",".join(str(p[0]) for p in points),
        "longitude": ",".join(str(p[1]) for p in points),
        "daily": ",".join(WEATHER_VARS),
        "timezone": "Australia/Brisbane",
    }
    if archive:
        params.update(start_date=start.isoformat(), end_date=end.isoformat())
        base = ARCHIVE
    else:
        params.update(past_days=min(92, max(0, (dt.date.today() - start).days)), forecast_days=forecast_days)
        base = FORECAST
    data = json.loads(_get(base + "?" + urllib.parse.urlencode(params), timeout))
    if isinstance(data, dict):
        data = [data]
    dates = [dt.date.fromisoformat(d) for d in data[0]["daily"]["time"]]
    vals = np.full((len(dates), len(points), len(WEATHER_VARS)), np.nan, np.float32)
    for p, loc in enumerate(data):
        for v, name in enumerate(WEATHER_VARS):
            vals[:, p, v] = [np.nan if x is None else x for x in loc["daily"][name]]
    return dates, vals


def to_grid(points, values):
    """Interpolate point values [..., P] onto the 0.1 deg grid -> [..., 107, 106]."""
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    pts = np.array(points)
    tgt = np.stack(np.meshgrid(GRID_LATS, GRID_LONS, indexing="ij"), -1).reshape(-1, 2)
    flat = values.reshape(-1, values.shape[-1])
    out = np.empty((flat.shape[0], len(tgt)), np.float32)
    for i, v in enumerate(flat):
        ok = ~np.isnan(v)
        lin = LinearNDInterpolator(pts[ok], v[ok])(tgt)
        near = NearestNDInterpolator(pts[ok], v[ok])(tgt)
        out[i] = np.where(np.isnan(lin), near, lin)
    return out.reshape(values.shape[:-1] + (len(GRID_LATS), len(GRID_LONS)))


def fetch_nino34():
    """Weekly Nino 3.4 SST anomaly -> dict date -> value (week centre dates)."""
    text = _get(NINO_URL).decode()
    out = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0][:2].isdigit() and len(parts[0]) == 9:
            day = dt.datetime.strptime(parts[0], "%d%b%Y").date()
            # fixed-width: 'SST SSTA' pairs can run together, e.g. '20.6-0.1'
            fields = line[10:]
            nino34 = fields[26:39].split()
            try:
                out[day] = float(nino34[-1]) if len(nino34) == 2 else float(fields[26:39].strip()[4:])
            except (ValueError, IndexError):
                continue
    return out


def fetch_mjo():
    """Daily MJO RMM1/RMM2 -> dict date -> (rmm1, rmm2)."""
    text = _get(MJO_URL).decode()
    out = {}
    for line in text.splitlines()[2:]:
        p = line.split()
        if len(p) >= 5:
            try:
                r1, r2 = float(p[3]), float(p[4])
            except ValueError:
                continue
            if abs(r1) < 100 and abs(r2) < 100:
                out[dt.date(int(p[0]), int(p[1]), int(p[2]))] = (r1, r2)
    return out


def daily_series(values_by_date, dates):
    """Linear interpolation of a sparse date->value series onto `dates` (scalar or tuple values)."""
    keys = sorted(values_by_date)
    x = np.array([k.toordinal() for k in keys], float)
    y = np.array([values_by_date[k] for k in keys], float)
    q = np.array([d.toordinal() for d in dates], float)
    if y.ndim == 1:
        return np.interp(q, x, y).astype(np.float32)
    return np.stack([np.interp(q, x, y[:, i]) for i in range(y.shape[1])], 1).astype(np.float32)


def polite_sleep(seconds=2.0):
    time.sleep(seconds)


PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
HOURLY_VARS = ["wind_speed_10m", "cloud_cover", "shortwave_radiation", "temperature_2m"]


def fetch_previous_runs(points, start, end, leads=range(1, 8), timeout=180):
    """Archived forecasts: for each lead j, the forecast made j days before each date.
    -> dict j -> (dates, values[T, P, 4]) as daily wind mean, cloud mean, radiation sum (MJ/m2), air temp mean."""
    names = [f"{v}_previous_day{j}" for j in leads for v in HOURLY_VARS]
    params = {"latitude": ",".join(str(p[0]) for p in points), "longitude": ",".join(str(p[1]) for p in points),
              "hourly": ",".join(names), "start_date": start.isoformat(), "end_date": end.isoformat(),
              "timezone": "Australia/Brisbane"}
    data = json.loads(_get(PREVIOUS_RUNS + "?" + urllib.parse.urlencode(params), timeout))
    if isinstance(data, dict):
        data = [data]
    times = data[0]["hourly"]["time"]
    days = sorted({t[:10] for t in times})
    dates = [dt.date.fromisoformat(d) for d in days]
    day_of = np.array([days.index(t[:10]) for t in times])
    out = {}
    for j in leads:
        vals = np.full((len(dates), len(points), 4), np.nan, np.float32)
        for p, loc in enumerate(data):
            for v, name in enumerate(HOURLY_VARS):
                h = np.array([np.nan if x is None else x for x in loc["hourly"][f"{name}_previous_day{j}"]], float)
                for d in range(len(dates)):
                    sel = h[day_of == d]
                    if np.isnan(sel).all():
                        continue
                    vals[d, p, v] = np.nansum(sel) * 3600 / 1e6 if name == "shortwave_radiation" else np.nanmean(sel)
        out[j] = (dates, vals)
    return out
