"""
CoralWatch web server.

    .venv/bin/python -m backend.app        # then open http://127.0.0.1:8000

Endpoints
    GET  /api/health             what's working (NOAA, forecast model, LLM, report corpus)
    GET  /api/grid/{date}        map data for a day ('latest' = today's NOAA data, refreshed every 3 h)
    GET  /api/forecast?date=     7-day forecast issued on `date` (default: latest NOAA day)
    POST /api/explain            streamed LLM briefing / answer, grounded in the numbers and reports
    /                            the website

Every NOAA or LLM failure degrades gracefully: saved snapshots, cached days, or the
offline briefing - so a demo keeps working even without internet or an API key.
"""

import datetime as dt
import json
import logging
import os
import re
import threading
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

from coralwatch import geo, noaa, stats  # noqa: E402
from .ai import llm, rag  # noqa: E402
from .forecasting import forecast_v1, forecast_v2  # noqa: E402


class _ForecastChooser:
    """Physics-guided v2 ensemble when trained, otherwise the first CNN-ConvLSTM model."""

    @staticmethod
    def available():
        return forecast_v2.available() or forecast_v1.available()

    @staticmethod
    def name():
        if forecast_v2.available():
            return forecast_v2.meta()["label"]
        return forecast_v1.load_model()[1]["name"] if forecast_v1.available() else None

    @staticmethod
    def forecast(issue=None):
        return (forecast_v2 if forecast_v2.available() else forecast_v1).forecast(issue)


forecast_service = _ForecastChooser()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("coralwatch")

WEBSITE = ROOT / "website"
SNAPSHOTS_JS = WEBSITE / "data" / "snapshots.js"
CACHE = Path(__file__).resolve().parent / "data" / "cache"
LATEST_TTL = 3 * 3600
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

app = FastAPI(title="CoralWatch", docs_url="/api/docs")
_grid_cache = {}          # date -> grid
_latest = {"grid": None, "fetched": 0.0, "error": None}
_forecast_cache = {}      # issue date -> forecast
_lock = threading.Lock()


# ── Map grids ─────────────────────────────────────────────────────────
def _load_snapshots():
    if not SNAPSHOTS_JS.exists():
        return {}, None
    text = SNAPSHOTS_JS.read_text()
    latest = json.loads(re.search(r"SNAPSHOT_LATEST = (.*?);", text).group(1))
    body = text[text.index("window.SNAPSHOTS = ") + len("window.SNAPSHOTS = "):].strip().rstrip(";")
    return json.loads(body), latest


SNAPSHOTS, SNAPSHOT_LATEST = _load_snapshots()


def clip_to_park(grid):
    mask = geo.park_mask(grid["lats"], grid["lons"]).ravel()
    for k, vals in grid["vars"].items():
        grid["vars"][k] = [v if m else None for v, m in zip(vals, mask)]
    return grid


def grid_arrays(grid):
    shape = (len(grid["lats"]), len(grid["lons"]))
    return {k: np.array([np.nan if v is None else v for v in vals], float).reshape(shape)
            for k, vals in grid["vars"].items()}


def get_grid(date):
    """Grid for `date` ('latest' or YYYY-MM-DD), with a 'source' field saying where it came from."""
    if date == "latest":
        with _lock:
            fresh = _latest["grid"] and time.time() - _latest["fetched"] < LATEST_TTL
            if not fresh:
                try:
                    g = clip_to_park(noaa.fetch_map_grid("latest", stride=2, timeout=60))
                    g["source"] = "live"
                    _latest.update(grid=g, fetched=time.time(), error=None)
                    _grid_cache[g["date"]] = g
                except Exception as e:
                    log.warning("Live NOAA fetch failed: %s", e)
                    _latest["error"] = str(e)
                    if not _latest["grid"]:
                        if not SNAPSHOT_LATEST:
                            raise HTTPException(503, "NOAA is unreachable and no saved data exists")
                        g = dict(SNAPSHOTS[SNAPSHOT_LATEST], source="snapshot")
                        return g
            return _latest["grid"]

    if date in SNAPSHOTS:
        return dict(SNAPSHOTS[date], source="snapshot")
    if date in _grid_cache:
        return _grid_cache[date]
    disk = CACHE / f"grid_{date}.json"
    if disk.exists():
        g = json.loads(disk.read_text())
        _grid_cache[date] = g
        return g
    try:
        g = clip_to_park(noaa.fetch_map_grid(date, stride=2, timeout=90))
    except Exception as e:
        log.warning("NOAA fetch for %s failed: %s", date, e)
        raise HTTPException(503, f"Couldn't get NOAA data for {date}. Try a saved date.")
    g["source"] = "live"
    _grid_cache[date] = g
    CACHE.mkdir(parents=True, exist_ok=True)
    disk.write_text(json.dumps(g))
    return g


def valid_date(date):
    if date != "latest":
        if not DATE_RE.match(date):
            raise HTTPException(400, "Date must be YYYY-MM-DD or 'latest'")
        d = dt.date.fromisoformat(date)
        if not dt.date(1985, 4, 1) <= d <= dt.date.today():
            raise HTTPException(400, "Date must be between 1985-04-01 and today")
    return date


# ── Forecast ──────────────────────────────────────────────────────────
def get_forecast(issue=None):
    key = issue or "latest"
    cached = _forecast_cache.get(key)
    if cached and (issue or time.time() - cached[1] < LATEST_TTL):
        return cached[0]
    f = forecast_service.forecast(dt.date.fromisoformat(issue) if issue else None)
    _forecast_cache[key] = (f, time.time())
    _forecast_cache[f["issue_date"]] = (f, time.time())
    return f


# ── API ───────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "latest_live": _latest["grid"]["date"] if _latest["grid"] else None,
        "latest_error": _latest["error"],
        "snapshot_dates": sorted(SNAPSHOTS),
        "forecast_model": forecast_service.name(),
        "llm": (lambda pv: {"configured": pv[0] != "offline", "provider": pv[0], "model": pv[1]})(llm.provider()),
        "corpus_passages": len(rag.index().passages),
    }


@app.get("/api/grid/{date}")
def grid(date: str):
    return get_grid(valid_date(date))


@app.get("/api/forecast")
def forecast(date: str | None = None):
    if not forecast_service.available():
        raise HTTPException(503, "No trained forecast model yet (run forecast/train.py)")
    if date:
        valid_date(date)
    try:
        return get_forecast(None if date in (None, "latest") else date)
    except Exception as e:
        log.warning("Forecast failed: %s", e)
        raise HTTPException(503, f"Forecast unavailable: {e}")


class ExplainRequest(BaseModel):
    date: str = "latest"
    zone: str = Field("N", pattern="^(N|C)$")
    question: str | None = Field(None, max_length=500)


def build_facts(date, zone_id):
    g = get_grid(date)
    a = grid_arrays(g)
    zones = stats.zone_stats(g["lats"], g["lons"], a["baa"], dhw=a.get("dhw"), ssta=a.get("ssta"), hotspot=a.get("hotspot"))
    observed = next(z for z in zones if z["zone"] == zone_id)
    facts = {"date": g["date"], "zone": {"id": zone_id, "name": geo.zone_by_id(zone_id)["name"]},
             "data_source": "NOAA Coral Reef Watch 5 km daily v3.1, clipped to the Great Barrier Reef Marine Park",
             "observed": observed, "forecast": None}
    if forecast_service.available():
        try:
            f = get_forecast(g["date"])
            obs_model = next(z for z in f["observed_zones"] if z["zone"] == zone_id)
            observed["dhw_change_7d"] = obs_model.get("dhw_change_7d")
            observed["ssta_change_7d"] = obs_model.get("ssta_change_7d")
            facts["forecast"] = {
                "model": f["model"], "issued": f["issue_date"], "target_date": f["target_date"],
                "zone": next(z for z in f["forecast_zones"] if z["zone"] == zone_id),
                "date_was_in_training_data": f["in_training_data"], "skill": f["skill"],
                "what_is_forecast": f.get("target_description", "NOAA daily alert level"),
            }
            if "forecast_zones_by_lead" in f:
                facts["forecast"]["zone_by_lead_days"] = {
                    k: {kk: z[kk] for kk in ("reefs_at_alert", "worst_level", "mean_alert_probability") if kk in z}
                    for k, zs in f["forecast_zones_by_lead"].items() for z in zs if z["zone"] == zone_id}
                facts["forecast"]["weather_used"] = ("a real weather forecast" if f.get("weather") == "forecast"
                                                     else "observed (reanalysis) weather, since the date is in the past")
        except Exception as e:
            log.warning("No forecast for facts: %s", e)
    return facts


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/explain")
def explain(req: ExplainRequest):
    date = valid_date(req.date)
    facts = build_facts(date, req.zone)
    # Only use what was published before this date. Reports mostly come out mid-year,
    # so a Jan-Jun date may only use reports from the previous year or earlier.
    d = dt.date.fromisoformat(facts["date"])
    max_year = d.year if d.month >= 7 else d.year - 1
    sources = rag.index().search(rag.build_query(req.zone, facts, req.question), k=6, max_year=max_year)

    def stream():
        yield _sse("context", {
            "facts": facts,
            "sources": [{"n": k, "title": rag.pretty_title(s["title"]), "source": s["source"], "published": s.get("published"),
                         "page": s.get("page"), "url": s.get("url"), "snippet": " ".join(s["text"].split()[:45]) + "…"}
                        for k, s in enumerate(sources, 1)],
        })
        try:
            for kind, payload in llm.stream_explanation(facts, sources, req.question):
                yield _sse(kind, {"text": payload} if kind == "delta" else payload)
        except Exception as e:  # never leave the browser hanging
            log.exception("Explain stream failed")
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _refresh_forever():
    """Keep today's NOAA data and forecast warm, so page loads are instant and data stays current."""
    while True:
        try:
            _latest["fetched"] = 0          # force a refresh
            get_grid("latest")
            if forecast_service.available():
                _forecast_cache.pop("latest", None)
                get_forecast(None)
            log.info("Refreshed live NOAA data (%s)", _latest["grid"]["date"] if _latest["grid"] else "unavailable")
        except Exception as e:
            log.warning("Background refresh failed: %s", e)
        time.sleep(LATEST_TTL)


@app.on_event("startup")
def _start_refresher():
    if os.environ.get("CORALWATCH_NO_REFRESH") != "1":
        threading.Thread(target=_refresh_forever, daemon=True).start()


app.mount("/", StaticFiles(directory=WEBSITE, html=True), name="website")


def main():
    import uvicorn
    host = os.environ.get("CORALWATCH_HOST", "127.0.0.1")
    port = int(os.environ.get("CORALWATCH_PORT", "8000"))
    print(f"\n  CoralWatch running at http://{host}:{port}\n")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
