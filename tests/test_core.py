"""Offline tests for CoralWatch (no internet or API key needed).

    .venv/bin/python -m pytest tests -q
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("CORALWATCH_NO_REFRESH", "1")
sys.path.insert(0, str(ROOT))

from coralwatch import geo, stats  # noqa: E402


def test_baa_rule_matches_noaa_definition():
    hs = np.array([-0.2, 0.0, 0.5, 1.0, 1.5, 2.0, 1.2, np.nan])
    dhw = np.array([0.0, 3.0, 5.0, 0.5, 4.0, 8.0, 7.99, 3.0])
    out = geo.baa_from(hs, dhw)
    assert out[:7].tolist() == [0, 0, 1, 2, 3, 4, 3]
    assert np.isnan(out[7])


def test_park_mask_contains_known_reef_and_excludes_land():
    lats = np.array([-16.7, -19.5])     # Cairns reefs / inland Queensland
    lons = np.array([146.0, 144.0])
    mask = geo.park_mask(lats, lons)
    assert mask[0, 0]                   # offshore Cairns - inside the park
    assert not mask[1, 1]               # inland - outside


def test_zone_stats_counts_levels_and_reefs():
    lats = np.round(np.arange(-10.425, -21.0, -0.1), 3)
    lons = np.round(np.arange(142.325, 152.9, 0.1), 3)
    park = geo.park_mask(lats, lons)
    baa = np.where(park, 4.0, np.nan)   # everything at Alert Level 2
    out = stats.zone_stats(lats, lons, baa, dhw=np.where(park, 9.0, np.nan))
    for z in out:
        assert z["worst_level"] == "Alert Level 2"
        assert z["area_share_by_level"]["Alert Level 2"] == 1.0
        assert z["reefs_at_alert"] == z["reefs_total"] > 100
        assert z["dhw_max"] == 9.0


def test_rag_respects_publication_year():
    from backend.ai import rag
    idx = rag.index()
    if not idx.passages:
        pytest.skip("corpus not built")
    hits = idx.search("coral bleaching heat stress mass bleaching", k=6, max_year=2019)
    assert hits and all(h["published"] is None or h["published"] <= 2019 for h in hits)
    titles = [h["title"] for h in hits]
    assert max(titles.count(t) for t in titles) <= 2   # at most two passages per report


def test_offline_briefing_uses_facts():
    from backend.ai import llm
    facts = {"date": "2024-03-03", "zone": {"id": "N", "name": "Northern GBR"},
             "observed": {"reefs_at_alert": 472, "reefs_total": 1173, "worst_level": "Alert Level 2",
                          "dhw_max": 11.8, "dhw_change_7d": 1.2}, "forecast": None}
    text = llm.offline_briefing(facts, [])
    assert "472 of 1173" in text and "11.8" in text and "### Suggested actions" in text


def test_documents_carry_facts_and_sources_with_citations():
    from backend.ai import llm
    facts = {"date": "2024-03-03", "zone": {"id": "C", "name": "Central GBR"},
             "observed": {"reefs_at_alert": 737, "reefs_total": 1424, "worst_level": "Alert Level 2"}, "forecast": None}
    src = [{"source": "AIMS", "title": "Report", "published": 2023, "page": 4, "text": "Heat stress..."}]
    content = llm.build_content(facts, src)
    docs = [c for c in content if c["type"] == "document"]
    assert len(docs) == 2 and all(d["citations"]["enabled"] for d in docs)
    assert "737 of 1424 reefs in the Central GBR" in docs[0]["source"]["data"]
    assert docs[1]["title"].startswith("AIMS - Report (2023), p. 4")
    assert content[-1]["type"] == "text" and "Central GBR" in content[-1]["text"]


def test_verify_flags_invented_numbers_and_uncited_claims():
    from backend.ai import grounding
    facts = "On 2024-03-03, 472 of 1173 reefs were at Alert. 41.1% of the area was at Alert Level 1. DHW maximum 11.8."
    good = "### What the satellites show\n472 of 1,173 reefs were at Alert, with DHW up to 11.8 and 41% of the area at Alert Level 1. [Data]"
    r = grounding.verify(good, facts, [])
    assert r["grounded"] and r["numbers_checked"] == 4 and not r["unverified_numbers"]
    bad = "About 650 reefs bleached last week. [Data] Mortality reached 37% in 2016."
    r = grounding.verify(bad, facts, [])
    assert not r["grounded"]
    assert "650" in r["unverified_numbers"] and "37%" in r["unverified_numbers"]
    assert any("Mortality" in c for c in r["uncited_claims"])


def test_rag_drops_weakly_related_passages():
    from backend.ai.rag import BM25Index
    idx = BM25Index([{"title": "a", "text": "coral bleaching heat stress degree heating weeks", "published": 2020},
                     {"title": "b", "text": "water quality nitrogen sampling coral", "published": 2020}])
    hits = idx.search("bleaching heat stress degree heating weeks", k=6)
    assert [h["title"] for h in hits] == ["a"]


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from backend.app import app
    return TestClient(app)


def test_api_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_grid_snapshot_and_validation(client):
    from backend.app import SNAPSHOTS
    if not SNAPSHOTS:
        pytest.skip("no snapshots")
    day = sorted(SNAPSHOTS)[0]
    r = client.get(f"/api/grid/{day}")
    assert r.status_code == 200 and r.json()["date"] == day
    assert client.get("/api/grid/not-a-date").status_code == 400
    assert client.get("/api/grid/1970-01-01").status_code == 400


def test_api_explain_streams_offline(client, monkeypatch):
    from backend import app as appmod
    from backend.app import SNAPSHOTS
    if not SNAPSHOTS:
        pytest.skip("no snapshots")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("CORALWATCH_LLM", "offline")      # don't pick up a running Ollama
    monkeypatch.setattr(appmod.forecast_service, "available", lambda: False)
    day = sorted(SNAPSHOTS)[0]
    r = client.post("/api/explain", json={"date": day, "zone": "N"})
    assert r.status_code == 200
    events = [line[7:] for line in r.text.splitlines() if line.startswith("event: ")]
    assert events[0] == "context" and "delta" in events and "verify" in events and events[-1] == "done"
    done = json.loads(r.text.strip().splitlines()[-1][6:])
    assert done["mode"] == "offline"
    assert client.post("/api/explain", json={"date": day, "zone": "X"}).status_code == 422


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _events(chunks, cites=()):
    """Raw stream events: one text block; citations (document indexes) attached to it."""
    ev = [_Obj(type="content_block_start")]
    ev += [_Obj(type="content_block_delta", delta=_Obj(type="text_delta", text=c)) for c in chunks]
    ev += [_Obj(type="content_block_delta", delta=_Obj(type="citations_delta", citation=_Obj(document_index=i))) for i in cites]
    ev += [_Obj(type="content_block_stop")]
    return ev


class _FakeStream:
    def __init__(self, events, stop_reason="end_turn"):
        self._events = events
        self._stop = stop_reason

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        class U: input_tokens, output_tokens = 1200, 180
        class M: pass
        m = M(); m.stop_reason = self._stop; m.model = "claude-opus-5"; m.usage = U()
        return m


def _fake_client(events, stop_reason="end_turn", seen=None):
    class Messages:
        def stream(self, **kw):
            if seen is not None:
                seen.update(kw)
            return _FakeStream(events, stop_reason)
    class Beta: messages = Messages()
    class Client:
        def __init__(self, **kw): self.beta = Beta()
    return Client


def test_llm_stream_path_with_mocked_claude(monkeypatch):
    import anthropic
    from backend.ai import llm
    seen = {}
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    events = _events(["### What the satellites show\n", "472 of 1173 reefs are at Alert."], cites=(0, 0, 1))
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client(events, seen=seen))
    facts = {"date": "2024-03-03", "zone": {"id": "N", "name": "Northern GBR"},
             "observed": {"reefs_at_alert": 472, "reefs_total": 1173}, "forecast": None}
    out = list(llm.stream_explanation(facts, [{"source": "AIMS", "title": "R", "published": 2023, "page": 1, "text": "t"}]))
    text = "".join(p for k, p in out if k == "delta")
    assert text == "### What the satellites show\n472 of 1173 reefs are at Alert.[Data][S1]"   # markers after the cited block
    verify = next(p for k, p in out if k == "verify")
    assert verify["grounded"] and verify["numbers_verified"] == 2
    assert out[-1] == ("done", {"mode": "llm", "model": "claude-opus-5", "input_tokens": 1200, "output_tokens": 180})
    assert seen["model"] == "claude-opus-5" and seen["fallbacks"] == "default"
    docs = [c for c in seen["messages"][0]["content"] if c["type"] == "document"]
    assert len(docs) == 2 and docs[0]["citations"] == {"enabled": True}


def test_llm_refusal_before_output_falls_back_offline(monkeypatch):
    import anthropic
    from backend.ai import llm
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(anthropic, "Anthropic", _fake_client([], stop_reason="refusal"))  # no events = no output
    facts = {"date": "2024-03-03", "zone": {"id": "N", "name": "Northern GBR"}, "observed": {}, "forecast": None}
    events = list(llm.stream_explanation(facts, []))
    assert events[-1][0] == "done" and events[-1][1]["mode"] == "offline"


def test_dhw_dropoff_matches_noaa_formula():
    """With HotSpot fixed at 1.5, the known DHW drop-off for lead k is 1.5*k/7,
    and DHW(t+k) = DHW(t) + new hot spots - drop-off stays at 84*1.5/7 = 18."""
    import datetime as dt
    sys.path.insert(0, str(ROOT / "forecast" / "2_model"))
    from physnet import Builder, Config

    class TinyCube:
        pass
    c = TinyCube()
    n = 120
    c.dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    c.index = {d: i for i, d in enumerate(c.dates)}
    c.valid = np.ones((3, 3), bool)
    c.lats, c.lons = np.array([-15.0, -15.1, -15.2]), np.array([146.0, 146.1, 146.2])
    c.x = np.zeros((n, 4, 3, 3), np.float16)
    c.x[:, 1] = 1.5                        # HotSpot
    c.x[:, 2] = 84 * 1.5 / 7               # DHW at steady state
    c.x[:, 3] = 29.0
    c.baa = geo.baa_from(c.x[:, 1].astype(float), c.x[:, 2].astype(float)).astype(np.float32)
    b = Builder(c, None, Config(weather_past=False, weather_future=False, indices=False))
    t = 100
    drop = b.dropoff(t)
    for k in (1, 7, 14):
        assert np.allclose(drop[k - 1], 1.5 * k / 7, atol=1e-2)
        dhw_future = c.x[t, 2].astype(float) + k * 1.5 / 7 - drop[k - 1]
        assert np.allclose(dhw_future, 18.0, atol=1e-2)


def test_context_check_catches_number_quoted_for_wrong_day():
    from backend.ai import grounding
    facts = ("For 2024-03-04 (1 day ahead), the model forecasts 837 reefs in the Central GBR at Alert Level 1 or higher.\n"
             "For 2024-03-10 (7 days ahead), the model forecasts 1010 reefs in the Central GBR at Alert Level 1 or higher.")
    wrong = "The model forecasts 837 reefs at Alert Level 1 or higher on 2024-03-10. [Data]"
    right = "The model forecasts 1010 reefs at Alert Level 1 or higher on 2024-03-10. [Data]"
    r = grounding.verify(wrong, facts, [])
    assert r["unverified_numbers"] == [] and r["context_mismatches"] and not r["grounded"]
    assert grounding.remove_unsupported(wrong, r)[1] == 1
    assert grounding.verify(right, facts, [])["grounded"]


def test_local_attribution_and_tidy():
    from backend.ai import grounding
    docs = [("Data", "On 2024-03-03, 412 reefs in the Northern GBR were at Alert Level 1 or higher."),
            ("S1", "The 2016 marine heatwave caused severe bleaching on northern reefs.")]
    text, n = grounding.attribute("On 2024-03-03, 412 reefs were at Alert Level 1 or higher.\nCorals on Mars glow 99 times brighter.", docs)
    assert n == 1 and "412 reefs were at Alert Level 1 or higher. [Data]" in text
    assert "Mars" in text and "Mars glow 99 times brighter. [" not in text   # unsupported stays untagged
    tidy = grounding.tidy_local("## satellites show\nA. B. C. D. E. F.\n## actions\nDo X.")
    assert "### What the satellites show" in tidy and "### Suggested actions" in tidy
    assert "E." not in tidy                                                   # capped at 4 sentences
