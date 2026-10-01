"""Retrieval over the reef-report corpus (backend/data/corpus.json).

BM25 keyword ranking - fast, dependency-free and easy to explain. Two rules keep it
honest for past dates: only passages published up to the year being explained are
eligible, and at most two passages come from any one report so the LLM sees
several sources.
"""

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus.json"

STOPWORDS = set("""a an and are as at be been but by for from has have in into is it its of on or
that the their there these this to was were which with will within also than more most such
can may our we per not""".split())

ZONE_TERMS = {
    "N": "northern far northern cape york cooktown lizard island torres strait port douglas",
    "C": "central cairns townsville whitsunday innisfail mackay",
}


TITLE_FIXES = {
    "past-reef-health-updates": "Reef Authority - past Reef health updates",
    "reef-health-updates": "Reef Authority - Reef health updates",
}


def pretty_title(title):
    """Readable report name from a PDF file name."""
    if title in TITLE_FIXES:
        return TITLE_FIXES[title]
    t = re.sub(r"[_]+", " ", title)
    t = re.sub(r"\b(final|FINAL|web|v\d|F\d|\(\d\))\b", "", t)
    t = re.sub(r"\b\d{1,2}(January|February|March|April|May|June|July|August|September|October|November|December)\d{4}\b", "", t)
    t = re.sub(r"\b(\d{6,}\w*|[A-Z][a-z]{2}\d{2})\b", "", t)             # dates like 040822F3, Aug21
    t = re.sub(r"(20\d{2})[ _-](20\d{2}|\d{2})\b", r"\1-\2", t)          # 2022 2023 -> 2022-2023
    t = re.sub(r"\bGBR\b", "GBR", re.sub(r"\s+", " ", t)).strip(" -")
    return t


def tokenize(text):
    return [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if w not in STOPWORDS and len(w) > 1]


class BM25Index:
    def __init__(self, passages, k1=1.5, b=0.75):
        self.passages = passages
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(p["text"] + " " + p["title"])) for p in passages]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(passages)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query, k=6, max_year=None, per_doc=2, min_rel=0.35):
        """Top passages for the query. Passages scoring below min_rel x the best score are
        dropped: weakly related text invites the LLM to speculate."""
        q = tokenize(query)
        scored = []
        for i, tf in enumerate(self.tfs):
            p = self.passages[i]
            if max_year is not None and p.get("published") is not None and p["published"] > max_year:
                continue
            s = 0.0
            for t in q:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        floor = scored[0][0] * min_rel if scored else 0
        out, used = [], Counter()
        for s, i in scored:
            if s < floor:
                break
            p = self.passages[i]
            if used[p["title"]] >= per_doc:
                continue
            used[p["title"]] += 1
            out.append({**p, "score": round(s, 2)})
            if len(out) == k:
                break
        return out


@lru_cache(maxsize=1)
def index():
    passages = json.loads(CORPUS.read_text()) if CORPUS.exists() else []
    return BM25Index(passages)


def build_query(zone_id, facts, question=None):
    """Retrieval query from the situation: zone names, stress level words, season, question."""
    parts = ["coral bleaching heat stress sea surface temperature degree heating weeks"]
    parts.append(ZONE_TERMS.get(zone_id, "northern central great barrier reef"))
    worst = (facts.get("observed") or {}).get("worst_level") or ""
    if "Alert" in worst:
        parts.append("mass bleaching severe alert mortality aerial survey coral cover decline")
    elif worst in ("Watch", "Warning"):
        parts.append("marine heatwave warming watch early warning")
    else:
        parts.append("recovery coral cover monitoring")
    if question:
        parts.append(question)
    return " ".join(parts)
