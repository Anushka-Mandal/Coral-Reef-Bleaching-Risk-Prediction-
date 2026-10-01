"""
eval_grounding.py
=================
Measures how grounded the AI briefings are, across many dates and both zones: a
hallucination rate you can report in the paper. Needs ANTHROPIC_API_KEY (otherwise it
scores the offline briefing, which is grounded by construction).

    .venv/bin/python -m backend.ai.eval_grounding                    # default dates
    .venv/bin/python -m backend.ai.eval_grounding --dates 2024-03-03 2022-03-12

For each briefing it records the fact-check report (numbers verified, factual passages
cited) and writes backend/data/grounding_eval.json with the totals:
    number accuracy   = verified numbers / all checked numbers
    citation coverage = cited factual passages / all factual passages
    fully grounded    = share of briefings with no unverified number and no uncited claim
"""

import argparse
import datetime as dt
import json
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from ..app import build_facts  # noqa: E402
from . import llm, rag  # noqa: E402

DEFAULT_DATES = ["2016-03-23", "2017-03-21", "2020-03-08", "2022-03-12", "2024-03-03", "2025-03-09", "latest"]
QUESTIONS = [None, "How does this compare with earlier bleaching years?", "Which areas are most at risk next week?"]
OUT = ROOT / "backend" / "data" / "grounding_eval.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=DEFAULT_DATES)
    ap.add_argument("--questions", action="store_true", help="also ask the example questions")
    args = ap.parse_args()

    rows = []
    for date in args.dates:
        for zone in ("N", "C"):
            facts = build_facts(date, zone)
            d = dt.date.fromisoformat(facts["date"])
            max_year = d.year if d.month >= 7 else d.year - 1
            for q in (QUESTIONS if args.questions else [None]):
                sources = rag.index().search(rag.build_query(zone, facts, q), k=6, max_year=max_year)
                report, info = None, None
                for kind, payload in llm.stream_explanation(facts, sources, q):
                    if kind == "verify":
                        report = payload
                    elif kind == "done":
                        info = payload
                rows.append({"date": facts["date"], "zone": zone, "question": q, "mode": info["mode"], **report})
                print(f"{facts['date']} {zone} {'briefing' if q is None else 'Q'}: numbers {report['numbers_verified']}/{report['numbers_checked']}, "
                      f"cited {report['cited_factual_sentences']}/{report['factual_sentences']}, grounded={report['grounded']} ({info['mode']})", flush=True)

    tot = lambda k: sum(r[k] for r in rows)
    summary = {
        "briefings": len(rows),
        "mode": sorted({r["mode"] for r in rows}),
        "number_accuracy": round(tot("numbers_verified") / max(1, tot("numbers_checked")), 4),
        "citation_coverage": round(tot("cited_factual_sentences") / max(1, tot("factual_sentences")), 4),
        "fully_grounded_share": round(sum(r["grounded"] for r in rows) / max(1, len(rows)), 4),
        "unverified_numbers": sorted({n for r in rows for n in r["unverified_numbers"]}),
    }
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("\n" + json.dumps(summary, indent=1))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
