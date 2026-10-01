# How the LLM gets its data

CoralWatch uses an LLM (Claude Opus 5 through the Anthropic API) to **explain** the
risk, not to predict it. The prediction comes from the trained forecast model, which
can be scored against the truth. The LLM turns numbers and reports into a cited,
plain-language briefing.

Code: [`backend/app.py`](../backend/app.py) (`build_facts`, `/api/explain`),
[`backend/ai/rag.py`](../backend/ai/rag.py) (report search),
[`backend/ai/llm.py`](../backend/ai/llm.py) (prompt and streaming).

## The pipeline for one briefing

```
 user picks a date + zone on the website
            │
            ▼
 1. FACTS (numbers)                        2. SOURCES (text)
    NOAA map data for that day               BM25 search over 3,180 report passages
    → reefs at each alert level              (AIMS, MMP, Reef Authority, NOAA pages)
    → DHW, HotSpot, anomaly (mean/max)       → query built from zone + stress level
    last-7-day change in DHW/anomaly         → only reports published before the date
    7-day forecast from the trained model    → max 2 passages per report, top 6
    + the model's measured test skill
            │                                         │
            └──────────────┬──────────────────────────┘
                           ▼
 3. PROMPT  system: role + rules (use only FACTS/SOURCES, cite [S#], no invented numbers)
            user:   FACTS as JSON + numbered SOURCES + the task or question
                           │
                           ▼
 4. Claude streams the answer → server relays it to the browser word by word
    (no API key / API error → a rule-based offline briefing from the same FACTS)
```

## 1. Facts: the numbers

Built by `build_facts()` from the day's NOAA grid, clipped to the Marine Park:

```json
{
  "date": "2024-03-03",
  "zone": {"id": "N", "name": "Northern GBR"},
  "observed": {
    "reefs_total": 1173, "reefs_at_alert": 472,
    "area_share_by_level": {"No Stress": 0.0, "Watch": 0.101, "Warning": 0.211,
                            "Alert Level 1": 0.411, "Alert Level 2": 0.277},
    "worst_level": "Alert Level 2",
    "dhw_mean": 6.28, "dhw_max": 11.8, "ssta_mean": 1.63, "hotspot_max": 1.89,
    "dhw_change_7d": 1.1
  },
  "forecast": {
    "model": "PhysNet + weather forecast", "issued": "2024-03-03", "target_date": "2024-03-10",
    "zone": {"reefs_at_alert": 246, "worst_level": "Alert Level 2", "mean_alert_probability": 0.40},
    "zone_by_lead_days": {"1": {"reefs_at_alert": 531}, "3": {"reefs_at_alert": 469},
                          "7": {"reefs_at_alert": 246}, "14": {"reefs_at_alert": 0}},
    "weather_used": "observed (reanalysis) weather, since the date is in the past",
    "date_was_in_training_data": false,
    "skill": {"model_test_7d": {"accuracy": 0.779, "macro_f1": 0.725, "alert_f1": 0.717},
              "persistence_test_7d": {"accuracy": 0.656, "macro_f1": 0.562, "alert_f1": 0.565}}
  }
}
}
```

The model's skill is included on purpose, so the LLM can tell the reader how far
to trust the forecast.

## 2. Sources: the reports

[`backend/ai/build_corpus.py`](../backend/ai/build_corpus.py) reads the PDFs in
`NLP/nlp_dataset/*.zip`, splits them into ~220-word passages with page numbers,
and records when each report was published. It also adds NOAA's product pages
(what DHW, HotSpot and BAA mean). Run it again whenever you add reports.

At question time [`rag.py`](../backend/ai/rag.py) ranks passages with BM25, keeping
only reports **published before the date being explained**. Reports come out
mid-year, so for a January–June date only the previous year's reports count. An
explanation of March 2024 can't quote the report that later described March 2024.

## 3. The prompt

The system prompt (in `backend/ai/llm.py`) sets strict grounding rules:
- say only what the documents support, and cite it
- copy numbers exactly, with no calculating or estimating
- say "the data and reports provided don't cover this" instead of guessing
- never claim bleaching happened unless a report says so, and name the year it refers to
- state the forecast's uncertainty
- use four fixed headings

The user message is a list of **documents** followed by the task:

```
[document] CoralWatch data: Northern GBR, 2024-03-03       <- the facts as plain sentences, e.g.
           "On 2024-03-03, 472 of 1173 reefs in the Northern GBR were at Bleaching Alert Level 1 or higher."
[document] AIMS Long-Term Monitoring Program - AIMS LTMP Report GBR coral status 2022-2023 (2023), p. 5
[document] …                                               <- up to 6 report passages
[text]     Write the briefing for the Northern GBR as of 2024-03-03.
```

Every document has `"citations": {"enabled": true}`. For the question box, the task becomes
the user's question.

## 4. The API call

```python
client.beta.messages.stream(
    model="claude-opus-5",
    max_tokens=4000,
    system=SYSTEM_PROMPT,
    thinking={"type": "adaptive"},
    output_config={"effort": "medium"},
    betas=["server-side-fallback-2026-07-01"],
    fallbacks="default",          # if the model declines, the API retries on a fallback model
    messages=[{"role": "user", "content": build_content(facts, sources, question)}],
)
```

The answer streams to the browser as Server-Sent Events:
- `context`: the sources, shown in the right-hand column
- `delta`: text, plus `[Data]` / `[S#]` citation markers
- `verify`: the fact-check report
- `done`: the final event

## How hallucinations are reduced

Six layers, from prevention to detection (code: `backend/ai/`):

| Layer | What it does | Where |
|---|---|---|
| 1. Documents + Citations API | The numbers are sent as a plain-text **"CoralWatch data" document** (one fact per sentence) next to the report passages, all with `citations: {enabled: true}`. Claude's citations are guaranteed to point to real text in these documents, and the site shows them as `[Data]` / `[S1]`… chips after each passage | `grounding.facts_to_text`, `llm.build_content` |
| 2. Strict grounding prompt | Every sentence with a number, date, trend, bleaching or forecast claim must be cited. Numbers must be copied exactly (no calculating or rounding). "The data and reports provided don't cover this" instead of guessing. Heat stress is not bleaching; say which year a report refers to | `llm.SYSTEM_PROMPT` |
| 3. Relevance filter + no hindsight | Only passages scoring at least 35% of the best match are passed in, since weakly related text invites speculation. Only reports published before the date are allowed | `rag.search(min_rel=0.35)`, `app.explain` |
| 4. Automatic fact-check | After every answer: each number (counts, decimals, percentages, years) must match the data or the passages (whole numbers exactly; decimals within rounding), and every factual passage must carry a citation (advice under "Suggested actions" is exempt). The site shows a ✓/⚠ badge and **highlights** any number it couldn't verify | `grounding.verify` |
| 5. Right number, right day | A sentence that pairs a date with a number must match one data/report sentence containing that same date and number. This catches a real number used in the wrong context, e.g. the 1-day forecast quoted as the 7-day forecast (each forecast lead is written with its own calendar date) | `grounding.context_mismatches` |
| 6. Measurement | Runs briefings across dates and zones and reports number accuracy, citation coverage and the share of fully grounded briefings, a hallucination rate for the paper | `python -m backend.ai.eval_grounding` → `backend/data/grounding_eval.json` |

Citations stream as `citations_delta` events. The server adds the markers at the end of
each cited text block and passes the exact block boundaries to the fact-check.

Note: citations can't be combined with structured outputs (`output_config.format`), so
answers are free text checked after generation.

## Turning it on

The server picks the model automatically, in this order:

| Provider | When | Cost |
|---|---|---|
| **Local (free)**: `qwen2.5:3b` via Ollama | Ollama is running and has the model | Free, runs on the laptop, no internet needed |
| Claude (`claude-opus-5`) | `ANTHROPIC_API_KEY` is set in `.env` | Paid API |
| Offline briefing | neither is available | Free, template text, no Q&A |

Force one with `CORALWATCH_LLM=local|claude|offline`. `/api/health` shows which one is active.

### Free local model (recommended for the demo)

```bash
brew install ollama            # once
ollama pull qwen2.5:3b         # once, ~2 GB
ollama serve                   # keep this running in its own terminal
./run.sh                       # open http://127.0.0.1:8000
```

Use a different model with `CORALWATCH_LOCAL_MODEL=llama3.2:3b` (or `qwen2.5:7b` on a
16 GB machine). Small models don't cite reliably, so the local path adds extra steps
after generation (`llm._stream_local`):

1. `tidy_local`: exact section headings, copied document titles removed, max 4 sentences per section
2. `attribute`: each untagged factual sentence gets a `[Data]` / `[S#]` tag only if every
   number in it and at least 35% of its key words appear in that document
3. `verify` (layers 4 and 5 above)
4. **strict mode** (`CORALWATCH_STRICT=1`, default): sentences that still fail are deleted
   before the answer is shown; the badge says how many were removed

Measured on an 8 GB Mac (`eval_grounding --dates 2024-03-03 2022-03-12`, 4 briefings):
100% of numbers verified, 100% of factual sentences cited, 4/4 fully grounded, 0 sentences
removed. Each briefing takes about 50–90 s. Limitation: a 3B model writes plainer text than
Claude and handles follow-up questions less well; the checks guarantee the numbers and
sources, not the quality of the reasoning.

### Claude (paid)

1. Get an API key from the Anthropic Console.
2. `cp .env.example .env` and set `ANTHROPIC_API_KEY=...`
3. `./run.sh` and open http://127.0.0.1:8000

## Why not feed the reports into the prediction?

We tried fusing BGE report embeddings with the CNN (see `FUSION/`). It didn't help,
and our review found why: the reports are yearly and published after the event,
so at daily resolution they carry no forward-looking signal, and using them risks
leaking the answer. Retrieval-grounded explanation is the right job for text.
