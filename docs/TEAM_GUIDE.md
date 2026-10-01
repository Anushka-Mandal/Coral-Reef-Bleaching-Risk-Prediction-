# CoralWatch: what changed, and how it all fits together

A guide for the team. It explains every addition and change made to the capstone since
the original CNN / NLP / Fusion version, why each one was made, where the code lives and
how to run it. No prior knowledge of the new code is assumed.

> **Short on time?** Read sections 1, 2 and 3. The rest is reference.

Other docs, for depth:
- [RESULTS_EXPLAINED.md](RESULTS_EXPLAINED.md): every number on one page
- [PROJECT_REPORT.md](PROJECT_REPORT.md): full report, phase by phase, for the paper
- [LLM.md](LLM.md): how the AI briefing works and how hallucinations are reduced
- [forecast/README.md](../forecast/README.md), [backend/README.md](../backend/README.md), [website/README.md](../website/README.md)

---

## Contents

1. [The project in one minute](#1-the-project-in-one-minute)
2. [Run it on your laptop](#2-run-it-on-your-laptop)
3. [Old vs new at a glance](#3-old-vs-new-at-a-glance)
4. [Why we changed things: problems in the original](#4-why-we-changed-things-problems-in-the-original)
5. [Every addition, part by part](#5-every-addition-part-by-part)
6. [Changes to existing files](#6-changes-to-existing-files)
7. [Folder map](#7-folder-map)
8. [Results and what to show](#8-results-and-what-to-show)
9. [FAQ](#9-faq)
10. [Glossary](#10-glossary)
11. [Open work](#11-open-work)

---

## 1. The project in one minute

**What it does.** CoralWatch forecasts NOAA's coral bleaching heat-stress alert level
for every reef area (1,927 cells of 0.1°) of the **northern and central Great Barrier
Reef**, 1 to 14 days ahead. It shows today's conditions and the forecast on a live map,
and an AI writes a short briefing that explains the risk and cites reef reports.

**The pieces:**

```
 NOAA satellite data ─┐
 (SST, anomaly,       │      ┌──────────────────────┐      ┌──────────────┐
  HotSpot, DHW)       ├────► │ Forecast model       │ ───► │ Live website │
 Weather forecast ────┤      │ (physics-guided      │      │ map + layers │
 ENSO / MJO indices ──┘      │  CNN-ConvLSTM)       │      └──────┬───────┘
                             └──────────┬───────────┘             │
                                        │ numbers                 │ "Explain"
 Reef reports (AIMS, MMP,               ▼                         ▼
 Reef Authority) ──► 3,180 passages ─► search ─► LLM briefing with citations,
                                                 fact-checked before it's shown
```

**Headline results** (test seasons 2024–2025, never seen during training):

| | Our model | "Same as today" baseline |
|---|---|---|
| 7 days ahead | **82.4%** | 68.7% |
| 7 days ahead, using the weather forecasts actually issued at the time | **76.9%** | 63.8% |
| 14 days ahead | **76.7%** | 50.8% |

These are lower than the old 83.75% / 90.1%, but the old numbers were inflated
(section 4). The new numbers come from a strict, honest test and beat a proper baseline.

---

## 2. Run it on your laptop

**First time only:**

```bash
cd Capstone
./run.sh                       # creates .venv, installs requirements.txt, starts the server
```

**For the free AI briefings** (optional, but needed for the AI panel to write text):

```bash
brew install ollama            # once
ollama pull qwen2.5:3b         # once, about 2 GB
ollama serve                   # keep this running in a separate terminal
```

Then open **http://127.0.0.1:8000**.

| Want to… | Command |
|---|---|
| Start the website + live data | `./run.sh` |
| Forecast in the terminal | `.venv/bin/python -m backend.forecasting.forecast_now` |
| Forecast for a past date | `.venv/bin/python -m backend.forecasting.forecast_now --date 2024-03-03` |
| Run the tests (16, offline) | `.venv/bin/python -m pytest tests -q` |
| Measure AI grounding | `.venv/bin/python -m backend.ai.eval_grounding` |
| Open the site without a server | double-click `website/index.html` (saved dates only, no forecast/AI) |

Nothing needs an API key or a paid account. Everything falls back gracefully: if NOAA is
down the site uses saved days; if the weather service is down the forecast uses physics
only; if no LLM is running the AI panel uses a template briefing.

---

## 3. Old vs new at a glance

| Area | Original version | Now |
|---|---|---|
| Input values | Coloured PNGs averaged to **grey** (+3 °C and −3 °C looked the same) | **Real values** decoded from the PNG colours (error ≈ 0.01) |
| Inputs | 4 image channels, 7 days | 16 channels (ocean + weather + climate indices + position + season), 4 weeks |
| Output | **One** alert level for the whole region | Alert level for **every reef cell**, 1–14 days ahead, plus chance of Alert |
| Target | Daily alert level | NOAA's official **7-day maximum alert** (daily also reported) |
| Model | CNN-LSTM | **PhysNet**: CNN → ConvLSTM → NOAA's DHW formula built in, 3-model ensemble + physics for short leads |
| Evaluation | Test set used to pick the model; no baseline | Train / validate / test by season, baselines, confidence intervals, significance tests, real-forecast test |
| NLP | BGE embeddings fused into the prediction | Report passages **searched** to ground the AI's explanations |
| LLM | None | Free local model (Qwen 2.5 via Ollama) or Claude, with citations and automatic fact-checking |
| Website | None | Landing page, live GBR map, forecast layer, AI panel |
| Code | Colab notebooks | Scripts, a web server, caching, tests, one-command start |

**Is the CNN-LSTM gone?** No. The new model keeps the same idea (CNN reads each day's
map, an LSTM learns how it changes) but was rebuilt so it works on whole maps and real
values. See the FAQ.

---

## 4. Why we changed things: problems in the original

A review of the original code found these issues. Each one is something a reviewer or
examiner could catch, so it had to be fixed before writing a paper.

| # | Problem | How we know | Consequence |
|---|---|---|---|
| 1 | The test set was used to choose the best epoch | `best_val_acc` equals test accuracy in every results file | Accuracy was inflated |
| 2 | No baseline | "Same as 7 days ago" already scores **81.1%** | The 83.75% model added only about 2.6 points |
| 3 | Colours averaged to grey | `gray = arr.mean(axis=2)` in `CNN/SCRIPTS/stack_images.py` | Hot and cold anomalies became identical |
| 4 | Sequences built by file order | Jan–Aug 2023 anomaly images are missing | A "7-day window" could silently span months |
| 5 | One label for the whole region | `CNN/baa_labels.csv` has one value per day | No reef-level map possible |
| 6 | Different tasks in one results table | Some models predicted the same day, others 7 days ahead | Results not comparable |
| 7 | Fusion's 90.1% isn't comparable | 3 classes, same-day target, test-set early stopping | Not a like-for-like gain |
| 8 | NLP added no signal | The fusion's BGE file had only 7 distinct vectors (one per year) | The model saw the same text for every day of a year |
| 9 | Reports describe the past | Annual reports come out after the summer | Using them as inputs leaks the future |

Also found in the literature survey (Chapter 3 / Phase-1 report): author names that don't
match the reference list, conflicting study regions for Yang et al. (2024), and three
references we couldn't verify (González-Rivero 2024, Hensel 2025, Ichikawa 2024). **These
still need fixing by us.**

Also, the teacher's guidance that pre-2018 data is outdated was followed: all training
and testing use 2018–2025 only.

---

## 5. Every addition, part by part

### 5.1 Data

**Recovering real values from our images**
(`forecast/1_data_preparation/recover_from_images.py`)
- Our PNGs in `CNN/2_images/` were painted with fixed colour scales (viridis for SST,
  HotSpot and DHW; red-blue for SST anomaly). Matching each pixel's colour back to its
  scale gives the original number.
- Checked against NOAA's raw files: mean error 0.006 °C (anomaly), 0.010 °C (HotSpot),
  0.018 °C-weeks (DHW), 0.008 °C (SST). The alert level matches NOAA's on 97–99% of cells.
- Output: `forecast/data/cube.npz`, 2,891 days × 4 variables × a 107×106 grid.
- The 240 missing SST-anomaly days of 2023 were rebuilt from SST and the long-term average
  (check error 0.01–0.03 °C).
- **So yes, we still use our images.** They are the data source; we just read them properly.

**Checking NOAA's rules on our data** (results in `forecast/5_results/data_checks/`)
- Alert level (BAA) from HotSpot and DHW: our formula matches NOAA on all 240,555 pixels checked.
- DHW = sum of daily HotSpots ≥ 1 °C over 84 days ÷ 7: correlation with NOAA's DHW 0.996–0.999.

**Weather and climate data** (`forecast/1_data_preparation/download_weather.py`, `coralwatch/weather.py`)
- Wind, cloud, sunlight and air temperature from Open-Meteo (ERA5) at 74 points over the reef, 2017–2025.
- ENSO (Niño 3.4, from NOAA CPC) and MJO (from the Bureau of Meteorology).
- Archived weather *forecasts* (Open-Meteo "previous runs") for a realistic test.
- Output: `forecast/data/weather.npz`.

**Raw NOAA download** (`forecast/1_data_preparation/download_noaa.py`)
- A polite downloader for the raw files. NOAA's servers throttle bulk downloads, so only a
  partial download exists. The image recovery made it unnecessary.

**Reef geography** (`website/build_data.py` → `website/data/gbr.js`)
- Official GBR Marine Park boundary and 2,722 reef outlines from GBRMPA open data.
- Northern / central split at 16.5°S (an approximate division, not an official boundary).

### 5.2 Forecast model

**v1: first corrected model** (`forecast/v1_first_model/`)
- Baselines (persistence, climatology, trend + NOAA rule), gradient boosting and a
  CNN-ConvLSTM on real values, evaluated properly.
- Result: 66.3% vs 65.6% baseline at 7 days (daily target). Honest, but only slightly
  better than the baseline. That led to v2.

**v2: PhysNet, the physics-guided model** (`forecast/2_model/physnet.py`)

```
10 frames (7 daily + 3 weekly averages) × 16 channels
      │
      ▼
CNN encoder (conv → batch-norm → ReLU, keeps the map shape)
      │
      ▼
ConvLSTM (an LSTM over whole maps, so every cell gets its own forecast)
      │
      ├──► HotSpot head: predicts HotSpot for each of the next 14 days
      │          │
      │          ▼
      │    DHW formula layer: NOAA's exact formula, incl. the known drop-off of old days
      │          │
      └──────────┴──► Alert-level head: alert level for leads 1–14 days
```

- **Why physics?** NOAA computes alert levels with a fixed formula from HotSpot and DHW.
  Building that formula into the network means it only has to learn the uncertain part
  (future temperature), not rediscover the formula.
- Loss: cross-entropy (alert level) + ordinal term (being off by 2 levels is worse than
  by 1) + Huber (HotSpot error).
- `forecast/2_model/dataset.py` builds samples calendar-aware (no gaps), applies the season
  split and computes the targets. `forecast/2_model/paths.py` holds every file location.

**Training** (`forecast/3_training/train_v2.py`)
- Data split by season: train 2018–2021, validate 2022–2023, test 2024–2025 (Nov–Apr warm season).
- Experiments (ablations), so we can say what each part contributes:
  `ocean` (sea data only), `past_wx` (+ past weather and indices), `full` (+ weather forecast),
  `no_physics` (formula layer removed), `full_no_airtemp`, `max7`, `max7_noise`.
- The final model averages 3 runs (`max7_s0`, `max7_s1`, `max7_s2`).
- Saved models: `forecast/6_saved_models/v2/`. `meta.json` says which models the live site uses.

**Improvement round** (also in v2)
- **Target changed to NOAA's 7-day maximum alert**: the highest level over the 7 days
  ending on the forecast day. This is NOAA's official product and what reef managers use.
  It is less noisy than the daily level: 77.9% → 82.4%.
- **Hybrid** (`forecast/4_evaluation/hybrid.py`): for 1–4 days ahead, simple physics
  (today's HotSpot + the DHW formula) is better, so it's used there; the network handles
  5–14 days. The cut-off was chosen on validation data only.
- **Training with forecast errors** (`forecast/1_data_preparation/forecast_error.py`): we
  added realistic weather-forecast errors to the training data. It did **not** help, and
  we report that honestly.

### 5.3 Evaluation (`forecast/4_evaluation/`)

| Script | What it measures | Output |
|---|---|---|
| `evaluate_v2.py --target max7` | Accuracy per lead, 95% confidence intervals (moving-block bootstrap, 7-day blocks), paired significance tests vs the baseline, reliability of the probabilities, 2024 case study, ablations | `5_results/v2_max7_final/RESULTS.md`, `results.json`, PNG figures |
| `eval_real_forecasts.py --target max7` | Re-scores Jan–Apr 2024 and 2025 using the weather forecasts **actually issued** at the time, not the weather that later happened | `real_forecast_check.json` |
| `hybrid.py --target max7` | Physics vs network for each lead; picks the switch-over on validation | `hybrid.json` |

All training and evaluation logs are in `forecast/logs/`.

### 5.4 Backend server (`backend/`)

A FastAPI web server (`backend/app.py`) that serves the website and these endpoints:

| Endpoint | What it returns |
|---|---|
| `GET /api/health` | What's working: NOAA, forecast model, which LLM, number of report passages |
| `GET /api/grid/{date}` | Map data for a day. `latest` = today's NOAA data |
| `GET /api/forecast?date=` | Forecast for every reef cell, 1–14 days ahead |
| `POST /api/explain` | The AI briefing, streamed word by word (server-sent events) |

- **Real-time data:** every 3 hours the server fetches the newest NOAA days (112 days of
  history), the 16-day weather forecast and the ENSO/MJO indices, then reruns the forecast.
  Everything is cached in `backend/data/cache/`.
- **Live forecast code:** `backend/forecasting/forecast_v2.py` (current model + hybrid +
  fallbacks), `forecast_v1.py` (older fallback), `forecast_now.py` (terminal tool).
- **Shared library** `coralwatch/`, used by the server and the model:
  `geo.py` (park boundary, zones, reef cells, NOAA alert rule), `noaa.py` (NOAA data
  access + caching), `stats.py` (per-zone counts), `weather.py` (weather + indices).

### 5.5 NLP: from embeddings to search

The old NLP turned reports into BGE embeddings for the fusion model. Since that added no
predictive signal (problem 8) and risks hindsight (problem 9), the reports now serve as
**evidence for explanations**:

- `backend/ai/build_corpus.py`: our 14 report PDFs (from `NLP/nlp_dataset/`) plus NOAA
  product pages, split into **3,180 passages** with page numbers and publication years →
  `backend/data/corpus.json`.
- `backend/ai/rag.py`: BM25 keyword search that finds the passages most relevant to the
  current situation.
  - **No hindsight:** only reports published *before* the date being explained are used.
  - At most 2 passages per report; weakly related passages (below 35% of the best match) are dropped.

The original `NLP/`, `NLP-bge/` and `FUSION/` folders are untouched.

### 5.6 LLM: the AI briefing

When someone clicks **Explain** on the map, the server:

1. Collects the **facts**: today's NOAA numbers for the zone, the forecast for each lead
   (each with its calendar date), and the model's measured accuracy.
2. Finds up to 6 relevant **report passages** (section 5.5).
3. Sends both to the LLM with strict rules: use only these documents, tag every factual
   sentence with its source (`[Data]`, `[S1]`, `[S2]` …), and copy numbers exactly.
4. **Fact-checks** the answer before showing it (below).

**Which LLM?** Chosen automatically (`backend/ai/llm.py`, `provider()`):

| Order | Provider | When | Cost |
|---|---|---|---|
| 1 | **Qwen 2.5 3B via Ollama** (local) | Ollama is running with the model | **Free**, runs on the laptop |
| 2 | Claude (`claude-opus-5`) | `ANTHROPIC_API_KEY` set in `.env` | Paid |
| 3 | Offline template briefing | Neither available | Free, no Q&A |

Force one with `CORALWATCH_LLM=local|claude|offline` in `.env`.

**How hallucinations are reduced** (`backend/ai/grounding.py`)

| Layer | What it does |
|---|---|
| 1. Documents only | The numbers are given as a plain-text "CoralWatch data" document, one fact per sentence, next to the report passages |
| 2. Strict prompt | Tag every factual sentence; copy numbers exactly; say "the data doesn't cover this" instead of guessing; heat stress ≠ bleaching |
| 3. Relevance filter + no hindsight | Weak or future passages are never shown to the model |
| 4. Number check | Every number in the answer must appear in the data or the passages (whole numbers exactly; decimals within rounding) |
| 5. Right number, right day | A sentence with a date and a number must match a source sentence with the same date and number. This catches a real number used for the wrong day, e.g. the 1-day forecast quoted as the 7-day one |
| 6. Citation check | Every factual sentence must carry a source tag (advice under "Suggested actions" is exempt) |

Extra steps for the small local model, which doesn't cite reliably:
- `tidy_local`: fixes headings, removes copied document titles, caps each section at 4 sentences.
- `attribute`: adds a source tag to an untagged sentence only if all its numbers and enough of its words appear in that source.
- **Strict mode** (on by default): sentences that still fail the checks are **deleted**
  before the answer is shown, and the badge says how many.

On the website, a ✓/⚠ badge shows the result of the check, and any unverified number is highlighted.

**Measured** (`backend/ai/eval_grounding.py` → `backend/data/grounding_eval.json`, 4
briefings, local model): 100% of numbers verified, 100% of factual sentences cited, 4/4
fully grounded, 0 sentences removed. About 50–90 s per briefing on an 8 GB Mac.

**Limitation:** the checks guarantee the numbers and the sources, not the quality of the
reasoning. A 3B model writes plainer text than Claude and handles follow-up questions less well.

### 5.7 Website (`website/`)

- **Landing page** (`index.html`, `style.css`, `landing.js`): hero animation, key numbers,
  why it matters, the 5 alert levels, how it works, the dataset chart and the roadmap.
- **Live map** (`app.js`), limited to the GBR Marine Park, northern and central zones:
  - Layers: Bleaching Alert Area, DHW, HotSpot, SST anomaly, SST.
  - Preset dates: today (live) and the peak alert day of each bleaching summer (2016, 2017, 2020, 2022, 2024, 2025). Any date since 1985 loads live from NOAA.
  - Zone cards: reefs at each alert level; hover values.
  - **Forecast layer** with a 1 / 3 / 7 / 14-day lead selector and "chance of Alert".
  - **AI panel**: streams the briefing with source chips, the ✓/⚠ verification badge and the provider name.
- Works without the server too: `data/snapshots.js` holds saved days.

### 5.8 Tests (`tests/test_core.py`)

16 tests, all offline (no internet, no API key):

| Area | Tests |
|---|---|
| NOAA rules | Alert-level rule; DHW drop-off formula |
| Geography | Park mask; zone statistics |
| Search | Publication-date filter; relevance floor |
| LLM | Offline briefing; documents and citations; mocked Claude stream; refusal fallback |
| Fact-checking | Invented numbers and uncited claims flagged; number used for the wrong day caught; local auto-attribution and tidy-up |
| API | Health; grid + date validation; explain stream |

Run them before pushing changes.

### 5.9 Docs and setup files

| File | Purpose |
|---|---|
| `README.md` | Project overview and commands (rewritten) |
| `docs/TEAM_GUIDE.md` | This guide |
| `docs/PROJECT_REPORT.md` | Full report, phases A–H |
| `docs/RESULTS_EXPLAINED.md` | Every number on one page |
| `docs/LLM.md` | How data is fed to the LLM; hallucination layers; setup |
| `forecast/5_results/README.md` | Which result folder is which |
| `run.sh` | One-command setup and start |
| `requirements.txt` | Python packages (tested with Python 3.14) |
| `.env.example` | Template for optional settings (copy to `.env`; never commit `.env`) |

---

## 6. Changes to existing files

**Nothing inside `CNN/`, `NLP/`, `NLP-bge/`, `FUSION/`, `PAPERS/` or `Coll-Reports/` was
modified by this work.** All new code is in new folders.

Changed at the top level:
- `README.md`: rewritten for the new system (the old one is in git history).
- `.gitignore`: also ignores `.venv/`, `.env`, `forecast/data/` (large data) and the caches.

**Heads-up about deleted files.** `git status` shows the old loose files at the repo root as
deleted: experiment notebooks (`reef_fusion_model*.ipynb`, `reef_bge_fusion*.ipynb`,
`reef_minilm_chunked*.ipynb`, `reef_scibert_fusion*.ipynb`, `reef_climatebert_fusion.ipynb`,
`SeasonalEmbed12Jun.ipynb`, `Copy_of_NLP_mar29.ipynb`), scripts (`climatebert_gbr_final.py`,
`coral_nlp_pipeline_fast.py`), `coral_climatebert_gbr_embeddings.csv` and the confusion-matrix
and training-curve PNGs. They're still in the last commit (`13b6b83`). If anyone needs
one back:

```bash
git checkout 13b6b83 -- "reef_fusion_model.ipynb"
```

---

## 7. Folder map

```
Capstone/
├── CNN/  NLP/  NLP-bge/  FUSION/      original work (unchanged)
├── PAPERS/  Coll-Reports/             papers, literature survey, phase reports (unchanged)
│
├── coralwatch/        shared library: geography, NOAA access, zone stats, weather
├── forecast/          research pipeline, numbered in order
│   ├── 1_data_preparation/   images → real values; weather; forecast errors; NOAA downloader
│   ├── 2_model/              physnet.py, dataset.py, paths.py
│   ├── 3_training/           train_v2.py
│   ├── 4_evaluation/         evaluate_v2.py, eval_real_forecasts.py, hybrid.py
│   ├── 5_results/            v2_max7_final/ (HEADLINE), v2_daily_target/, v1_first_corrected_model/, data_checks/
│   ├── 6_saved_models/       v2/ (live models + meta.json), v1/
│   ├── data/                 cube.npz, weather.npz, caches (git-ignored, large)
│   ├── logs/                 every training/evaluation log
│   └── v1_first_model/       first corrected model code
├── backend/           web server + AI
│   ├── app.py                FastAPI server
│   ├── forecasting/          live forecast (v2, v1 fallback, terminal tool)
│   ├── ai/                   corpus, search, LLM, fact-checking, grounding eval
│   └── data/                 corpus.json, grounding_eval.json, cache/
├── website/           landing page + live map + AI panel
├── tests/             16 offline tests
├── docs/              this guide and the other docs
├── run.sh  requirements.txt  .env.example
└── README.md
```

---

## 8. Results and what to show

**Final model** (NOAA 7-day maximum alert, test seasons 2024–2025, 1,927 reef cells):

| Days ahead | Our forecast | Baseline | Gain |
|---|---|---|---|
| 1 | 97.1% | 94.7% | +2.4 |
| 3 | 90.6% | 85.0% | +5.6 |
| 7 | **82.4%** [79.6–84.9] | 68.7% | +13.7 |
| 14 | 76.7% | 50.8% | +25.9 |

- With real archived weather forecasts, 7 days: **76.9%** [73.0–80.7] vs 63.8%. Significant at p < 0.001.
- 7-day extras: Alert F1 0.829, Alert recall 83.1%, 95.0% within one level, Brier skill 0.73.

**How we got here:**

| Stage | 7-day accuracy | Baseline |
|---|---|---|
| Original CNN-LSTM (inflated) | 83.75% | none (a trivial rule got 81.1%) |
| v1 corrected CNN-ConvLSTM, daily target | 66.3% | 65.6% |
| PhysNet, ocean data only | 69.3% | 65.6% |
| + weather forecast, daily target | 77.9% | 65.6% |
| **+ 7-day maximum target (final)** | **82.4%** | **68.7%** |

**Key findings for the paper:**
1. The weather forecast is the biggest single gain (69% → 78%).
2. The DHW formula layer helps most at long leads (14-day macro-F1 0.670 vs 0.625 without it).
3. The gain holds with real forecasts (76.9% vs 63.8%).
4. Report text doesn't improve the prediction, but it is useful for grounded explanations.
5. Colour-coded images can be turned back into exact values, which fixes the grey-image problem.

**To show the mentor:** the live website, then from `forecast/5_results/v2_max7_final/`:
`lead_curve.png` (accuracy vs days ahead), `case_2024.png` (the 2024 event),
`ablation.png` (what each part adds), `reliability.png`, and `RESULTS.md`.

---

## 9. FAQ

**Did we throw away the CNN-LSTM?**
We kept the idea and rebuilt it. Inputs changed from grey 224×224 images to real-valued
107×106 maps with 16 channels, and the output changed from one label to a full map, so the
old weights couldn't be reused (they had also been picked using the test set). The original
notebooks are still in `CNN/4_models/`.

**Are we still working with images?**
Yes. `CNN/2_images/` is the data source. We decode the colours back into real values instead of averaging them to grey.

**Is the NLP still part of the project?**
Yes, in a different role. The reports are searched to give the AI evidence for its
explanations, instead of being fused into the prediction.

**Why is the accuracy lower than before?**
The old numbers were measured on the data used to choose the model, without a baseline,
for one regional label. The new ones are on unseen seasons, for 1,927 cells, against a
baseline. 82.4% vs 68.7% is a real improvement; 83.75% vs 81.1% was not.

**Why 7-day maximum and not the daily level?**
It's NOAA's official alert product, what reef managers act on, and less noisy. The daily
result (77.9% vs 65.6%) is also reported.

**Do we need to pay for anything?**
No. The free local model works. Claude is optional and needs a paid API key.

**Why is the AI slow?**
A 3B model on an 8 GB laptop takes 50–90 s. Close other apps, or use `qwen2.5:7b` on a 16 GB machine (`CORALWATCH_LOCAL_MODEL=qwen2.5:7b`).

**Does the forecast predict bleaching?**
It predicts NOAA's **heat-stress alert**, which is the standard early warning for
bleaching, not observed bleaching itself. Checking against the 2020 aerial surveys is open work.

**Where's the data? `forecast/data/` is empty after cloning.**
It's git-ignored because it's large. Rebuild it with
`forecast/1_data_preparation/recover_from_images.py` and `download_weather.py`, or copy it from a teammate.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **SST** | Sea surface temperature (°C) |
| **SST anomaly (SSTA)** | SST minus the long-term average for that day |
| **HotSpot (HS)** | How far SST is above the warmest-month average (°C) |
| **DHW** | Degree Heating Weeks: heat stress built up over 12 weeks, from HotSpots ≥ 1 °C (°C-weeks) |
| **BAA** | NOAA's Bleaching Alert Area level: 0 No Stress, 1 Watch, 2 Warning, 3 Alert Level 1 (DHW ≥ 4), 4 Alert Level 2 (DHW ≥ 8) |
| **7-day maximum (max7)** | Highest BAA level over the 7 days ending on a date. NOAA's official product |
| **Lead** | How many days ahead the forecast is |
| **Persistence / baseline** | "Same as today": the forecast to beat |
| **ConvLSTM** | An LSTM that works on maps, keeping the spatial layout |
| **PhysNet** | Our physics-guided CNN-ConvLSTM with NOAA's DHW formula built in |
| **Ensemble** | Average of several trained models (we use 3) |
| **Hybrid** | Physics for 1–4 days ahead, the network for 5–14 |
| **Ablation** | Removing one part to measure its contribution |
| **Bootstrap CI** | A confidence interval made by resampling the test days |
| **ERA5 / Open-Meteo** | Weather reanalysis data / the free weather API we use |
| **ENSO / MJO** | Large climate patterns (El Niño; a tropical rain/wind cycle) that affect reef heat |
| **BM25** | A classic keyword-search ranking method |
| **RAG** | Retrieval-augmented generation: finding passages and giving them to the LLM |
| **Ollama** | Free tool that runs open-source LLMs on your laptop |
| **Grounded** | Every number and claim in the AI answer traces back to a source |

---

## 11. Open work

Ideas for who can pick up what:

| Task | Why | Effort |
|---|---|---|
| Fix the literature-survey citations (section 4) | Needed before any submission | Small |
| Validate against the 2020 aerial bleaching surveys (Zenodo record 13637906) | Shows the alerts relate to real bleaching | Medium |
| Draft the paper (target: IEEE InGARSS) | Results are ready | Large |
| Spot-check recovered values against more raw NOAA files | Strengthens the data section | Small |
| Compare with NOAA's / BoM's official outlooks | Reviewers will ask | Medium |
| Add subsurface ocean data (Copernicus, free account) | Possible accuracy gain | Medium |
| Test the Claude path with a real API key | Only tested with mocks so far | Small (needs a key) |
| Commit the new work to git | Nothing new is committed yet | Small |

Known limitations to state in the paper: only 2 test seasons and one region; fixed NOAA
thresholds although coral heat tolerance may change; no subsurface data; the AI checks
guarantee numbers and sources, not reasoning quality.
