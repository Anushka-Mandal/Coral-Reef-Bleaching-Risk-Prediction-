# CoralWatch: complete project report

**Coral Reef Bleaching Risk Prediction using Satellite Alerts and NLP on Observational Reports**
Region: northern and central Great Barrier Reef (GBR) · Data years: 2018–2025

This document records everything done in the project, from the original version to now:
what was changed and why, the complete workflow, what each folder contains, the results,
and what is still open.

---

## 1. Summary

**Goal.** Forecast coral bleaching heat-stress alerts, days ahead, for every reef area of
the northern and central GBR, show them on a live map, and explain the risk in plain
language using reef monitoring reports.

**Final results** (unseen 2024–2025 test seasons, 1,927 reef cells, 95% block-bootstrap confidence intervals):

| Target | 1 day | 3 days | 7 days | 14 days |
|---|---|---|---|---|
| NOAA 7-day maximum alert, hybrid (headline) | **97.1%** | **90.6%** | **82.4%** [79.6–84.9] | **76.7%** [73.5–79.6] |
| "Same as today" baseline | 94.7% | 85.0% | 68.7% | 50.8% |
| Daily alert level, hybrid | 87.5% | 79.7% | 77.9% [74.4–81.3] | 74.2% |
| "Same as today" baseline | 87.1% | 77.0% | 65.6% | 51.3% |

With the **weather forecasts actually issued at the time** (Jan–Apr 2024 and 2025 peaks,
7 days ahead): **76.9%** [73.0–80.7] vs 63.8% for the baseline (7-day maximum target), and **70.9%**
vs 57.4% (daily target). Both are significant at p < 0.001.

The original version reported 83.75% (CNN-LSTM) and 90.1% (fusion), but those numbers were
inflated (see section 3). The new numbers come from a stricter, harder and honest evaluation.

---

## 2. The original project (starting point)

```
NOAA data 2018–2025 (SST, SST anomaly, HotSpot, DHW)
   → coloured PNG maps (viridis / RdBu_r)                     CNN/SCRIPTS/nc_to_png.py
   → colours averaged to grey, resized to 224×224, stacked    CNN/SCRIPTS/stack_images.py
   → CNN-LSTM / CNN-GRU / ConvLSTM, one alert level for the whole region, 7 days ahead
     best: CNN-LSTM with dropout, reported 83.75%              CNN/4_models, CNN/5_results
NLP: BGE-large embeddings of AIMS, MMP and Reef health reports NLP/, NLP-bge/
Fusion: CNN embeddings + BGE embeddings, cross-attention, reported 90.1%   FUSION/
```

These original folders are **unchanged**. All new work is in new folders.

---

## 3. Review of the original version: what was wrong

| # | Problem | Evidence | Effect |
|---|---|---|---|
| 1 | Test set used to pick the best epoch | `best_val_acc` equals test accuracy in every results file | Reported accuracy inflated |
| 2 | No baseline | "Same as 7 days ago" already scored **81.1%** | The 83.75% model added about 2.6 points |
| 3 | Colours averaged to grey | `gray = arr.mean(axis=2)` in stack_images.py | For SST anomaly, +3 °C and −3 °C became the same grey |
| 4 | Sequences built by file order | Jan–Aug 2023 SST-anomaly images missing | A "7-day window" could silently span months |
| 5 | One label for the whole region | `baa_labels.csv` has one value per day | No reef-level map possible |
| 6 | Tasks mixed in the results table | Baseline models predicted the same day; dropout / early-stopping models 7 days ahead | Results not comparable |
| 7 | Fusion not comparable | 3 classes, same-day target, test-set early stopping | The 90.1% is not a like-for-like gain |
| 8 | NLP carried no signal | Fusion's BGE file had **7 distinct vectors (one per year)**; the newer one had 4. 2024 and 2025 shared an unseen vector | NLP added essentially nothing to prediction |
| 9 | Reports describe the past | Annual reports are published after each summer | Using them as inputs risks hindsight leakage |

Also found: problems in the literature survey (Chapter 3 / Phase-1 report).
- Author names in the text don't match the reference list.
- Yang et al. (2024) has conflicting study regions.
- Three references could not be verified: González-Rivero 2024, Hensel 2025, Ichikawa 2024.

---

## 4. Everything done, in order

### Phase A: first website (live NOAA map)
- Built a website showing **real NOAA Coral Reef Watch data** from the PacIOOS ERDDAP server.
- Snapshot data is saved for demo dates, and any date since 1985 can be loaded live.
- Rebuilt it as a full **landing page**: hero, key numbers, why it matters, alert levels, how it works, live map, dataset chart, roadmap.
- The map is clipped to the **official GBR Marine Park boundary** (GBRMPA open data), with **2,722 reef outlines**, northern/central zones, zone statistics and hover values.
- Preset dates are the real peak-alert days of each bleaching summer: 2016, 2017, 2020, 2022, 2024, 2025.

### Phase B: corrected forecasting pipeline (v1)
- **Tried to download raw NOAA data** (`forecast/1_data_preparation/download_noaa.py`). NOAA's servers (PacIOOS and CoastWatch) throttle bulk downloads and blocked the downloader, so it is slow; only a partial download exists.
- **Recovered the real values from your own images** (`forecast/1_data_preparation/recover_from_images.py`). Each PNG pixel was painted from a fixed colour scale, so matching each pixel's colour to the scale gives the original value back.
  - Checked against real NOAA data: mean error 0.006 °C (SST anomaly), 0.010 °C (HotSpot), 0.018 °C-weeks (DHW), 0.008 °C (SST).
  - The alert level computed from the recovered values matches NOAA's on 98.8–99.4% of cells.
- **Verified NOAA's alert rule:** our BAA formula from HotSpot and DHW matches NOAA's published BAA on all 240,555 pixels checked.
- **New task:** a forecast for **every Marine Park cell** (1,927 cells, 0.1° grid), warm season only.
- **Split by season:** train 2018–2021, validate 2022–2023, test 2024–2025.
- **Models (v1):** persistence, climatology, trend + NOAA rule, gradient boosting, CNN-ConvLSTM on raw values.
- **Result:** CNN-ConvLSTM 66.3% at 7 days vs 65.6% for persistence, with Alert recall 71% vs 58%.
- **Noise-ceiling analysis:** "same as today" gets only 87.3% even at 1 day, so 85% exact at 7 days isn't realistic for the daily target.

### Phase C: backend, real-time data, LLM and tests
- **FastAPI server** (`backend/app.py`):
  - serves the website,
  - fetches today's NOAA data every 3 h (cached, with snapshot fallback),
  - runs the forecast on live data,
  - streams AI briefings.
- **Report corpus (NLP):** 14 report PDFs plus NOAA product pages, turned into **3,180 passages** with page numbers and publication years (`backend/ai/build_corpus.py`).
- **Retrieval** (`backend/ai/rag.py`):
  - BM25 keyword search.
  - Only passages **published before the date being explained** are used (for Jan–Jun dates, the previous year's reports or earlier), so there's no hindsight.
  - At most 2 passages per report.
- **LLM** (`backend/ai/llm.py`, [LLM.md](LLM.md)):
  - Claude Opus 5 via the Anthropic SDK, with refusal fallback.
  - Gets facts (numbers, forecast, model skill) plus 6 numbered sources and strict rules.
  - Streams a cited briefing or Q&A answer.
  - Falls back to an offline briefing without an API key. It has not yet been tested with a real key.
- **Website:** AI briefing panel, forecast layer, "Today (live)" data.
- **Tests:** 12 offline tests at this stage (`tests/test_core.py`; 16 now).

### Phase D: physics-guided model (v2)
- **Full-year dataset** from the images: 2,891 days, 4 channels (added SST). The 240 missing SST-anomaly days of 2023 were rebuilt from SST and climatology (held-out check error 0.01–0.03 °C).
- **Verified NOAA's DHW formula** on our data: sum of HotSpots ≥ 1 °C over 84 days ÷ 7. Correlation with the real DHW: 0.996–0.999.
- **Weather drivers** (`forecast/1_data_preparation/download_weather.py`): Open-Meteo / ERA5 wind, cloud, sunlight and air temperature at 74 points, 2017–2025.
- **Climate drivers:** ENSO (Niño 3.4) and MJO (RMM1/2).
- **PhysNet** (`forecast/2_model/physnet.py`):
  - CNN encoder → ConvLSTM over 4 weeks.
  - Predicts HotSpot for the next 14 days.
  - A **NOAA DHW formula layer** computes DHW exactly.
  - Outputs the alert level for leads 1–14.
  - Loss: cross-entropy + ordinal term + HotSpot error.
  - Final model: an average of 3 trained models.
- **Ablations:** ocean only; + past weather and indices; + weather forecast; no physics layer; no air temperature.
- **Evaluation** (`forecast/4_evaluation/evaluate_v2.py`): lead-time curve, 95% bootstrap confidence intervals, paired significance tests, reliability check, 2024 case study.
- **Real-forecast test** (`forecast/4_evaluation/eval_real_forecasts.py`): re-scored with the weather forecasts actually issued in 2024–2025.
- **Hybrid** (`forecast/4_evaluation/hybrid.py`): physics persistence for short leads, the network for longer ones, with the cut-off chosen on validation only.
- **Live forecast** (`backend/forecasting/forecast_v2.py`):
  - 112 days of NOAA data, the real 16-day weather forecast and live indices.
  - Falls back to the ocean-only model if the weather service is down.
- **Website:** lead-time selector (1/3/7/14 days) and "chance of Alert".

### Phase E: improvement round 2
- **Target changed to NOAA's official 7-day maximum alert:** the highest level over the 7 days ending on the forecast day. For short leads, the model also gets the part of the window already observed.
- **Forecast-error training** (`forecast/1_data_preparation/forecast_error.py`): real forecast errors measured from May–October 2024 and 2025 archived forecasts (months never scored in the test), then added to the training weather. It did **not** improve results (an honest negative result).
- **Results:** 82.4% at 7 days (baseline 68.7%); 76.9% with real forecasts (baseline 63.8%).
- **Hybrid:** physics persistence for 1–4 days, the network for 5–14 days. Beats the baseline at every lead.

### Phase F: literature review and data checks
- **Your `PAPERS` folder:** mostly bleaching **detection** from photos or satellite images (YOLOv8, other CNNs, Sentinel-2) and NOAA's monitoring overview.
- **Published work:** about 35 papers across 6 areas. We found **no** published reef-scale (0.1°), 1–14-day forecast of bleaching alerts, **no** physics-guided model with NOAA's DHW formula built in, and **no** date-filtered LLM explanations tied to live forecasts.
- **eReefs** (ocean model) archives end on **17 Jan 2024**, where the test period starts, so it can't be used.
- **Global Coral Bleaching Database:** no northern/central GBR records after 2017.
- **Cheung et al. 2025 (Zenodo):** 2016/2017/2020 GBR aerial surveys, a lead for validating against observed bleaching in 2020.

### Phase G: live system switched to the 7-day-maximum model
- The website and API now serve the **7-day-maximum-alert ensemble**, with physics persistence for 1–4 day leads.
- If the weather service is down, the forecast falls back to physics persistence for every lead.
- New terminal tool `backend/forecasting/forecast_now.py` runs the same real-time pipeline and can save the forecast.
- Spot check for the forecast issued 3 Mar 2024:
  - +7 days (week to 10 Mar): forecast 748 / 1,010 Alert reefs (north / central) vs observed **744 / 873**.
  - +14 days: forecast 353 / 37 vs observed **195 / 4**.

### Phase H: free local LLM and stronger hallucination checks
- The AI briefing now runs on a **free, open-source model**, Qwen 2.5 3B via Ollama, on the laptop. Claude remains optional (paid); the offline template stays as the last fallback.
- Hallucination checks (`backend/ai/grounding.py`): number check, citation check and a new **date-number check** that catches a real number used for the wrong forecast day. Each forecast lead is now given to the LLM with its calendar date.
- For the small model: automatic source tagging, heading and length clean-up, and strict mode, which deletes any sentence that fails the checks.
- Measured: 4/4 briefings fully grounded, 100% of numbers verified, 100% of factual sentences cited (`backend/data/grounding_eval.json`).
- Tests: 16.

---

## 5. Complete workflow

```
1. DATA
   CNN/2_images/*.png ──recover_from_images.py──► forecast/data/cube.npz
        (colour → real value, checked vs NOAA)       2,891 days × [SST anomaly, HotSpot, DHW, SST]
                                                     × 107×106 grid, Marine Park mask (1,927 cells)
   Open-Meteo, NOAA CPC, BoM ──download_weather.py──► forecast/data/weather.npz
   website/data/gbr.js (GBRMPA boundary + reefs) ◄── website/build_data.py

2. TRAINING                  forecast/3_training/train_v2.py
   dataset.Cube (calendar-aware samples, season split, 7-day-max target)
   physnet.Builder (10 frames × 16 channels + known DHW drop-off + future weather)
   physnet.PhysNet (CNN → ConvLSTM → HotSpot head → DHW formula layer → alert head)
   ► forecast/6_saved_models/v2/*.pt, log.json, meta.json

3. EVALUATION
   evaluate_v2.py [--target max7]  ► results/v2*/RESULTS.md, figures (lead curve, reliability, case study, ablation)
   eval_real_forecasts.py          ► real_forecast_check*.json (real archived weather forecasts)
   hybrid.py                       ► hybrid*.json (physics vs network per lead, chosen on validation)

4. LIVE SYSTEM               ./run.sh → http://127.0.0.1:8000
   backend/app.py
     /api/grid/{date}   NOAA map data (live, cached, snapshot fallback)
     /api/forecast      backend/forecasting/forecast_v2.py: NOAA 112 days + weather forecast + indices → ensemble → hybrid
     /api/explain       build_facts → rag.py (BM25, no hindsight) → llm.py (Claude, streamed; offline fallback)
   website/  landing page · live reef map · forecast layer (1/3/7/14 d) · AI briefing
```

**Commands, in order:**
```bash
.venv/bin/python forecast/1_data_preparation/recover_from_images.py
.venv/bin/python forecast/1_data_preparation/download_weather.py
.venv/bin/python forecast/3_training/train_v2.py                     # ablations + ensemble (daily target)
.venv/bin/python forecast/3_training/train_v2.py --only max7 --seed 0   # repeat for seeds 1, 2
.venv/bin/python forecast/4_evaluation/evaluate_v2.py --target max7
.venv/bin/python forecast/4_evaluation/eval_real_forecasts.py --target max7
.venv/bin/python forecast/4_evaluation/hybrid.py --target max7
.venv/bin/python -m backend.ai.build_corpus
./run.sh
```

---

## 6. What changed vs the original

| Area | Before | Now |
|---|---|---|
| Input values | Grey pixels from coloured images | Real physical values decoded from the colours |
| Channels | 4 | 16 (ocean + weather + indices + position + season) |
| History | 7 days | 4 weeks (7 daily frames + 3 weekly averages) |
| Output | 1 regional level, 7 days ahead | Every reef cell, 1–14 days ahead, plus chance of Alert |
| Target | Daily alert level | NOAA's official 7-day maximum alert (daily also reported) |
| Model | CNN-LSTM | Physics-guided CNN-ConvLSTM with NOAA DHW formula layer, average of 3 models, hybrid |
| Evaluation | Test set used for model choice, no baseline | Season split, 3 baselines, CIs, significance, reliability, real-forecast test |
| NLP | BGE-large embeddings fused into prediction | Report passages retrieved (BM25) to ground LLM explanations |
| LLM | None | Free local model (Qwen 2.5 via Ollama) or Claude; cited, fact-checked, no hindsight, streamed |
| Website | None | Landing page, live map, forecast layer, AI panel |
| Operations | Colab notebooks, manual uploads | Scripts, live server, caching, fallbacks, tests, `run.sh` |

**Files changed outside the new folders:**
- `README.md` rewritten (the old one is in git history).
- `.gitignore` extended: `.venv`, `.env`, `forecast/data/`, caches.

Nothing in `CNN/`, `NLP/`, `NLP-bge/`, `FUSION/`, `PAPERS/` or `Coll-Reports/` was modified.

---

## 7. What is inside each folder

### New folders

**`coralwatch/`**: shared library used by the model and the server.

| File | Contents |
|---|---|
| `geo.py` | Marine Park boundary and reefs, park mask, zones, reef-to-cell lookup, NOAA BAA rule |
| `noaa.py` | NOAA data access (PacIOOS and CoastWatch ERDDAP), cached day downloads, map grids |
| `stats.py` | Zone statistics: reefs and area share at each alert level, DHW, anomaly |
| `weather.py` | Open-Meteo weather (archive, live forecast, archived past forecasts), ENSO and MJO |

**`forecast/`**: forecasting research pipeline, numbered in pipeline order like `CNN/`. Details: [forecast/README.md](../forecast/README.md).

| Folder | Contents |
|---|---|
| `1_data_preparation/` | `recover_from_images.py` (real values from the PNGs, checked vs NOAA), `download_weather.py`, `forecast_error.py`, `download_noaa.py` |
| `2_model/` | `physnet.py` (physics-guided CNN-ConvLSTM), `dataset.py` (dataset, season split, targets), `paths.py` (all file locations) |
| `3_training/` | `train_v2.py`: model, ablations, 3-run ensemble |
| `4_evaluation/` | `evaluate_v2.py`, `eval_real_forecasts.py`, `hybrid.py` |
| `5_results/` | `v2_max7_final/` (**headline**), `v2_daily_target/` (daily target + ablations), `v1_first_corrected_model/`, `data_checks/` |
| `6_saved_models/` | `v2/` (all v2 models, `meta.json` = live model, `log.json` = training history), `v1/` |
| `data/` | `cube.npz`, `weather.npz`, `forecast_error_stats.json`, bleaching database, caches |
| `logs/` | Every training and evaluation log |
| `v1_first_model/` | First corrected model code (baselines, gradient boosting, CNN-ConvLSTM) |

**`backend/`**: web server and AI. Details: [backend/README.md](../backend/README.md).

| Item | Contents |
|---|---|
| `app.py` | FastAPI server: website, live data, forecast and explain endpoints, background refresh |
| `forecasting/` | `forecast_v2.py` (live final model + hybrid + fallback), `forecast_v1.py` (fallback), `forecast_now.py` (terminal tool) |
| `ai/` | `build_corpus.py` (3,180 report passages), `rag.py` (retrieval, no hindsight), `llm.py` (prompt, streaming, offline briefing) |
| `data/` | `corpus.json`, `cache/` (NOAA days, weather, map grids) |

**`website/`**: the site (also opens without the server).

| File | Contents |
|---|---|
| `index.html`, `style.css` | Page structure and deep-ocean styling |
| `app.js` | Live map, layers, forecast layer with lead selector, zone cards, AI briefing |
| `landing.js` | Hero animation, alert-level cards, dataset chart |
| `build_data.py` | Downloads the map snapshots and GBRMPA geography |
| `data/gbr.js`, `data/snapshots.js` | Marine Park boundary + 2,722 reefs; saved NOAA days |
| `README.md` | How the site works |

**Other new items:**

| Item | Contents |
|---|---|
| `tests/test_core.py` | 16 offline tests: BAA rule, park mask, zone stats, retrieval filters, LLM prompt and streaming, fact-checking, API, DHW formula |
| `docs/TEAM_GUIDE.md`, `docs/LLM.md`, `docs/PROJECT_REPORT.md`, `docs/RESULTS_EXPLAINED.md` | Team guide; how data is fed to the LLM; this report; all numbers |
| `run.sh`, `requirements.txt`, `.env.example` | One-command start, dependencies, API-key template |
| `.venv/` | Python environment (not in git) |

### Original folders (unchanged)

| Folder | Contents |
|---|---|
| `CNN/` | Raw NetCDF (2018–2019), PNG images 2018–2025, stacked dataset, v1 notebooks, results, scripts |
| `NLP/`, `NLP-bge/` | Report ZIPs, BGE embeddings, NLP notebooks |
| `FUSION/` | CNN + BGE cross-attention fusion notebook and model |
| `PAPERS/`, `Coll-Reports/` | Reference papers, literature survey, phase reports |

---

## 8. Key findings

1. The original 83.75% / 90.1% were inflated by test-set model selection and a missing baseline.
2. Daily alert levels are inherently noisy: even a 1-day "same as today" forecast reaches only 87%.
3. **Weather forecasts are the biggest driver of 7-day skill** (69% → 78% on the daily target). Past weather adds nothing beyond sea-temperature history.
4. The **DHW formula layer** helps, most at long leads (14-day macro-F1 0.670 vs 0.625 without it).
5. NOAA's 7-day maximum alert is more predictable and more useful: 82.4% at 7 days.
6. With real archived forecasts the gain holds: 76.9% vs 63.8%.
7. Training with realistic forecast error didn't help.
8. Report text doesn't add predictive signal; it is valuable for grounded explanations.
9. Colour-coded images can be converted back to exact values, which fixes the grayscale problem.

## 9. Limitations and open items
- We forecast NOAA's **heat-stress** alert, not observed bleaching. The next step is a check against the 2020 aerial surveys.
- Values are recovered from images; spot-checking against raw NOAA files would strengthen this.
- Only 2 test seasons and one region.
- Fixed NOAA thresholds, although coral heat tolerance is rising.
- No subsurface ocean data: eReefs ends Jan 2024, and Copernicus needs a free account.
- No head-to-head comparison with NOAA's or BoM's operational outlooks.
- The Claude path has not been tested with a real API key (the free local model has).
- Fix the literature-survey citation problems before submitting.
