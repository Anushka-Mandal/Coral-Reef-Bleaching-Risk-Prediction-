# CoralWatch: coral bleaching risk for the northern and central Great Barrier Reef

**Coral Reef Bleaching Risk Prediction using Satellite Alerts and NLP on Observational Reports**, capstone project, Department of CSE.

CoralWatch forecasts NOAA's Bleaching Alert level for every reef cell of the northern
and central Great Barrier Reef **7 days ahead**. It shows today's heat stress and the
forecast on a live map, and uses an LLM to explain the risk with citations to reef
reports.

| Part | What it does | Where |
|---|---|---|
| Data | NOAA Coral Reef Watch 5 km daily SST anomaly, HotSpot, Degree Heating Weeks → Bleaching Alert Area | `forecast/`, `coralwatch/noaa.py` |
| Forecast | Physics-guided CNN-ConvLSTM (NOAA DHW formula layer, weather forecasts), 1–14 days, with baselines and ablations | `forecast/` |
| Reports (NLP) | AIMS, Marine Monitoring Program and Reef Authority reports split into 3,180 searchable passages | `backend/ai/build_corpus.py`, `backend/ai/rag.py` |
| LLM | A free local model (Qwen 2.5 via Ollama) or Claude explains the forecast and live numbers, citing report passages and fact-checked; offline fallback | `backend/ai/llm.py`, [docs/LLM.md](docs/LLM.md) |
| Website | Landing page, live map clipped to the Marine Park, forecast layer, AI briefing | `website/` |
| Server | Serves the site; live NOAA data (cached), forecast and LLM endpoints | `backend/app.py` |

## Run it

```bash
./run.sh                                     # website + live data, http://127.0.0.1:8000
.venv/bin/python -m backend.forecasting.forecast_now      # real-time forecast in the terminal
.venv/bin/python -m backend.forecasting.forecast_now --date 2024-03-03 --save forecast.json
```

**Real-time data is fed in automatically.** Every 3 hours the server downloads the newest NOAA
Coral Reef Watch days (112 days of history, cached), the Open-Meteo weather (past 40 days + the next
16-day forecast) and the ENSO/MJO indices, then runs the forecast from the latest NOAA day.
The live model is the 7-day-maximum-alert ensemble with physics persistence for 1–4 day leads;
if the weather service is down it falls back to physics persistence for every lead.

- **LLM (free):** `ollama pull qwen2.5:3b` once, then keep `ollama serve` running. Optional paid alternative: `ANTHROPIC_API_KEY=...` in `.env`. With neither, briefings use an offline rule-based mode. Details: [docs/LLM.md](docs/LLM.md).
- **New to the project?** Start with [docs/TEAM_GUIDE.md](docs/TEAM_GUIDE.md).
- **Offline use:** `website/index.html` also opens directly in a browser, with saved data and no server. The forecast and AI features then show as unavailable.
- **Tests:** `.venv/bin/python -m pytest tests -q`

## Rebuild everything from scratch

```bash
.venv/bin/python forecast/1_data_preparation/recover_from_images.py   # dataset from CNN/2_images (2018-2025), checked against NOAA
.venv/bin/python forecast/1_data_preparation/download_weather.py      # weather + ENSO/MJO drivers (Open-Meteo, NOAA, BoM)
.venv/bin/python forecast/3_training/train_v2.py              # physics-guided model, ablations, 3-seed ensemble
.venv/bin/python forecast/4_evaluation/evaluate_v2.py           # lead-time curve, CIs, significance, reliability
.venv/bin/python forecast/4_evaluation/eval_real_forecasts.py   # re-test with real archived weather forecasts
.venv/bin/python forecast/4_evaluation/hybrid.py                # choose physics persistence vs model per lead (validation)
.venv/bin/python -m backend.ai.build_corpus           # report passages for the LLM
cd website && python3 build_data.py                # map snapshots + reef geography
```

`forecast/1_data_preparation/download_noaa.py` fetches the raw values straight from NOAA instead. NOAA's
servers throttle bulk downloads heavily, so it downloads slowly and resumes where
it stopped.

## Results

**Live model: NOAA 7-day maximum alert** (the highest alert level over the week; NOAA's
operational product). Test 2024–2025, hybrid: **97.1%** at 1 day, **90.6%** at 3 days,
**82.4%** at 7 days (persistence 68.7%), **76.7%** at 14 days (persistence 50.8%). With the
real archived weather forecasts (Jan–Apr 2024 and 2025, 7 days ahead): **76.9%** vs 63.8%.
Tables: [forecast/5_results/v2_max7_final/RESULTS.md](forecast/5_results/v2_max7_final/RESULTS.md).

**Daily alert level** (the harder, noisier target):

Alert level for 1,927 reef cells, forecast 1–14 days ahead. Test seasons 2024–2025 were
never used for training or model choice. Brackets show 95% confidence intervals from a moving-block bootstrap (7-day blocks).
Full tables: [forecast/5_results/v2_daily_target/RESULTS.md](forecast/5_results/v2_daily_target/RESULTS.md).

**7-day lead, test seasons**

| Model | Accuracy | Macro-F1 | Alert F1 | Within one level |
|---|---|---|---|---|
| Persistence ("same as today") | 0.656 [0.610–0.700] | 0.562 | 0.565 | 0.926 |
| Physics persistence (today's HotSpot + NOAA DHW formula) | 0.681 | 0.632 | 0.650 | 0.917 |
| CNN-ConvLSTM, first corrected version | 0.663 | 0.599 | 0.600 | 0.906 |
| PhysNet, ocean data only | 0.693 | 0.591 | 0.585 | 0.924 |
| **PhysNet + weather forecast, ensemble of 3** | **0.779 [0.744–0.813]** | **0.725** | **0.717** | **0.952** |

**With the weather forecasts that were really issued** (Jan–Apr peaks of 2024 and 2025,
233 forecast days, 7-day lead): **70.9% [66.0–75.9]** vs 57.4% for persistence
(p < 0.001). Using observed instead of forecast weather gives 75.5%, so real forecasts
cost about 5 points, but the gain over the baseline holds.

**By lead time (test), operational hybrid:** physics persistence for 1–2 days, the PhysNet
ensemble for 3–14 days (chosen on validation). It beats persistence at every lead:
1 day 87.5% (persistence 87.1%) · 3 days 79.7% (77.0%) · 7 days 77.9% (65.6%) ·
14 days 74.2% (51.3%). See `forecast/5_results/v2_daily_target/hybrid.json`.

**Findings**
- Weather forecasts are what make a week-ahead forecast possible: 69% → 78%. Past weather adds nothing beyond sea-temperature history.
- The DHW formula layer helps (macro-F1 0.705 vs 0.699 with the same inputs), most at long leads (14 days: 0.670 vs 0.625).
- Air temperature isn't a hidden leak: without it, the model still scores 74.6%.
- The alert probabilities are informative (Brier skill score 0.62 vs climatology) and slightly under-confident.

## What changed from the first version (and why)

The first pipeline (notebooks in `CNN/4_models`, `FUSION/`) reported 83.75% (CNN-LSTM)
and 90.1% (fusion). A review found problems that made those numbers unreliable:

| Problem | Fix |
|---|---|
| The test set was also used to pick the best epoch | Separate validation seasons; test used once |
| No baseline; "same as 7 days ago" already scored 81% | Persistence, climatology and trend baselines reported alongside |
| Values were colour-mapped, then averaged to grey (SST anomaly +3 °C looked like −3 °C) | Raw physical values, recovered exactly from the images or downloaded |
| Sequences built by file index, so gaps (e.g. Jan–Aug 2023) broke the 7-day windows | Calendar-aware samples; incomplete weeks dropped |
| One label for the whole region, so no map was possible | A forecast for every Marine Park cell (1,927 cells at 0.1°) |
| Fusion used 3 classes, a same-day target and a different setup, so it wasn't comparable | Report text now grounds LLM explanations; fusion kept as a documented negative result |
| Winter days inflated accuracy | Only warm-season (Nov–Apr) target days are scored |

## Project layout

```
coralwatch/        shared code: geography, NOAA access, BAA rule, zone statistics
forecast/          dataset, models, training, evaluation, results
backend/           web server, forecast service, report search, LLM
website/           landing page + live map (static files)
docs/LLM.md        how data is fed to the LLM
tests/             offline tests
CNN/, NLP/, FUSION/  first-version data, notebooks and results
```

Data: NOAA Coral Reef Watch (CoastWatch and PacIOOS ERDDAP). Geography: © Commonwealth of Australia (GBRMPA), CC BY 4.0.
