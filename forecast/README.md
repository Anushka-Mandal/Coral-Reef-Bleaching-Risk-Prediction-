# forecast/: the forecasting model

Forecasts NOAA's Bleaching Alert level for every reef cell of the northern and central
Great Barrier Reef (1,927 cells, 0.1° grid), 1 to 14 days ahead. Folders are numbered in
pipeline order, the same way as `CNN/`.

| Folder | What's inside |
|---|---|
| `1_data_preparation/` | `recover_from_images.py`: real values from `CNN/2_images` (checked vs NOAA) · `download_weather.py`: weather + ENSO/MJO · `forecast_error.py`: size of real weather-forecast errors · `download_noaa.py`: raw NOAA download (throttled, partial) |
| `2_model/` | `physnet.py`: the physics-guided CNN-ConvLSTM · `dataset.py`: dataset, season split, targets · `paths.py`: every file location |
| `3_training/` | `train_v2.py`: trains the model, its ablations and the 3-run ensemble |
| `4_evaluation/` | `evaluate_v2.py`: full evaluation · `eval_real_forecasts.py`: re-test with real archived forecasts · `hybrid.py`: physics vs network per lead (chosen on validation) |
| `5_results/` | Results. **`v2_max7_final/` is the headline.** See [5_results/README.md](5_results/README.md) |
| `6_saved_models/` | Trained models: `v2/` (current), `v1/` (first corrected model) |
| `data/` | Datasets and download caches. See [data/README.md](data/README.md) |
| `logs/` | Output logs of every training and evaluation run |
| `v1_first_model/` | The first corrected model (CNN-ConvLSTM, baselines, gradient boosting), kept for comparison |

## The model in one picture

```
4 weeks of daily maps (SST anomaly, HotSpot, DHW, SST) + weather (+ 14-day forecast) + ENSO/MJO
   → CNN encoder per day → ConvLSTM over time → HotSpot for the next 14 days
   → NOAA DHW formula layer → alert level (7-day maximum) for every reef cell, 1–14 days ahead
   → average of 3 trained models → hybrid: physics persistence for 1–4 days, network for 5–14
```

## Run the pipeline

```bash
.venv/bin/python forecast/1_data_preparation/recover_from_images.py
.venv/bin/python forecast/1_data_preparation/download_weather.py
.venv/bin/python forecast/3_training/train_v2.py                         # ablations (daily target)
.venv/bin/python forecast/3_training/train_v2.py --only max7 --seed 0    # final model; repeat for seeds 1, 2
.venv/bin/python forecast/4_evaluation/evaluate_v2.py --target max7
.venv/bin/python forecast/4_evaluation/eval_real_forecasts.py --target max7
.venv/bin/python forecast/4_evaluation/hybrid.py --target max7
```

Real-time use: `.venv/bin/python -m backend.forecasting.forecast_now` (see `backend/`).

## Headline results (test seasons 2024–2025)

- **82.4%** [79.6–84.9] accuracy 7 days ahead vs 68.7% for "same as today".
- **76.9%** [73.0–80.7] with the real weather forecasts of the time vs 63.8%.
- 97.1% at 1 day → 76.7% at 14 days, better than "same as today" at every lead.

All numbers explained: [../docs/RESULTS_EXPLAINED.md](../docs/RESULTS_EXPLAINED.md).
