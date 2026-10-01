# All the numbers, on one page

**"Baseline"** always means the simplest possible forecast: *"the alert level will be the
same as today."* A model is only useful if it beats this. All accuracies are for the unseen
test years 2024–2025 unless marked otherwise.

## The two numbers to remember

| Number | What it means |
|---|---|
| **82.4%** | Accuracy 7 days ahead. The model is given the weather that **actually happened** as its "next 14 days of weather". This is the model's quality with a perfect weather forecast. Baseline: 68.7%. |
| **76.9%** | Accuracy 7 days ahead in the **peak bleaching months** (Jan–Apr of 2024 and 2025). The model is given the **weather forecasts that were really issued at the time**. This is what you'd get using it for real. Baseline on these same days: 63.8%. |

**Why two baselines (68.7% vs 63.8%)?** They cover different days. 68.7% is over all
test days (Nov–Apr); 63.8% is only the Jan–Apr peak, when conditions change fastest and
forecasting is hardest. Each model is always compared with the baseline **on the same days**.
The gain is similar in both cases: about 13–14 points.

## Final model, by how far ahead (NOAA 7-day maximum alert)

| Days ahead | Our forecast | Baseline | Gain |
|---|---|---|---|
| 1 day | 97.1% | 94.7% | +2.4 |
| 3 days | 90.6% | 85.0% | +5.6 |
| 7 days | 82.4% | 68.7% | +13.7 |
| 14 days | 76.7% | 50.8% | +25.9 |

For 1–4 days ahead, the forecast uses **physics persistence** (today's HotSpot plus NOAA's
DHW formula). For 5–14 days it uses the neural network. The split was chosen on validation data.

## Other quality numbers for the final model (7 days ahead)

| Number | What it means |
|---|---|
| 0.829 Alert F1 | How well it picks out reef cells at Alert Level 1 or higher (1 is perfect) |
| 83.1% Alert recall | Share of Alert-level cells it caught a week early |
| 95.0% within one level | Forecasts that were right or off by just one level |
| 0.73 Brier skill score | Quality of its "chance of Alert" probabilities (0 = no better than average, 1 = perfect) |

## How the accuracy improved, step by step

| Stage | 7-day accuracy | Baseline | Notes |
|---|---|---|---|
| Original project | 83.75% | (none reported; a trivial rule scored 81.1%) | Inflated: test set used to pick the model |
| Original fusion (CNN + NLP) | 90.1% | – | Not comparable: 3 classes, same-day target |
| Corrected CNN-ConvLSTM (v1), daily target | 66.3% | 65.6% | Honest evaluation, real values, every reef cell |
| Physics-guided model, ocean data only | 69.3% | 65.6% | NOAA's DHW formula built in |
| + weather forecast, daily target | 77.9% | 65.6% | With real forecasts: 70.9% vs 57.4% |
| **+ NOAA 7-day maximum alert target (final)** | **82.4%** | **68.7%** | With real forecasts: **76.9% vs 63.8%** |

The **daily** target (the level on one exact day) is harder, because it flips day to day.
The **7-day maximum** (the highest level over a week) is NOAA's official product and what
reef managers use. The two targets' accuracies shouldn't be compared directly; compare
each one with its own baseline.

## Data-quality numbers

| Number | What it means |
|---|---|
| ≈ 0.01 error | Difference between values recovered from the PNG images and NOAA's raw files |
| 97–99% | Cells where the recovered alert level equals the raw NOAA alert level |
| 0.996–0.999 | Correlation between NOAA's DHW formula applied to our data and the real DHW |
| 1,927 | Reef cells forecast (0.1° grid, northern + central Marine Park) |

## Did we reuse the original CNN + LSTM?

We kept **the same idea** and rebuilt it, because the original couldn't produce a map or use real values.

| Part | Original | Now |
|---|---|---|
| CNN (reads each day's map) | 4 conv blocks → one 256-number summary | Conv blocks (same style: conv → batch-norm → ReLU) that keep the map shape |
| LSTM (learns change over time) | LSTM on the summary vector | **ConvLSTM**: an LSTM that works on the whole map, so every reef cell gets its own forecast |
| Output | 1 alert level for the whole region | Alert level for all 1,927 reef cells, 1–14 days ahead |
| Added | – | NOAA DHW formula layer, weather inputs, 4 weeks of history |

The original trained weights weren't reused. The inputs changed from grey 224×224 images
to real-valued 107×106 maps with 16 channels, the output changed from one label to a map,
and those weights had been selected using the test set. The original notebooks remain in
`CNN/4_models/` unchanged.
