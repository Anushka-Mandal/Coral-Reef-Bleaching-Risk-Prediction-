# Forecast results v2 - physics-guided model (NOAA 7-day maximum alert)

Held-out test seasons 2024-2026 (408 forecast days, 1,927 reef cells each). Brackets: 95% confidence intervals from a moving-block bootstrap (7-day blocks) over forecast days.

**Selected on validation:** PhysNet + weather forecast, 7-day maximum alert - ensemble of 3

## 7-day lead

| Model | Accuracy | Macro-F1 | Alert F1 | Alert recall | Within one level |
|---|---|---|---|---|---|
| Persistence | 0.687 [0.640-0.730] | 0.677 [0.635-0.706] | 0.720 [0.640-0.775] | 0.735 [0.661-0.803] | 0.928 [0.898-0.954] |
| Trend + NOAA rule | 0.720 [0.677-0.764] | 0.725 [0.690-0.757] | 0.777 [0.709-0.823] | 0.712 [0.631-0.778] | 0.937 [0.915-0.958] |
| Physics persistence | 0.719 [0.669-0.761] | 0.714 [0.667-0.750] | 0.796 [0.721-0.847] | 0.720 [0.615-0.800] | 0.942 [0.922-0.960] |
| PhysNet + weather forecast, 7-day maximum alert (seed 0) | 0.815 [0.785-0.840] | 0.807 [0.774-0.827] | 0.818 [0.751-0.869] | 0.820 [0.731-0.880] | 0.947 [0.922-0.967] |
| PhysNet 7-day maximum alert, trained with forecast error (seed 0) | 0.796 [0.765-0.828] | 0.792 [0.758-0.819] | 0.823 [0.759-0.872] | 0.802 [0.710-0.877] | 0.949 [0.928-0.968] |
| PhysNet + weather forecast, 7-day maximum alert - ensemble of 3 | 0.824 [0.796-0.849] | 0.817 [0.784-0.838] | 0.829 [0.768-0.879] | 0.831 [0.749-0.891] | 0.950 [0.926-0.970] |

## 3-day lead

| Model | Accuracy | Macro-F1 | Alert F1 | Alert recall | Within one level |
|---|---|---|---|---|---|
| Persistence | 0.850 [0.824-0.873] | 0.850 [0.827-0.867] | 0.876 [0.838-0.904] | 0.883 [0.840-0.918] | 0.967 [0.952-0.980] |
| Trend + NOAA rule | 0.908 [0.887-0.928] | 0.916 [0.898-0.931] | 0.949 [0.926-0.964] | 0.926 [0.888-0.952] | 0.987 [0.979-0.992] |
| Physics persistence | 0.906 [0.883-0.927] | 0.914 [0.892-0.930] | 0.952 [0.928-0.968] | 0.920 [0.878-0.950] | 0.987 [0.980-0.993] |
| PhysNet + weather forecast, 7-day maximum alert (seed 0) | 0.884 [0.866-0.902] | 0.878 [0.858-0.893] | 0.912 [0.880-0.937] | 0.917 [0.878-0.946] | 0.975 [0.961-0.986] |
| PhysNet 7-day maximum alert, trained with forecast error (seed 0) | 0.882 [0.861-0.901] | 0.874 [0.851-0.891] | 0.918 [0.888-0.940] | 0.908 [0.863-0.941] | 0.977 [0.967-0.987] |
| PhysNet + weather forecast, 7-day maximum alert - ensemble of 3 | 0.891 [0.873-0.906] | 0.881 [0.860-0.896] | 0.918 [0.890-0.942] | 0.911 [0.868-0.942] | 0.978 [0.966-0.987] |

## 1-day lead

| Model | Accuracy | Macro-F1 | Alert F1 | Alert recall | Within one level |
|---|---|---|---|---|---|
| Persistence | 0.947 [0.938-0.955] | 0.948 [0.939-0.954] | 0.957 [0.944-0.967] | 0.959 [0.942-0.971] | 0.988 [0.983-0.992] |
| Trend + NOAA rule | 0.969 [0.962-0.976] | 0.972 [0.966-0.977] | 0.983 [0.975-0.989] | 0.975 [0.963-0.984] | 0.996 [0.993-0.998] |
| Physics persistence | 0.971 [0.964-0.979] | 0.974 [0.967-0.980] | 0.986 [0.978-0.991] | 0.974 [0.960-0.984] | 0.996 [0.994-0.998] |
| PhysNet + weather forecast, 7-day maximum alert (seed 0) | 0.919 [0.906-0.931] | 0.906 [0.890-0.918] | 0.931 [0.906-0.951] | 0.938 [0.908-0.960] | 0.980 [0.971-0.988] |
| PhysNet 7-day maximum alert, trained with forecast error (seed 0) | 0.912 [0.897-0.925] | 0.898 [0.875-0.913] | 0.930 [0.907-0.949] | 0.914 [0.879-0.942] | 0.981 [0.973-0.988] |
| PhysNet + weather forecast, 7-day maximum alert - ensemble of 3 | 0.923 [0.909-0.935] | 0.907 [0.889-0.920] | 0.937 [0.913-0.955] | 0.928 [0.895-0.953] | 0.982 [0.973-0.990] |

## 14-day lead

| Model | Accuracy | Macro-F1 | Alert F1 | Alert recall | Within one level |
|---|---|---|---|---|---|
| Persistence | 0.508 [0.448-0.569] | 0.493 [0.445-0.538] | 0.570 [0.490-0.632] | 0.574 [0.473-0.670] | 0.876 [0.836-0.915] |
| Trend + NOAA rule | 0.541 [0.483-0.598] | 0.549 [0.497-0.591] | 0.609 [0.521-0.688] | 0.598 [0.489-0.694] | 0.854 [0.808-0.893] |
| Physics persistence | 0.542 [0.484-0.599] | 0.530 [0.477-0.579] | 0.646 [0.545-0.723] | 0.634 [0.498-0.749] | 0.866 [0.825-0.898] |
| PhysNet + weather forecast, 7-day maximum alert (seed 0) | 0.746 [0.711-0.780] | 0.739 [0.700-0.766] | 0.767 [0.699-0.820] | 0.719 [0.634-0.787] | 0.933 [0.905-0.958] |
| PhysNet 7-day maximum alert, trained with forecast error (seed 0) | 0.710 [0.667-0.747] | 0.702 [0.654-0.736] | 0.728 [0.628-0.797] | 0.675 [0.545-0.784] | 0.922 [0.892-0.950] |
| PhysNet + weather forecast, 7-day maximum alert - ensemble of 3 | 0.767 [0.735-0.796] | 0.766 [0.731-0.792] | 0.800 [0.739-0.850] | 0.771 [0.687-0.835] | 0.943 [0.917-0.964] |

## Better than persistence at 7 days? (paired bootstrap, one-sided p-values)

| Model | Accuracy | Macro-F1 | Alert F1 |
|---|---|---|---|
| Trend + NOAA rule | 0.042 | 0.003 | 0.012 |
| Physics persistence | 0.067 | 0.035 | 0.006 |
| PhysNet + weather forecast, 7-day maximum alert (seed 0) | 0.000 | 0.000 | 0.000 |
| PhysNet 7-day maximum alert, trained with forecast error (seed 0) | 0.000 | 0.000 | 0.000 |
| PhysNet + weather forecast, 7-day maximum alert - ensemble of 3 | 0.000 | 0.000 | 0.000 |

p < 0.05 means the model is better than persistence with 95% confidence.

## Reliability of the 7-day alert probability

Brier score 0.0409 vs 0.1528 for climatology (skill score 0.732; above 0 = more useful than the long-term average).

| Forecast probability | Mean forecast | Observed frequency | Cells |
|---|---|---|---|
| 0.0-0.1 | 0.00 | 0.01 | 596,922 |
| 0.1-0.2 | 0.15 | 0.21 | 20,738 |
| 0.2-0.3 | 0.25 | 0.33 | 13,355 |
| 0.3-0.4 | 0.35 | 0.42 | 10,631 |
| 0.4-0.5 | 0.45 | 0.49 | 10,038 |
| 0.5-0.6 | 0.55 | 0.55 | 9,587 |
| 0.6-0.7 | 0.65 | 0.61 | 10,854 |
| 0.7-0.8 | 0.75 | 0.67 | 14,100 |
| 0.8-0.9 | 0.85 | 0.73 | 19,592 |
| 0.9-1.0 | 0.98 | 0.94 | 80,399 |

## Validation scores used for model choice (mean of accuracy and macro-F1, 7-day lead)

- PhysNet + weather forecast, 7-day maximum alert (seed 0): 0.7928
- PhysNet 7-day maximum alert, trained with forecast error (seed 0): 0.7632
- PhysNet + weather forecast, 7-day maximum alert (seed 1): 0.7831
- PhysNet + weather forecast, 7-day maximum alert (seed 2): 0.7842
