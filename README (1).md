# AirSense India: Air Quality Prediction (SDG 3)

Predicts the Air Quality Index (AQI) of Indian cities from pollutant levels, city and month, and
forecasts next-day AQI. Runs fully in the terminal (VS Code). Models: rule-based CPCB formula (baseline),
Linear Regression, Random Forest, Gradient Boosting (the best one is chosen automatically).

## Project structure
```
airsense_india/
├── data/city_day.csv      <- YOU download this
├── common.py              constants, AQI categories, CPCB sub-index formula, health advice
├── train_model.py         cleaning + EDA + training + evaluation + saving
├── predict.py             interactive terminal app (7 options)
├── test_basic.py          sanity tests for common.py
├── requirements.txt
├── models/                (created by train_model.py)
└── outputs/               (created by train_model.py: plots, metrics.json, cleaned data)
```

## How to run
1. Download city_day.csv from https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india
   and put it in the `data/` folder.
2. Open the folder in VS Code, open the terminal (Ctrl + `).
3. `pip install -r requirements.txt`
4. `python test_basic.py`   (optional, 1 second)
5. `python train_model.py`  (about 1-3 minutes; creates plots, metrics and models)
6. `python predict.py`

## What the app can do
1. Quick prediction: only PM2.5 and PM10 (the two values most air-quality apps show).
2. Detailed prediction: all pollutants; press Enter for any value you don't know.
3. Replay a real day: readings are filled from the dataset; compares prediction with the actual AQI
   (same day and next day). Random days come from the test period the model never saw.
4. City insights: monthly pattern and share of days per AQI category.
5. Compare cities for a chosen month.
6. Model performance: all evaluation tables.

Every prediction shows a likely range (from test-set errors), the main driver pollutant
(CPCB sub-index) and health advice.

## Method (what makes the results trustworthy)
- Missing values are NOT filled before splitting. Imputation happens inside the pipeline and is fitted on
  training data only (Gradient Boosting handles missing values natively).
- Evaluation uses a time-based split (train on the past, test on the latest 20% of dates). A random split is
  also reported to show how much it overestimates.
- Time-series cross-validation (5 folds) shows the result is not a lucky split.
- AQI is capped at 500 (the end of the official scale) and training/prediction use the same cap.
- Ahmedabad is excluded by default (see `EXCLUDE_CITIES` in train_model.py): its CO values average ~22 mg/m3
  (about 1.2 elsewhere) and most AQI values above 500 come from it.
- Feature importance uses permutation importance (more reliable than the Random Forest's built-in one).
- "Inputs needed" experiment shows accuracy when the user types 0, 2, 4 or all pollutant values.
- Next-day forecast is compared with the "tomorrow = today" baseline and uses yesterday's AQI and a 7-day mean.

## Limitations
- AQI is calculated from pollutants, so same-day prediction is mostly learning a known formula. The practical
  value is (a) estimating AQI when only some pollutants are known and (b) next-day forecasting.
- No weather data (temperature, wind, humidity). Month stands in for season.
- The test period (from about Dec 2019) includes the COVID-19 lockdown, when AQI dropped sharply; this is a
  harder, unusual test.
- Cities cover different years (several only have 2019-2020 data), so city comparisons are indicative.
- Educational project, not an official forecast.
