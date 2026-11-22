"""
AirSense India: interactive terminal app
Run:  python predict.py      (after running train_model.py once)
"""
import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import (AQI_CAP, BUCKETS, HEALTH_ADVICE, MONTHS, UNITS, dominant_pollutant, to_bucket)

MODEL_PATH = Path("models/aqi_model.joblib")
METRICS_PATH = Path("outputs/metrics.json")
CLEAN_PATH = Path("outputs/air_quality_clean.csv")
LINE = "=" * 64
LOW_DATA_ROWS = 500   # cities with fewer daily records than this get a warning

# Terminal colours (256-colour codes work in VS Code and macOS Terminal).
USE_COLOR = sys.stdout.isatty()
BUCKET_256 = {"Good": (34, 15), "Satisfactory": (112, 0), "Moderate": (220, 0),
              "Poor": (208, 0), "Very Poor": (196, 15), "Severe": (88, 15)}


def badge(text, bucket):
    if not USE_COLOR:
        return f"[{text}]"
    bg, fg = BUCKET_256[bucket]
    return f"\033[48;5;{bg}m\033[38;5;{fg}m\033[1m {text} \033[0m"


def color_bar(value, vmax, bucket, width=40):
    filled = int(max(0, min(value, vmax)) / vmax * width)
    if not USE_COLOR:
        return "#" * filled + "." * (width - filled)
    return f"\033[38;5;{BUCKET_256[bucket][0]}m{'█' * filled}\033[0m{'░' * (width - filled)}"


def load_bundle():
    if not MODEL_PATH.exists():
        sys.exit("\nModel not found. Run  python train_model.py  first.\n")
    b = joblib.load(MODEL_PATH)
    return b["models"], b["meta"]


# ----------------------------- input helpers --------------------------------
def ask_city(meta):
    cities = meta["cities"]
    print("\nAvailable cities:")
    for i in range(0, len(cities), 4):
        print("  " + "".join(f"{c + ('*' if meta['city_rows'][c] < LOW_DATA_ROWS else ''):<20}"
                             for c in cities[i:i + 4]))
    print("  (* = limited data, less reliable)")
    default = "Delhi" if "Delhi" in cities else cities[0]
    lookup = {c.lower(): c for c in cities}
    while True:
        text = input(f"\nEnter city name (default {default}): ").strip().lower() or default.lower()
        if text in lookup:
            return lookup[text]
        matches = [c for c in cities if text in c.lower()]
        if len(matches) == 1:
            return matches[0]
        print("  City not recognised, please type it as shown in the list.")


def ask_month(default=11):
    while True:
        text = input(f"Enter month (1-12 or name, default {default}): ").strip().lower() or str(default)
        if text.isdigit() and 1 <= int(text) <= 12:
            return int(text)
        for i, name in enumerate(MONTHS, start=1):
            if len(text) >= 3 and name.lower().startswith(text):
                return i
        print("  Please enter a number from 1 to 12 or a month name.")


def ask_value(name, typical, plausible_max):
    """Returns a float, or NaN when the user does not know the value (just press Enter)."""
    unit = UNITS.get(name, "ug/m3")
    hint = "typical n/a" if typical is None or np.isnan(typical) else f"typical {typical:g}"
    while True:
        text = input(f"  {name:<8} ({unit:<5}) {hint:<14} Enter = don't know: ").strip()
        if text == "":
            return np.nan
        try:
            value = float(text)
            if value < 0:
                raise ValueError
        except ValueError:
            print("    Please enter a non-negative number, or press Enter to skip.")
            continue
        if value > plausible_max:
            print(f"    Warning: {value:g} is above 99.9% of all readings in the data ({plausible_max:g}).")
            if input("    Keep this value? [y/N]: ").strip().lower() != "y":
                continue
        return value


# ----------------------------- output helpers -------------------------------
def print_result(city, month_text, pred, band, typical, given, total, dominant, meta, note=None):
    bucket = to_bucket(pred)[0]
    low, high = max(0, pred - band), min(AQI_CAP, pred + band)
    print("\n" + LINE)
    print(f" City: {city}   {month_text}")
    print(f" PREDICTED AQI : {pred:.0f}   ->  {badge(bucket.upper(), bucket)}")
    print(f" Likely range  : {low:.0f} to {high:.0f}  (80% of past test predictions of this level "
          f"were within +/-{band:.0f})")
    print(f" [{color_bar(pred, AQI_CAP, bucket)}] 0..{AQI_CAP}")
    if typical is not None:
        print(f" Typical AQI in {city}: {typical:.0f}  ({pred - typical:+.0f} vs typical)")
    if dominant:
        print(f" Main driver   : {dominant[0]} (CPCB sub-index {dominant[1]:.0f}) among the values you entered")
    if given is not None:
        print(f" Inputs given  : {given} of {total} pollutants")
        if given == 0:
            print(" WARNING: no pollutant values entered, so this is only a typical value for the city and month.")
    if city in meta["unseen_cities"]:
        print(f" NOTE: {city} has no data in the training period; the model relies on pollutant levels only.")
    elif meta["city_rows"][city] < LOW_DATA_ROWS:
        print(f" NOTE: only {meta['city_rows'][city]} daily records exist for {city}; treat with caution.")
    if note:
        print(f" {note}")
    print(LINE)
    print(" Health advice:")
    print("  " + HEALTH_ADVICE[bucket])
    print(LINE)


# ----------------------------- 1 & 2: predict -------------------------------
def predict_flow(models, meta, mode):
    city = ask_city(meta)
    month = ask_month()
    feats = meta["features"][mode]
    pollutants = [f for f in feats if f in meta["pollutants"]]
    print(f"\nEnter pollutant levels for {city}. Press Enter for any value you don't know.")
    values = {p: ask_value(p, meta["medians"][city].get(p), meta["plausible_max"][p]) for p in pollutants}

    row = pd.DataFrame([{"City": city, **values, "Month": month}])[feats]
    pred = float(np.clip(models[mode].predict(row)[0], 0, AQI_CAP))
    band = meta["error80"][mode][to_bucket(pred)[0]]
    given = sum(not np.isnan(v) for v in values.values())
    print_result(city, f"Month: {MONTHS[month - 1]}", pred, band, meta["city_median_aqi"][city],
                 given, len(pollutants), dominant_pollutant(values), meta)


# ----------------------------- 3: replay a real day -------------------------
def replay_day(models, meta):
    if not CLEAN_PATH.exists():
        print("\noutputs/air_quality_clean.csv not found. Run train_model.py first.")
        return
    data = pd.read_csv(CLEAN_PATH, parse_dates=["Date"])
    cutoff = pd.Timestamp(meta["cutoff"])
    city = ask_city(meta)
    sub = data[data["City"] == city]
    lo, hi = sub["Date"].min().date(), sub["Date"].max().date()
    print(f"\nData for {city}: {lo} to {hi}. The model was trained on days before {cutoff.date()}.")
    while True:
        text = input("Enter a date (YYYY-MM-DD) or press Enter for a random unseen day: ").strip()
        if text == "":
            pool = sub[(sub["Date"] >= cutoff) & sub["AQI_next"].notna()]
            if pool.empty:
                pool = sub[sub["AQI_next"].notna()]
            row = pool.iloc[random.randrange(len(pool))]
            break
        try:
            match = sub[sub["Date"] == pd.Timestamp(text)]
        except ValueError:
            print("  Please use the format YYYY-MM-DD.")
            continue
        if match.empty:
            print(f"  No record for that date. Try a date between {lo} and {hi}.")
            continue
        row = match.iloc[0]
        break

    date = row["Date"]
    shown = {p: row[p] for p in meta["pollutants"]}
    full = pd.DataFrame([row])[meta["features"]["full"]]
    pred = float(np.clip(models["full"].predict(full)[0], 0, AQI_CAP))
    actual = float(row["AQI"])
    seen = date < cutoff or meta["refit_on_all_data"]

    print("\n" + LINE)
    print(f" {city}, {date.date()}   ({'IN the training data' if seen else 'NOT seen by the model during training'})")
    print(LINE)
    print(" Readings taken from the dataset (nothing typed):")
    line = ""
    for p, v in shown.items():
        line += f"  {p}: {'missing' if pd.isna(v) else f'{v:g}':<9}"
        if len(line) > 56:
            print(line); line = ""
    if line:
        print(line)
    print(LINE)
    pb, ab = to_bucket(pred)[0], to_bucket(actual)[0]
    print(f" Same-day  predicted {pred:5.0f} ({pb})   |   actual {actual:5.0f} ({ab})   |   error {pred - actual:+.0f}")
    if pd.notna(row["AQI_next"]):
        nd = pd.DataFrame([row])[meta["features"]["next_day"]]
        pred_n = float(np.clip(models["next_day"].predict(nd)[0], 0, AQI_CAP))
        act_n = float(row["AQI_next"])
        print(f" Next-day  predicted {pred_n:5.0f} ({to_bucket(pred_n)[0]})   |   actual {act_n:5.0f} "
              f"({to_bucket(act_n)[0]})   |   error {pred_n - act_n:+.0f}")
        print(f" Next-day  'tomorrow = today' baseline would say {actual:.0f}  (error {actual - act_n:+.0f})")
    else:
        print(" Next-day forecast: no consecutive next day in the data for this date.")
    print(LINE)


# ----------------------------- 4: city insights -----------------------------
def city_insights(meta):
    city = ask_city(meta)
    monthly = meta["monthly_median_aqi"][city]
    share = meta["category_share"][city]
    lo, hi = meta["city_period"][city]
    print("\n" + LINE)
    print(f" {city}: {meta['city_rows'][city]:,} daily records ({lo} to {hi}), median AQI {meta['city_median_aqi'][city]:.0f}")
    print(LINE)
    print(" Median AQI by month:")
    for m in range(1, 13):
        v = monthly.get(m)
        if v is None or np.isnan(v):
            print(f"  {MONTHS[m - 1][:3]}  no data")
        else:
            print(f"  {MONTHS[m - 1][:3]}  [{color_bar(v, AQI_CAP, to_bucket(v)[0], 30)}] {v:.0f}")
    valid = {m: v for m, v in monthly.items() if not np.isnan(v)}
    if valid:
        best, worst = min(valid, key=valid.get), max(valid, key=valid.get)
        print(f"\n Cleanest month: {MONTHS[best - 1]} ({valid[best]:.0f})   Most polluted: {MONTHS[worst - 1]} ({valid[worst]:.0f})")
    print("\n Share of days by category:")
    for b in BUCKETS:
        print(f"  {b:<13} {share[b]:5.1f}%  {'#' * int(share[b] / 2.5)}")
    if meta["city_rows"][city] < LOW_DATA_ROWS:
        print("\n NOTE: limited data, months may be missing or based on few days.")
    print(LINE)


# ----------------------------- 5: compare cities ----------------------------
def compare_cities(meta):
    month = ask_month()
    rows = [(c, meta["monthly_median_aqi"][c].get(month, np.nan)) for c in meta["cities"]]
    rows = sorted([r for r in rows if not np.isnan(r[1])], key=lambda r: r[1], reverse=True)
    print("\n" + LINE)
    print(f" Median AQI in {MONTHS[month - 1]} (worst to best)")
    print(LINE)
    for c, v in rows:
        flag = "*" if meta["city_rows"][c] < LOW_DATA_ROWS else " "
        print(f" {c + flag:<20}[{color_bar(v, AQI_CAP, to_bucket(v)[0], 30)}] {v:.0f}")
    print(" * limited data. Cities cover different years, so rankings are indicative only.")
    print(LINE)


# ----------------------------- 6: performance -------------------------------
def show_performance():
    if not METRICS_PATH.exists():
        print("\nmetrics.json not found. Run train_model.py first.")
        return
    with open(METRICS_PATH) as f:
        m = json.load(f)
    d = m["dataset"]
    print("\n" + LINE)
    print(f" {d['rows_after_cleaning']:,} daily records, {d['cities']} cities ({d['date_from']} to {d['date_to']})")
    print(f" Test period (never seen in training): {d['test_period_starts']} onwards. Best model: {m['best_model']}")
    print(LINE)
    print("\nSame-day AQI (train on the past, test on the future):")
    print(pd.DataFrame(m["same_day"]).T.to_string())
    print("\nSame-day AQI with a random split (optimistic, neighbouring days leak):")
    print(pd.DataFrame(m["same_day_random_split"]).T.to_string())
    cv = m["cross_validation_r2"]
    print(f"\nTime-series cross-validation R2: {cv['mean']:.3f} +/- {cv['std']:.3f}  (folds: {cv['folds']})")
    print("\nHow many inputs do you need to type?")
    print(pd.DataFrame(m["input_effort"]).T.to_string())
    print("\nNext-day AQI forecast:")
    print(pd.DataFrame(m["next_day"]).T.to_string())
    print("\nMost important inputs (R2 lost when shuffled):")
    for name, val in list(m["feature_importance"].items())[:6]:
        print(f"  {name:<10} {'#' * int(val * 40):<40} {val:.3f}")
    print("\nPlots are saved in the 'outputs' folder.")


def main():
    models, meta = load_bundle()
    print("\n" + LINE)
    print("   AirSense India: Air Quality Prediction (SDG 3)")
    print(LINE)
    actions = {
        "1": lambda: predict_flow(models, meta, "quick"),
        "2": lambda: predict_flow(models, meta, "full"),
        "3": lambda: replay_day(models, meta),
        "4": lambda: city_insights(meta),
        "5": lambda: compare_cities(meta),
        "6": show_performance,
    }
    while True:
        print("\n  1) Quick prediction       (only PM2.5 and PM10)")
        print("  2) Detailed prediction    (all pollutants, skip what you don't know)")
        print("  3) Replay a real day      (readings filled in automatically, compare with actual)")
        print("  4) City insights          (monthly pattern, share of bad days)")
        print("  5) Compare cities         (for a chosen month)")
        print("  6) Model performance")
        print("  7) Quit")
        choice = input("\nChoose an option [1-7]: ").strip()
        if choice == "7":
            print("Goodbye!")
            break
        if choice in actions:
            actions[choice]()
        else:
            print("Please type a number from 1 to 7.")


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
