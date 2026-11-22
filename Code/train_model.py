"""
AirSense India: Air Quality Prediction (SDG 3 - Good Health and Well-being)
===========================================================================
Complete pipeline in one script:
  1. Load data              6. Compare models (rule-based formula, Linear, Random Forest, Gradient Boosting)
  2. Clean (no leakage)     7. "How much input is really needed?" experiment
  3. EDA plots              8. Cross-validation, permutation importance, confusion matrix
  4. Time-based split       9. Next-day AQI forecast (with lag features)
  5. Pipelines             10. Save models + metrics for predict.py

Run:   python train_model.py
Needs: data/city_day.csv  (Kaggle: rohanrao/air-quality-data-in-india)
"""
import json
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # save figures to files; works on any machine, no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from common import (AQI_CAP, BUCKET_COLORS, BUCKETS, POLLUTANTS, SEASON_ORDER, SEASONS,
                    bucket_index, cpcb_aqi, to_bucket)

warnings.filterwarnings("ignore", category=FutureWarning)
sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DATA_PATH = Path("data/city_day.csv")
OUT_DIR = Path("outputs")        # plots + metrics.json + cleaned data
MODEL_DIR = Path("models")       # saved models
MISSING_THRESHOLD = 0.50         # drop pollutant columns with more than 50% missing values
TEST_SIZE = 0.20                 # the LATEST 20% of dates are the test set
RANDOM_STATE = 42
# Ahmedabad's records are unreliable (mean CO ~22 mg/m3 vs ~1.2 elsewhere, AQI far above the
# 500 scale, PM10 mostly missing). Set to [] to keep it and compare the results.
EXCLUDE_CITIES = ["Ahmedabad"]
# False: the saved models are trained only on the training period, so the numbers in metrics.json
# describe exactly the saved models and "Replay a real day" is an honest test.
# True : refit on all data before saving (slightly better live predictions, replay is then in-sample).
REFIT_ON_ALL_DATA = False

MODELS = {
    "Linear Regression": lambda: LinearRegression(),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": lambda: HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.08, random_state=RANDOM_STATE),
}
FORMULA = "CPCB formula (rule-based, no learning)"


# ----------------------------------------------------------------------------
# 1. LOAD
# ----------------------------------------------------------------------------
def load_data():
    if not DATA_PATH.exists():
        raise SystemExit(
            f"\nDataset not found at '{DATA_PATH}'.\n"
            "Download city_day.csv from\n"
            "  https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india\n"
            "and place it inside the 'data' folder (create the folder if needed).\n")
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    missing_cols = {"City", "Date", "AQI"} - set(df.columns)
    if missing_cols:
        raise SystemExit(f"city_day.csv is missing required columns: {missing_cols}")
    print(f"Loaded {df.shape[0]:,} rows, {df['City'].nunique()} cities, "
          f"{df['Date'].min().date()} to {df['Date'].max().date()}")
    return df


# ----------------------------------------------------------------------------
# 2. CLEAN  (NOTE: no imputation here. Missing values stay NaN and are handled
#    INSIDE the model pipelines, so nothing from the test period leaks into training.)
# ----------------------------------------------------------------------------
def preprocess(df):
    df = df.dropna(subset=["AQI"]).copy()                      # rows without the target are useless
    info = {"rows_with_aqi": int(len(df)), "aqi_above_500": int((df["AQI"] > AQI_CAP).sum())}

    if EXCLUDE_CITIES:
        info["excluded_rows"] = int(df["City"].isin(EXCLUDE_CITIES).sum())
        df = df[~df["City"].isin(EXCLUDE_CITIES)].copy()
        print(f"Excluded unreliable cities: {EXCLUDE_CITIES} ({info['excluded_rows']:,} rows)")

    present = [c for c in POLLUTANTS if c in df.columns]
    missing_share = df[present].isna().mean().sort_values(ascending=False)
    dropped = [c for c in present if missing_share[c] > MISSING_THRESHOLD]
    pollutants = [c for c in present if c not in dropped]
    df = df.drop(columns=dropped)
    print(f"Dropped mostly-empty columns: {dropped}\nPollutants used: {pollutants}")

    df["AQI"] = df["AQI"].clip(upper=AQI_CAP)                  # the official scale ends at 500
    df["Month"] = df["Date"].dt.month
    df["Season"] = df["Month"].map(SEASONS)
    df = add_lag_features(df)
    print(f"After cleaning: {len(df):,} rows; missing pollutant cells kept as NaN: "
          f"{int(df[pollutants].isna().sum().sum()):,}")
    return df, pollutants, dropped, missing_share, info


def add_lag_features(df):
    """Yesterday's AQI, 7-day average and tomorrow's AQI (only across truly consecutive days)."""
    parts = []
    for _, g in df.sort_values(["City", "Date"]).groupby("City"):
        g = g.copy()
        g["AQI_prev"] = g["AQI"].shift(1).where(g["Date"].diff().dt.days == 1)
        g["AQI_next"] = g["AQI"].shift(-1).where((g["Date"].shift(-1) - g["Date"]).dt.days == 1)
        g["AQI_mean7"] = g.set_index("Date")["AQI"].rolling("7D", min_periods=3).mean().to_numpy()
        parts.append(g)
    return pd.concat(parts).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. EDA
# ----------------------------------------------------------------------------
def save_fig(name):
    plt.tight_layout()
    plt.savefig(OUT_DIR / name, dpi=150, bbox_inches="tight")
    plt.close()


def run_eda(df, pollutants, missing_share):
    # 0 Missing data per pollutant (why imputation matters)
    plt.figure(figsize=(8, 4.5))
    ms = missing_share[pollutants + [c for c in missing_share.index if c not in pollutants]] * 100
    plt.barh(ms.index[::-1], ms.values[::-1], color="#534AB7")
    plt.axvline(MISSING_THRESHOLD * 100, color="r", ls="--", lw=1, label="drop threshold")
    plt.xlabel("% of rows missing")
    plt.title("Missing values per pollutant")
    plt.legend()
    save_fig("eda_0_missing_values.png")

    # 1 AQI distribution + days per category
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["AQI"], bins=40, kde=True, ax=ax[0], color="#534AB7")
    ax[0].set_title("AQI distribution (capped at 500)")
    counts = pd.Series(to_bucket(df["AQI"])).value_counts().reindex(BUCKETS, fill_value=0)
    ax[1].bar(BUCKETS, counts.values, color=[BUCKET_COLORS[b] for b in BUCKETS])
    ax[1].set_title("Number of days per AQI category")
    ax[1].tick_params(axis="x", rotation=30)
    save_fig("eda_1_aqi_distribution.png")

    # 2 Most polluted cities
    top = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)[::-1]
    plt.figure(figsize=(8, 4.5))
    plt.barh(top.index, top.values, color="#534AB7")
    plt.title("Top 10 cities by average AQI")
    plt.xlabel("Average AQI")
    save_fig("eda_2_top_cities.png")

    # 3 Seasonal pattern
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(data=df, x="Season", y="AQI", order=SEASON_ORDER, color="#9FE1CB")
    plt.title("AQI by season")
    save_fig("eda_3_seasonal_aqi.png")

    # 4 Delhi monthly trend
    focus = "Delhi" if "Delhi" in df["City"].values else df["City"].mode()[0]
    trend = df[df["City"] == focus].set_index("Date")["AQI"].resample("MS").mean()
    plt.figure(figsize=(10, 4))
    trend.plot(color="#0F6E56")
    plt.title(f"{focus}: monthly average AQI")
    plt.ylabel("AQI")
    save_fig("eda_4_city_trend.png")

    # 5 Correlation heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[pollutants + ["AQI"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation between pollutants and AQI")
    save_fig("eda_5_correlation.png")

    # 6 City x month heatmap (median AQI)
    pv = df.pivot_table(index="City", columns="Month", values="AQI", aggfunc="median")
    pv = pv.loc[pv.mean(axis=1).sort_values(ascending=False).index]
    plt.figure(figsize=(11, 8))
    sns.heatmap(pv, annot=True, fmt=".0f", cmap="YlOrRd", annot_kws={"size": 7},
                cbar_kws={"label": "median AQI"})
    plt.title("Median AQI by city and month")
    save_fig("eda_6_city_month_heatmap.png")
    print("EDA plots saved to", OUT_DIR)


# ----------------------------------------------------------------------------
# 4-5. SPLIT, PIPELINES, METRICS
# ----------------------------------------------------------------------------
def time_split(df):
    """Train on the past, test on the latest 20% of dates (this is how a real system is used)."""
    cutoff = df["Date"].sort_values().iloc[int(len(df) * (1 - TEST_SIZE))]
    return cutoff, df[df["Date"] < cutoff], df[df["Date"] >= cutoff]


def make_pipeline(kind, features):
    """One-hot City + numeric columns. Imputation (if the model needs it) is fitted on training data only.
    Gradient Boosting handles NaN natively, so it needs no imputation at all."""
    numeric = [c for c in features if c != "City"]
    num_step = "passthrough" if kind == "Gradient Boosting" else SimpleImputer(strategy="median")
    prep = ColumnTransformer(
        [("city", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["City"]),
         ("num", num_step, numeric)], sparse_threshold=0)
    return Pipeline([("prep", prep), ("model", MODELS[kind]())])


def predict_aqi(pipe, X):
    """Same clipping that predict.py uses, so reported metrics match what the user sees."""
    return np.clip(pipe.predict(X), 0, AQI_CAP)


def score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    gap = np.abs(bucket_index(y_true) - bucket_index(y_pred))
    return {
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 2),
        "Category accuracy": round(float((gap == 0).mean()), 4),
        "Within 1 category": round(float((gap <= 1).mean()), 4),
    }


def error_bands(y_true, y_pred, q=0.8):
    """For each PREDICTED category: the size of error that 80% of test predictions stay within."""
    err = np.abs(np.asarray(y_true, float) - np.asarray(y_pred, float))
    idx = bucket_index(y_pred)
    overall = float(np.quantile(err, q))
    return {b: round(float(np.quantile(err[idx == i], q)) if (idx == i).sum() >= 30 else overall, 1)
            for i, b in enumerate(BUCKETS)}


# ----------------------------------------------------------------------------
# 6-8. SAME-DAY MODELS
# ----------------------------------------------------------------------------
def compare_models(df, pollutants, features, cutoff, train, test):
    # (a) honest evaluation: time-based split
    res_time, preds = {}, {}
    formula = cpcb_aqi(test).fillna(train["AQI"].median()).clip(0, AQI_CAP)
    res_time[FORMULA] = score(test["AQI"], formula)
    for kind in MODELS:
        pipe = make_pipeline(kind, features).fit(train[features], train["AQI"])
        preds[kind] = predict_aqi(pipe, test[features])
        res_time[kind] = score(test["AQI"], preds[kind])
    best = min(MODELS, key=lambda k: res_time[k]["RMSE"])

    # (b) optimistic evaluation: random split (neighbouring days leak between train and test)
    tr_r, te_r = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    res_rand = {}
    for kind in MODELS:
        pipe = make_pipeline(kind, features).fit(tr_r[features], tr_r["AQI"])
        res_rand[kind] = score(te_r["AQI"], predict_aqi(pipe, te_r[features]))

    print("\n=== Same-day AQI, TIME split (train = before", cutoff.date(), ") ===")
    print(pd.DataFrame(res_time).T.to_string())
    print("\n=== Same-day AQI, RANDOM split (optimistic) ===")
    print(pd.DataFrame(res_rand).T.to_string())
    print(f"\nBest model by RMSE: {best}")
    return res_time, res_rand, best, preds[best]


def cross_validate(df, best, features):
    d = df.sort_values("Date")
    r2 = cross_val_score(make_pipeline(best, features), d[features], d["AQI"],
                         cv=TimeSeriesSplit(n_splits=5), scoring="r2")
    print("\nTime-series CV R2 per fold:", np.round(r2, 3), f"| mean {r2.mean():.3f} +/- {r2.std():.3f}")
    return {"folds": [round(float(x), 4) for x in r2], "mean": round(float(r2.mean()), 4),
            "std": round(float(r2.std()), 4)}


def input_effort_experiment(best, pollutants, features_full, train, test):
    """How accurate is the model if the user only knows a few things? (answers 'do I have to type everything?')"""
    levels = {
        "City + month only": ["City", "Month"],
        "Quick: PM2.5 + PM10": ["City", "PM2.5", "PM10", "Month"],
        "Medium: + NO2 and CO": ["City", "PM2.5", "PM10", "NO2", "CO", "Month"],
        "Full: all pollutants": features_full,
    }
    results, quick_pred = {}, None
    for name, feats in levels.items():
        feats = [f for f in feats if f in features_full]
        pipe = make_pipeline(best, feats).fit(train[feats], train["AQI"])
        pred = predict_aqi(pipe, test[feats])
        results[name] = {"inputs": len(feats) - 2, **score(test["AQI"], pred)}
        if name.startswith("Quick"):
            quick_pred, quick_feats = pred, feats
    print("\n=== Accuracy vs number of inputs typed (", best, ") ===")
    print(pd.DataFrame(results).T.to_string())
    return results, quick_feats, quick_pred


def plot_model_results(pipe, features, test, pred, results_effort):
    y = test["AQI"].to_numpy()
    # actual vs predicted
    plt.figure(figsize=(6, 6))
    plt.scatter(y, pred, s=6, alpha=0.35, color="#534AB7")
    plt.plot([0, AQI_CAP], [0, AQI_CAP], "r--", lw=1, label="Perfect prediction")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title("Test period: actual vs predicted AQI")
    plt.legend()
    save_fig("model_actual_vs_predicted.png")

    # permutation importance: how much does R2 drop when one input is shuffled?
    imp = permutation_importance(pipe, test[features], test["AQI"], n_repeats=5,
                                 random_state=RANDOM_STATE, scoring="r2")
    importance = dict(sorted(((f, max(float(m), 0.0)) for f, m in zip(features, imp.importances_mean)),
                             key=lambda kv: kv[1], reverse=True))
    plt.figure(figsize=(7, 5))
    plt.barh(list(importance)[::-1], list(importance.values())[::-1], color="#0F6E56")
    plt.title("Permutation importance (drop in R2 when shuffled)")
    plt.xlabel("R2 drop")
    save_fig("model_feature_importance.png")

    # confusion matrix by AQI category
    cm = confusion_matrix(bucket_index(y), bucket_index(pred), labels=range(6)).astype(float)
    cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(cm_pct, annot=True, fmt=".2f", cmap="Blues", xticklabels=BUCKETS, yticklabels=BUCKETS)
    plt.xlabel("Predicted category")
    plt.ylabel("Actual category")
    plt.title("Confusion matrix (rows sum to 1)")
    save_fig("model_confusion_matrix.png")

    # accuracy vs effort
    names = list(results_effort)
    plt.figure(figsize=(8, 4.2))
    plt.bar(names, [results_effort[n]["MAE"] for n in names], color="#534AB7")
    plt.ylabel("Mean absolute error (AQI points)")
    plt.title("Error vs number of inputs the user types")
    plt.xticks(rotation=15)
    save_fig("model_input_effort.png")
    return {k: round(v, 4) for k, v in importance.items()}


# ----------------------------------------------------------------------------
# 9. NEXT-DAY FORECAST
# ----------------------------------------------------------------------------
def next_day_experiment(df, pollutants, best, cutoff):
    d = df.dropna(subset=["AQI_next"])
    train, test = d[d["Date"] < cutoff], d[d["Date"] >= cutoff]
    base = ["City"] + pollutants + ["AQI", "Month"]
    lag = base + ["AQI_prev", "AQI_mean7"]
    results = {"Persistence baseline (tomorrow = today)": score(test["AQI_next"], test["AQI"])}
    for label, feats in [("Model: today's readings", base), ("Model: + yesterday & 7-day mean", lag)]:
        pipe = make_pipeline(best, feats).fit(train[feats], train["AQI_next"])
        pred = predict_aqi(pipe, test[feats])
        results[label] = score(test["AQI_next"], pred)
    bands = error_bands(test["AQI_next"], pred)
    print(f"\n=== Next-day AQI forecast ({best}; test = days from {cutoff.date()}) ===")
    print(pd.DataFrame(results).T.to_string())
    return results, lag, bands, train


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    raw = load_data()
    df, pollutants, dropped, missing_share, info = preprocess(raw)
    run_eda(df, pollutants, missing_share)
    df.to_csv(OUT_DIR / "air_quality_clean.csv", index=False)

    features = ["City"] + pollutants + ["Month"]
    cutoff, train, test = time_split(df)
    print(f"\nTrain: {len(train):,} rows (before {cutoff.date()}) | Test: {len(test):,} rows")

    res_time, res_rand, best, best_pred = compare_models(df, pollutants, features, cutoff, train, test)
    cv = cross_validate(df, best, features)
    effort, quick_feats, quick_pred = input_effort_experiment(best, pollutants, features, train, test)
    full_pipe = make_pipeline(best, features).fit(train[features], train["AQI"])
    importance = plot_model_results(full_pipe, features, test, best_pred, effort)
    nd_results, nd_feats, nd_bands, nd_train = next_day_experiment(df, pollutants, best, cutoff)

    # ---- final models that predict.py loads ----
    fit_df = df if REFIT_ON_ALL_DATA else train
    fit_nd = df.dropna(subset=["AQI_next"]) if REFIT_ON_ALL_DATA else nd_train
    models = {
        "full": make_pipeline(best, features).fit(fit_df[features], fit_df["AQI"]),
        "quick": make_pipeline(best, quick_feats).fit(fit_df[quick_feats], fit_df["AQI"]),
        "next_day": make_pipeline(best, nd_feats).fit(fit_nd[nd_feats], fit_nd["AQI_next"]),
    }
    g = df.groupby("City")
    share = (df.assign(Cat=to_bucket(df["AQI"])).groupby("City")["Cat"]
             .value_counts(normalize=True).unstack(fill_value=0).reindex(columns=BUCKETS, fill_value=0))
    meta = {
        "features": {"full": features, "quick": quick_feats, "next_day": nd_feats},
        "pollutants": pollutants,
        "dropped_columns": dropped,
        "excluded_cities": EXCLUDE_CITIES,
        "cities": sorted(df["City"].unique().tolist()),
        "best_model": best,
        "cutoff": str(cutoff.date()),
        "refit_on_all_data": REFIT_ON_ALL_DATA,
        "unseen_cities": sorted(set(df["City"]) - set(train["City"])),
        "medians": g[pollutants].median().round(2).to_dict(orient="index"),
        "plausible_max": {p: round(float(df[p].quantile(0.999)), 1) for p in pollutants},
        "city_median_aqi": g["AQI"].median().round(0).to_dict(),
        "city_rows": g.size().to_dict(),
        "city_period": {c: (str(x.min().date()), str(x.max().date())) for c, x in g["Date"]},
        "monthly_median_aqi": df.groupby(["City", "Month"])["AQI"].median().round(0)
                                .unstack().to_dict(orient="index"),
        "category_share": (share * 100).round(1).to_dict(orient="index"),
        "error80": {"full": error_bands(test["AQI"], best_pred),
                    "quick": error_bands(test["AQI"], quick_pred),
                    "next_day": nd_bands},
    }
    joblib.dump({"models": models, "meta": meta}, MODEL_DIR / "aqi_model.joblib", compress=3)

    summary = {
        "dataset": {
            "rows_after_cleaning": int(len(df)), "cities": int(df["City"].nunique()),
            "date_from": str(df["Date"].min().date()), "date_to": str(df["Date"].max().date()),
            "test_period_starts": str(cutoff.date()),
            "dropped_columns": dropped, "excluded_cities": EXCLUDE_CITIES, **info,
        },
        "best_model": best,
        "same_day": res_time,
        "same_day_random_split": res_rand,
        "cross_validation_r2": cv,
        "input_effort": effort,
        "next_day": nd_results,
        "feature_importance": importance,
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    size_mb = (MODEL_DIR / "aqi_model.joblib").stat().st_size / 1e6
    print(f"\nModels saved to {MODEL_DIR / 'aqi_model.joblib'} ({size_mb:.1f} MB)")
    print("Metrics saved to", OUT_DIR / "metrics.json")
    print("\nDone. Now run:  python predict.py")


if __name__ == "__main__":
    main()
