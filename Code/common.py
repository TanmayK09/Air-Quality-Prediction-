"""Shared constants and helpers for the AirSense India project."""
import numpy as np
import pandas as pd

# All pollutant columns that can appear in city_day.csv
POLLUTANTS = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
              "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
UNITS = {"CO": "mg/m3"}  # every other pollutant is in ug/m3
MONTHS = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]

# Month -> season (Indian climate)
SEASONS = {12: "Winter", 1: "Winter", 2: "Winter",
           3: "Summer", 4: "Summer", 5: "Summer",
           6: "Monsoon", 7: "Monsoon", 8: "Monsoon", 9: "Monsoon",
           10: "Post-monsoon", 11: "Post-monsoon"}
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Post-monsoon"]

# Official CPCB (India) AQI categories. The CPCB scale ends at 500.
AQI_CAP = 500
BUCKETS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
_EDGES = [50, 100, 200, 300, 400]  # upper edge of Good ... Very Poor (inclusive)

BUCKET_COLORS = {
    "Good": "#2e9e4f", "Satisfactory": "#8bc34a", "Moderate": "#f2c200",
    "Poor": "#f28c28", "Very Poor": "#e0432e", "Severe": "#8e1b1b",
}

HEALTH_ADVICE = {
    "Good": "Air quality is good. Enjoy outdoor activities as usual.",
    "Satisfactory": "Acceptable for most people. Unusually sensitive people may notice mild breathing discomfort.",
    "Moderate": "People with asthma, lung or heart disease, children and older adults may feel breathing "
                "discomfort. Limit long or heavy outdoor exertion.",
    "Poor": "Most people may feel discomfort on prolonged exposure. Sensitive groups should stay indoors "
            "and avoid outdoor exercise. Consider a mask outdoors.",
    "Very Poor": "Respiratory illness is likely on prolonged exposure. Avoid outdoor activity, keep windows "
                 "closed, and wear an N95 mask if you must go out.",
    "Severe": "Serious health risk even for healthy people. Stay indoors, avoid all outdoor exertion, use an "
              "air purifier if available, and seek medical help if you have trouble breathing.",
}


def bucket_index(values):
    """AQI value(s) -> category number 0..5 (0 = Good, 5 = Severe)."""
    v = np.atleast_1d(np.asarray(values, dtype=float))
    return np.digitize(v, _EDGES, right=True)


def to_bucket(values):
    """AQI value(s) -> CPCB category name(s) as a numpy array of strings."""
    return np.array(BUCKETS)[bucket_index(values)]


# ----------------------------------------------------------------------------
# CPCB sub-index formula (rule-based AQI). AQI = the HIGHEST sub-index among the
# pollutants that were measured. Each sub-index is a straight-line interpolation
# between the concentration breakpoints below (approximate CPCB 2014 table;
# the top value of each list is the point where the sub-index reaches 500).
# ----------------------------------------------------------------------------
_INDEX_POINTS = [0, 50, 100, 200, 300, 400, 500]
CPCB_BREAKPOINTS = {
    "PM2.5": [0, 30, 60, 90, 120, 250, 380],
    "PM10":  [0, 50, 100, 250, 350, 430, 510],
    "NO2":   [0, 40, 80, 180, 280, 400, 520],
    "O3":    [0, 50, 100, 168, 208, 748, 1000],
    "CO":    [0, 1, 2, 10, 17, 34, 50],
    "SO2":   [0, 40, 80, 380, 800, 1600, 2400],
    "NH3":   [0, 200, 400, 800, 1200, 1800, 2400],
}


def sub_index(pollutant, values):
    """Sub-index (0-500) of one pollutant. NaN stays NaN."""
    return np.interp(np.asarray(values, dtype=float), CPCB_BREAKPOINTS[pollutant], _INDEX_POINTS)


def cpcb_aqi(df):
    """Rule-based AQI for every row of a dataframe (NaN if no usable pollutant)."""
    subs = pd.DataFrame({p: sub_index(p, df[p].to_numpy(dtype=float))
                         for p in CPCB_BREAKPOINTS if p in df.columns}, index=df.index)
    return subs.max(axis=1, skipna=True)


def dominant_pollutant(values):
    """values = {pollutant: concentration or NaN}. Returns (name, sub_index) or None."""
    subs = {p: float(sub_index(p, v)) for p, v in values.items()
            if p in CPCB_BREAKPOINTS and v is not None and not np.isnan(v)}
    if not subs:
        return None
    name = max(subs, key=subs.get)
    return name, subs[name]
