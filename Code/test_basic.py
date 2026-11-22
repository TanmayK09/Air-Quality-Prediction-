"""Quick sanity tests.  Run:  python test_basic.py"""
import numpy as np
from common import bucket_index, cpcb_aqi, dominant_pollutant, sub_index, to_bucket
import pandas as pd

assert list(to_bucket([0, 50, 51, 100, 101, 200, 201, 300, 301, 400, 401, 500])) == [
    "Good", "Good", "Satisfactory", "Satisfactory", "Moderate", "Moderate",
    "Poor", "Poor", "Very Poor", "Very Poor", "Severe", "Severe"]
assert list(bucket_index([10, 450])) == [0, 5]
assert abs(sub_index("PM2.5", 30) - 50) < 1e-9 and abs(sub_index("PM2.5", 90) - 200) < 1e-9
assert sub_index("PM2.5", 5000) == 500                      # capped at scale maximum
assert np.isnan(sub_index("PM10", np.nan))
df = pd.DataFrame({"PM2.5": [60.0, np.nan], "PM10": [250.0, np.nan], "CO": [1.0, np.nan]})
assert cpcb_aqi(df).iloc[0] == 200 and np.isnan(cpcb_aqi(df).iloc[1])   # max of sub-indices
assert dominant_pollutant({"PM2.5": 60, "PM10": 250, "CO": np.nan})[0] == "PM10"
assert dominant_pollutant({"PM2.5": np.nan}) is None
print("All tests passed.")
