"""
Backtest: how much should an older season count?

Pretend it's October 2025. Using only 2023-24 and 2024-25, predict each
player's 2025-26 BPM, then compare with what actually happened.
Try different weights for the older season and different shrink strengths,
and keep whichever predicts best.
"""
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"
BPM_PRIOR = -1.0
AGE_GROUP_TRUST = 20


def key(name):
    k = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower().replace(".", "").strip()
    for suffix in (" jr", " sr", " ii", " iii", " iv"):
        if k.endswith(suffix):
            k = k[: -len(suffix)]
    return k


def read_season(fname):
    raw = pd.read_csv(DATA / fname)
    raw = raw[raw["Player"] != "League Average"].dropna(subset=["Team"])
    raw["key"] = raw["Player"].map(key)
    return raw.groupby("key", sort=False).first()[["Player", "Age", "G", "MP", "BPM"]]


s24, s25, s26 = (read_season(f"advanced_{y}.csv") for y in ["2023_24", "2024_25", "2025_26"])

# ---- Age curve, learned ONLY from 2023-24 -> 2024-25 (no peeking at the answer)
both = s24.join(s25, lsuffix="_old", rsuffix="_new", how="inner")
both = both[(both["MP_old"] >= 500) & (both["MP_new"] >= 500)]
both["change"] = both["BPM_new"] - both["BPM_old"]
bins = [18, 21, 23, 25, 27, 29, 31, 33, 35, 43]
grp = both.groupby(pd.cut(both["Age_old"], bins), observed=False)["change"].agg(["count", "mean"]).fillna(0)
shrunk = grp["mean"] * grp["count"] / (grp["count"] + AGE_GROUP_TRUST)
mids = [(lo + hi) / 2 for lo, hi in zip(bins[:-1], bins[1:])]
age_change = lambda age: float(np.interp(age, mids, shrunk))

# ---- Who we test on: 500+ minutes in 2025-26 and at least one earlier season
target = s26[s26["MP"] >= 500]
target = target[target.index.isin(s25.index) | target.index.isin(s24.index)]


def predict(k, w_older, prior_min):
    num = mins = 0.0
    for df, w, years_ago in [(s25, 1.0, 1), (s24, w_older, 2)]:
        if k in df.index:
            r = df.loc[k]
            bpm_then = r["BPM"] + sum(age_change(r["Age"] + y) for y in range(years_ago - 1))
            num += w * bpm_then * r["MP"]
            mins += w * r["MP"]
    if mins + prior_min == 0:
        return np.nan  # no shrink and no data: nothing to predict from
    level = (num + BPM_PRIOR * prior_min) / (mins + prior_min)
    age_last = s25.loc[k, "Age"] if k in s25.index else s24.loc[k, "Age"] + 1
    return level + age_change(age_last)


def score(preds):
    """Minutes-weighted average error (RMSE): lower is better."""
    err = preds - target["BPM"]
    return float(np.sqrt(np.average(err ** 2, weights=target["MP"])))


if __name__ == "__main__":
    print(f"Testing on {len(target)} players (500+ minutes in 2025-26)\n")

    naive = target.index.map(lambda k: s25.loc[k, "BPM"] if k in s25.index else s24.loc[k, "BPM"])
    print(f"Baselines:  everyone = average (0):   {score(pd.Series(0.0, index=target.index)):.2f}")
    print(f"            last season's raw BPM:    {score(pd.Series(naive, index=target.index)):.2f}\n")

    weights = [0.0, 0.25, 0.5, 0.75, 1.0]
    prior_mins = [0, 100, 250, 500, 1000]
    table = pd.DataFrame(index=[f"older season x{w}" for w in weights],
                         columns=[f"shrink {m}" for m in prior_mins], dtype=float)
    for w in weights:
        for m in prior_mins:
            preds = pd.Series([predict(k, w, m) for k in target.index], index=target.index)
            table.loc[f"older season x{w}", f"shrink {m}"] = score(preds)
    print("Prediction error (RMSE in BPM points; lower = better)")
    print(table.round(3).to_string())
    best = table.stack().idxmin()
    print(f"\nBest: {best[0]}, {best[1]} minutes  ->  error {table.stack().min():.3f}")
