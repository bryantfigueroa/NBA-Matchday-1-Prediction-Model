"""
NBA Week 1 prediction model, v1 (archived): the first, simpler version.
Kept to show where the project started; the current model is ../model.py.

Predicted margin (home POV) = home_rating - away_rating + HOME_COURT
Win probability             = NormalCDF(margin / MARGIN_SD)

Two ways to rate each team, plus a blend:
  - stats:  last season's point differential per game, shrunk toward 0
  - market: preseason betting win total, converted into points per game
  - blend:  weighted average of the two (this is what makes the pick)
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent.parent / "data"   # shared data folder

# ---- Knobs you can turn ---------------------------------------------------
REGRESS = 0.65      # keep 65% of last season's margin, shrink 35% toward average
HOME_COURT = 2.3    # points; recent NBA home edge is roughly 2-2.5
MARGIN_SD = 12.5    # how far real results scatter around the predicted margin
MARKET_WEIGHT = 0.5 # 0 = stats only, 1 = market only


# ---- Chunk 1: load data ---------------------------------------------------
last = pd.read_csv(DATA / "last_season_2025_26.csv")
totals = pd.read_csv(DATA / "win_totals_2026_27.csv")
games = pd.read_csv(DATA / "first_games.csv")

teams = last.merge(totals, on="team", validate="one_to_one")
assert len(teams) == 30, "every team needs both a last-season row and a win total"


# ---- Chunk 2: stats rating ------------------------------------------------
teams["last_mov"] = teams["point_diff_total"] / (teams["wins"] + teams["losses"])
teams["stats_rating"] = REGRESS * teams["last_mov"]


# ---- Chunk 3: market rating -----------------------------------------------
# Learn the wins <-> margin exchange rate from last season instead of assuming it.
slope, intercept = np.polyfit(teams["last_mov"], teams["wins"], 1)
teams["market_rating"] = (teams["win_total"] - intercept) / slope


# ---- Chunk 4: blend -------------------------------------------------------
teams["rating"] = (MARKET_WEIGHT * teams["market_rating"]
                   + (1 - MARKET_WEIGHT) * teams["stats_rating"])
teams["roster_shift"] = teams["market_rating"] - teams["stats_rating"]


# ---- Chunk 5: predict games -----------------------------------------------
def win_prob(margin: float) -> float:
    """Normal CDF: chance a team favored by `margin` points actually wins."""
    return 0.5 * (1 + math.erf(margin / (MARGIN_SD * math.sqrt(2))))


r = teams.set_index("team")
rows = []
for g in games.itertuples():
    out = {"date": g.date, "away": g.away, "home": g.home}
    for col, label in [("stats_rating", "stats"), ("market_rating", "market"), ("rating", "blend")]:
        margin = r.at[g.home, col] - r.at[g.away, col] + HOME_COURT
        out[f"{label}_margin"] = margin
        out[f"{label}_home_p"] = win_prob(margin)
    pick_home = out["blend_margin"] > 0
    out["pick"] = g.home if pick_home else g.away
    out["pick_p"] = out["blend_home_p"] if pick_home else 1 - out["blend_home_p"]
    out["models_disagree"] = (out["stats_margin"] > 0) != (out["market_margin"] > 0)
    rows.append(out)
preds = pd.DataFrame(rows)


# ---- Chunk 6: report ------------------------------------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 200)
    print(f"Wins = {intercept:.1f} + {slope:.2f} x MOV  (fit on 2025-26)\n")

    print("Ratings (points per game vs. an average team)")
    show = teams.sort_values("rating", ascending=False)[
        ["team", "last_mov", "stats_rating", "win_total", "market_rating", "roster_shift", "rating"]]
    print(show.round(1).to_string(index=False), "\n")

    print("First-game predictions")
    view = preds.assign(
        game=preds["away"] + " @ " + preds["home"],
        spread=preds["blend_margin"].map(lambda m: f"home {m:+.1f}"),
        stats_p=(preds["stats_home_p"] * 100).round(0),
        market_p=(preds["market_home_p"] * 100).round(0),
        pick_pct=(preds["pick_p"] * 100).round(0),
        flag=preds["models_disagree"].map({True: "<-- disagree", False: ""}),
    )[["date", "game", "stats_p", "market_p", "spread", "pick", "pick_pct", "flag"]]
    print("(stats_p / market_p = home win % under each model)")
    print(view.to_string(index=False))

    preds.to_csv(Path(__file__).parent / "predictions_v1.csv", index=False)
