"""
NBA week-1 prediction model: every team's first game of 2026-27.

Three team ratings (points per game vs. an average team):
  stats   last season's margin, shrunk toward 0 (shown for comparison only)
  market  preseason win total converted to points
  roster  build each 2026-27 roster from player stats, hand out 240 minutes,
          sum each player's impact x minutes

Final rating = MARKET_WEIGHT * market + (1 - MARKET_WEIGHT) * roster
Then each game gets two adjustments:
  injuries    players listed out/questionable
  new coach   NEW_COACH_PENALTY points against a team with a new head coach

Run:  python3 model.py                       (penalty from the knob below)
      python3 model.py --coach-penalty 1.0   (try a different penalty)
"""
import argparse
import math
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data"

# ---- Knobs ----------------------------------------------------------------
REGRESS = 0.65        # stats model: share of last season's margin we keep
HOME_COURT = 2.3
MARGIN_SD = 12.5
MARKET_WEIGHT = 0.5   # 0 = trust our roster math only, 1 = trust Vegas only

# Seasons of player stats, newest first, with how much each counts.
# To add an older season, download it and add a line here.
# Weights from backtest_weights.py: an older season was worth ~0.5-0.75 of the
# next one (results were flat across that range), so each step back is x0.6.
SEASONS = [("advanced_2025_26.csv", 1.0),
           ("advanced_2024_25.csv", 0.6),
           ("advanced_2023_24.csv", 0.36)]
AGE_GROUP_TRUST = 20  # players an age group needs before we fully trust its average change

BPM_PRIOR = -1.0      # what we assume about a player before seeing him play
BPM_PRIOR_MIN = 250   # how many minutes of evidence that prior is "worth"
REPLACEMENT = -2.0    # BPM of a freely available fill-in player
MPG_CAP = 36.0        # nobody projects above this
TEAM_MINUTES = 240.0  # 5 players x 48 minutes
QUESTIONABLE_WEIGHT = 0.5  # questionable = counts as half missing
NEW_COACH_PENALTY = 0.0    # points; 0 = off. Hypothesis to TEST, not a known effect
MATCHUP_SD = 1.5           # how big real head-to-head edges typically are (points); 0 = ignore h2h
PLAYOFF_GAME_WEIGHT = 0.5  # a playoff meeting counts as half a regular-season game

ABBR = {"ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BRK": "Brooklyn Nets",
        "CHO": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
        "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
        "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
        "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
        "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
        "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns",
        "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
        "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"}


def key(name: str) -> str:
    """Match names regardless of accents, periods and suffixes:
    'Nikola Vučević' == 'Nikola Vucevic', 'Walter Clayton Jr.' == 'Walter Clayton'."""
    k = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower().replace(".", "").strip()
    for suffix in (" jr", " sr", " ii", " iii", " iv"):
        if k.endswith(suffix):
            k = k[: -len(suffix)]
    return k


# ---- Chunk 1: one row per player per season --------------------------------
# Traded players have a combined "2TM/3TM" row plus one row per team, in the
# order they played there. Stats come from the combined row; team from the last.
def read_season(fname: str) -> pd.DataFrame:
    raw = pd.read_csv(DATA / fname)
    raw = raw[raw["Player"] != "League Average"].dropna(subset=["Team"])
    raw["key"] = raw["Player"].map(key)
    rows = []
    for _, grp in raw.groupby("key", sort=False):
        total = grp.iloc[0]
        rows.append({"key": total["key"], "player": total["Player"], "age": total["Age"],
                     "team": grp.iloc[-1]["Team"], "g": total["G"], "mp": total["MP"],
                     "bpm": total["BPM"]})
    return pd.DataFrame(rows).set_index("key")


def shrink(bpm_x_minutes, minutes):
    """Weighted average of a player's BPM, pulled toward the prior when minutes are few."""
    return (bpm_x_minutes + BPM_PRIOR * BPM_PRIOR_MIN) / (minutes + BPM_PRIOR_MIN)


# ---- Chunk 1a: how do players change with age? -----------------------------
# Compare the same players across back-to-back seasons (500+ minutes in
# both; every pair of seasons we have). Average the change by age group, then shrink each group's average
# toward 0 when it has few players (only 8 players were 36+).
def fit_age_curve(seasons_newest_first):
    pairs = []
    for newer, older in zip(seasons_newest_first[:-1], seasons_newest_first[1:]):
        both = older.join(newer, lsuffix="_old", rsuffix="_new", how="inner")
        pairs.append(both[(both["mp_old"] >= 500) & (both["mp_new"] >= 500)])
    both = pd.concat(pairs)
    both["change"] = both["bpm_new"] - both["bpm_old"]
    bins = [18, 21, 23, 25, 27, 29, 31, 33, 35, 43]
    grp = both.groupby(pd.cut(both["age_old"], bins), observed=False)["change"].agg(["count", "mean"])
    grp["mean"] = grp["mean"].fillna(0)
    grp["shrunk"] = grp["mean"] * grp["count"] / (grp["count"] + AGE_GROUP_TRUST)
    mids = [(lo + hi) / 2 for lo, hi in zip(bins[:-1], bins[1:])]
    change_at = lambda age: float(np.interp(age, mids, grp["shrunk"]))  # flat beyond the ends
    return change_at, grp, len(both)


# ---- Chunk 1b: combine seasons into one projection per player --------------
def load_players():
    seasons = [(read_season(f), w) for f, w in SEASONS]
    newest = seasons[0][0]
    age_change, age_table, n_pairs = fit_age_curve([df for df, _ in seasons])

    everyone = set().union(*(df.index for df, _ in seasons))
    rows = []
    for k in everyone:
        num = mins = games = 0.0
        for i, (df, w) in enumerate(seasons):
            if k in df.index:
                r = df.loc[k]
                # Bring an older season up to date: add the typical change for each
                # year since (a 20-year-old's 2024-25 understates his 2025-26 level).
                bpm_then = r["bpm"] + sum(age_change(r["age"] + y) for y in range(i))
                num += w * bpm_then * r["mp"]
                mins += w * r["mp"]
                games += w * r["g"]
        latest_i, latest = next((i, df.loc[k]) for i, (df, _) in enumerate(seasons) if k in df.index)
        age_now = latest["age"] + latest_i  # age as of 2025-26, even if he missed it
        last = newest.loc[k] if k in newest.index else None
        rows.append({
            "key": k, "player": latest["player"], "age": age_now,
            "team_2026": last["team"] if last is not None else None,
            "g": last["g"] if last is not None else 0, "mp": last["mp"] if last is not None else 0,
            "bpm_raw": last["bpm"] if last is not None else np.nan,
            "bpm_1yr": shrink(last["bpm"] * last["mp"], last["mp"]) if last is not None else np.nan,
            "bpm_now": shrink(num, mins),                       # all seasons combined
            "mpg": min(mins / games, MPG_CAP) if games else 0,  # season-weighted minutes per game
        })
    allp = pd.DataFrame(rows)
    allp["age_change"] = allp["age"].map(age_change)
    allp["bpm"] = allp["bpm_now"] + allp["age_change"]          # projection for 2026-27
    history = allp.set_index("key")
    players = allp[allp["team_2026"].notna()].reset_index(drop=True)  # only players active last season
    return players, history, age_table, n_pairs


# ---- Chunk 1c: what is a draft slot worth? --------------------------------
# Learn it from last year's rookies (2025 draft class), who just played a
# full season. Fit BPM and minutes per game against log(pick): the gap
# between pick 1 and pick 5 matters much more than between 25 and 30.
def rookie_curve(players: pd.DataFrame):
    d = pd.read_csv(DATA / "draft_2025.csv")
    d["key"] = d["player"].map(key)
    d = d.merge(players[["key", "bpm_now", "mpg", "mp"]], on="key", how="inner")  # drops rookies who never played
    x = np.log(d["pick"])
    bpm_fit = np.polyfit(x, d["bpm_now"], 1, w=np.sqrt(d["mp"]))  # trust players with more minutes more
    mpg_fit = np.polyfit(x, d["mpg"], 1)
    bpm_at = lambda pick: float(np.polyval(bpm_fit, np.log(pick)))
    mpg_at = lambda pick: float(np.clip(np.polyval(mpg_fit, np.log(pick)), 5, 32))
    return bpm_at, mpg_at, len(d)


# ---- Chunk 2: build rosters ------------------------------------------------
_warned = set()  # print each "skipped player" warning once

def build_rosters(players: pd.DataFrame, apply_moves: bool, add_rookies: bool = True) -> pd.DataFrame:
    p = players.copy()
    p["team"] = p["team_2026"]
    if apply_moves:
        moves = pd.read_csv(DATA / "transactions.csv")
        for m in moves.itertuples():
            hit = p["key"] == key(m.player)
            if not hit.any():
                if m.player not in _warned:
                    print(f"  ! skipped {m.player}: no 2025-26 NBA stats")
                    _warned.add(m.player)
                continue
            p.loc[hit, "team"] = m.new_team
        # Players who missed 2025-26 (e.g. a season-long injury). Blank bpm/mpg
        # in overrides.csv = use his older seasons; a number = your own estimate.
        for o in pd.read_csv(DATA / "overrides.csv").itertuples():
            h = history.loc[key(o.player)] if key(o.player) in history.index else None
            bpm = o.bpm if pd.notna(o.bpm) else h["bpm"]
            mpg = o.mpg if pd.notna(o.mpg) else h["mpg"]
            bpm_now = o.bpm if pd.notna(o.bpm) else h["bpm_now"]
            p = pd.concat([p, pd.DataFrame([{"key": key(o.player), "player": o.player, "team": o.team,
                                             "bpm": bpm, "bpm_now": bpm_now, "mpg": mpg}])])
    if apply_moves and add_rookies:
        draft = pd.read_csv(DATA / "draft_2026.csv")
        p = pd.concat([p, pd.DataFrame({
            "key": draft["player"].map(key), "player": draft["player"], "team": draft["team"],
            "bpm": draft["pick"].map(rookie_bpm), "bpm_now": draft["pick"].map(rookie_bpm),
            "mpg": draft["pick"].map(rookie_mpg)})])
    return p[p["team"] != "NONE"]


# ---- Chunk 3: hand out 240 minutes and sum impact --------------------------
# Rotation = most-used players first until 240 minutes are filled; the last
# man gets whatever is left. Impact per game = BPM x share of the game played.
def roster_strength(roster: pd.DataFrame, col: str = "bpm") -> pd.Series:
    """col='bpm_now' = players as they were last season; 'bpm' = projected for 2026-27."""
    out = {}
    for team, grp in roster.groupby("team"):
        grp = grp.sort_values("mpg", ascending=False)
        left, total = TEAM_MINUTES, 0.0
        for r in grp.itertuples():
            mins = min(r.mpg, left)
            total += getattr(r, col) * mins / 48
            left -= mins
            if left <= 0:
                break
        total += REPLACEMENT * max(left, 0) / 48  # thin roster: fill with replacement level
        out[ABBR[team]] = total
    return pd.Series(out)


# ---- Chunk 4: ratings -------------------------------------------------------
last = pd.read_csv(DATA / "last_season_2025_26.csv")
totals = pd.read_csv(DATA / "win_totals_2026_27.csv")
teams = last.merge(totals, on="team", validate="one_to_one")
teams["last_mov"] = teams["point_diff_total"] / (teams["wins"] + teams["losses"])
teams["stats_rating"] = REGRESS * teams["last_mov"]
slope, intercept = np.polyfit(teams["last_mov"], teams["wins"], 1)
teams["market_rating"] = (teams["win_total"] - intercept) / slope

players, history, age_table, n_age_pairs = load_players()
rookie_bpm, rookie_mpg, n_rookies = rookie_curve(players)
print("Building 2026-27 rosters:")
roster_now = build_rosters(players, apply_moves=True)
before = roster_strength(build_rosters(players, apply_moves=False), col="bpm_now")
after = roster_strength(roster_now)
no_rookies = roster_strength(build_rosters(players, apply_moves=True, add_rookies=False))
teams["roster_raw_2026"] = teams["team"].map(before)
teams["roster_raw_2027"] = teams["team"].map(after)

# Calibrate: how well does the roster sum explain last season's actual margins?
cal_slope, cal_int = np.polyfit(teams["roster_raw_2026"], teams["last_mov"], 1)
cal_r = np.corrcoef(teams["roster_raw_2026"], teams["last_mov"])[0, 1]
to_points = lambda raw: cal_int + cal_slope * raw
teams["roster_change"] = to_points(teams["roster_raw_2027"]) - to_points(teams["roster_raw_2026"])
teams["rookie_effect"] = cal_slope * (teams["roster_raw_2027"] - teams["team"].map(no_rookies))
teams["roster_rating"] = REGRESS * to_points(teams["roster_raw_2027"])

teams["rating"] = MARKET_WEIGHT * teams["market_rating"] + (1 - MARKET_WEIGHT) * teams["roster_rating"]
teams["vegas_vs_us"] = teams["market_rating"] - teams["roster_rating"]


# ---- Chunk 5: injuries for each team's first game ---------------------------
# Re-run the 240-minute allocation without the missing players, so their
# minutes go to the next men up in the actual rotation. A deep-bench player
# who wouldn't have played costs ~nothing. Questionable = QUESTIONABLE_WEIGHT
# chance he sits.
inj = pd.read_csv(DATA / "injuries.csv")
inj["key"] = inj["player"].map(key)
inj_cost = {}
for abbr, grp in inj.groupby("team"):
    team_roster = roster_now[roster_now["team"] == abbr]
    on_team = grp["key"].isin(team_roster["key"])
    for p in grp.loc[~on_team, "player"]:
        print(f"  ! injury list: {p} isn't in {abbr}'s projected rotation - ignored")
    grp = grp[on_team]
    outs = set(grp.loc[grp["status"] == "out", "key"])
    maybes = set(grp.loc[grp["status"] != "out", "key"])
    full = roster_strength(team_roster).iloc[0]
    no_outs = roster_strength(team_roster[~team_roster["key"].isin(outs)]).iloc[0]
    no_any = roster_strength(team_roster[~team_roster["key"].isin(outs | maybes)]).iloc[0]
    expected = (1 - QUESTIONABLE_WEIGHT) * no_outs + QUESTIONABLE_WEIGHT * no_any
    inj_cost[ABBR[abbr]] = float(REGRESS * cal_slope * (full - expected))  # raw sum -> rating points


# ---- Chunk 6: new head coaches ----------------------------------------------
NEW_COACH = {ABBR[t] for t in pd.read_csv(DATA / "coaches.csv")["team"]}


# ---- Chunk 6b: head-to-head matchups ----------------------------------------
# 1) For each past meeting: actual margin minus what the teams' full-season
#    margins + home court predicted. That leftover is the "matchup edge" + noise.
# 2) Average the leftovers; shrink hard, because a few games are mostly noise:
#       keep = MATCHUP_SD^2 / (MATCHUP_SD^2 + MARGIN_SD^2 / games)
# 3) Multiply by both teams' roster continuity: a matchup edge belongs to the
#    players who created it, so it fades as either roster turns over.
def roster_continuity() -> pd.Series:
    """Share of each team's 2025-26 minutes played by players still on that team."""
    raw = pd.read_csv(DATA / "advanced_2025_26.csv")
    raw = raw[(raw["Player"] != "League Average") & raw["Team"].isin(ABBR.keys())]
    raw["key"] = raw["Player"].map(key)
    current = roster_now.drop_duplicates("key").set_index("key")["team"]
    raw["stayed"] = raw.apply(lambda r: current.get(r["key"]) == r["Team"], axis=1)
    kept = raw[raw["stayed"]].groupby("Team")["MP"].sum()
    return (kept / raw.groupby("Team")["MP"].sum()).fillna(0)


CONTINUITY = roster_continuity()
LAST_MOV = teams.set_index("team")["last_mov"]


def h2h_table() -> pd.DataFrame:
    games_h2h = pd.read_csv(DATA / "h2h_games.csv")
    rows = []
    for g in games_h2h.itertuples():
        a, b = sorted([g.home, g.away])  # store every pair as (A, B) alphabetically
        hca = 0.0 if g.neutral else HOME_COURT
        expected_home = LAST_MOV[ABBR[g.home]] - LAST_MOV[ABBR[g.away]] + hca
        leftover_home = (g.home_pts - g.away_pts) - expected_home
        rows.append({"a": a, "b": b, "type": g.type,
                     "w": PLAYOFF_GAME_WEIGHT if g.type == "playoff" else 1.0,
                     "leftover_a": leftover_home if g.home == a else -leftover_home})
    df = pd.DataFrame(rows)
    out = []
    for (a, b), grp in df.groupby(["a", "b"]):
        n = grp["w"].sum()
        avg = np.average(grp["leftover_a"], weights=grp["w"])
        keep = MATCHUP_SD ** 2 / (MATCHUP_SD ** 2 + MARGIN_SD ** 2 / n) if MATCHUP_SD > 0 else 0.0
        cont = CONTINUITY[a] * CONTINUITY[b]
        out.append({"a": a, "b": b, "games": len(grp), "weighted_games": n, "avg_leftover_a": avg,
                    "keep": keep, "continuity": cont, "edge_a": avg * keep * cont})
    return pd.DataFrame(out)


H2H = h2h_table()
FULL_TO_ABBR = {v: k for k, v in ABBR.items()}


def h2h_edge(home: str, away: str) -> float:
    """Points added to the home team's margin from past meetings (0 if none)."""
    h, a = FULL_TO_ABBR[home], FULL_TO_ABBR[away]
    x, y = sorted([h, a])
    hit = H2H[(H2H["a"] == x) & (H2H["b"] == y)]
    if hit.empty:
        return 0.0
    edge_x = float(hit["edge_a"].iloc[0])
    return edge_x if h == x else -edge_x


# ---- Chunk 7: predict -------------------------------------------------------
def win_prob(margin):
    return 0.5 * (1 + math.erf(margin / (MARGIN_SD * math.sqrt(2))))

games = pd.read_csv(DATA / "first_games.csv")
r = teams.set_index("team")


def predict(coach_penalty: float) -> pd.DataFrame:
    rows = []
    for g in games.itertuples():
        out = {"date": g.date, "away": g.away, "home": g.home}
        inj_edge = inj_cost.get(g.away, 0) - inj_cost.get(g.home, 0)
        # A penalty on the away team helps the home side, and vice versa.
        coach_edge = coach_penalty * ((g.away in NEW_COACH) - (g.home in NEW_COACH))
        matchup_edge = h2h_edge(g.home, g.away)
        for col, label in [("stats_rating", "stats"), ("market_rating", "market"),
                           ("roster_rating", "roster"), ("rating", "blend")]:
            m = r.at[g.home, col] - r.at[g.away, col] + HOME_COURT + inj_edge + coach_edge + matchup_edge
            out[f"{label}_p"] = win_prob(m)
        out["margin"] = m
        out["injury_edge"] = inj_edge
        out["coach_edge"] = coach_edge
        out["h2h_edge"] = matchup_edge
        out["new_coach"] = ", ".join(t.split()[-1] for t in (g.away, g.home) if t in NEW_COACH)
        home_pick = out["margin"] > 0
        out["pick"] = g.home if home_pick else g.away
        out["pick_p"] = out["blend_p"] if home_pick else 1 - out["blend_p"]
        out["roster_vs_market"] = (out["roster_p"] > 0.5) != (out["market_p"] > 0.5)
        rows.append(out)
    return pd.DataFrame(rows)


# ---- Chunk 8: report --------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coach-penalty", type=float, default=NEW_COACH_PENALTY)
    penalty = ap.parse_args().coach_penalty
    preds = predict(penalty)

    pd.set_option("display.width", 220)
    print(f"\nCalibration: last-season margin = {cal_int:.1f} + {cal_slope:.2f} x roster sum "
          f"(correlation {cal_r:.2f})\n")

    print(f"Age curve: average BPM change to next season ({n_age_pairs} player-season pairs)")
    for interval, row in age_table.iterrows():
        print(f"  age {interval.left + 1}-{interval.right}: {row['shrunk']:+.2f}  (raw {row['mean']:+.2f}, n={int(row['count'])})")
    print()

    print("Key players: last season only -> all seasons -> 2026-27 projection (BPM)")
    for name in ["Jayson Tatum", "Tyrese Haliburton", "LeBron James", "Giannis Antetokounmpo",
                 "Joel Embiid", "Trae Young", "Tre Johnson", "Cooper Flagg"]:
        h = history.loc[key(name)]
        one = "  -- " if pd.isna(h["bpm_1yr"]) else f"{h['bpm_1yr']:+5.1f}"
        print(f"  {name:<22} age {int(h['age'])}: {one} -> {h['bpm_now']:+5.1f} -> {h['bpm']:+5.1f}")
    print()

    print(f"Rookie curve (fit on {n_rookies} players from the 2025 draft class):")
    for pk in [1, 2, 3, 5, 10, 20, 30]:
        print(f"  pick {pk:>2}: BPM {rookie_bpm(pk):+.1f}, {rookie_mpg(pk):.0f} min/game")
    print()

    print("Ratings, sorted by blend")
    cols = ["team", "stats_rating", "rookie_effect", "roster_change", "roster_rating", "market_rating", "vegas_vs_us", "rating"]
    print(teams.sort_values("rating", ascending=False)[cols].round(1).to_string(index=False), "\n")

    print("Head-to-head matchups (edge = points for team a)")
    print(H2H.round(2).to_string(index=False))
    print("Roster continuity (share of last season's minutes still on the team):",
          {t: round(float(CONTINUITY[t]), 2) for t in ["OKC", "SAS", "MIA", "MIN", "PHI", "MIL"]}, "\n")

    print("Injury cost for first game (points):", {k: round(v, 1) for k, v in inj_cost.items()}, "\n")

    pct = lambda s: (s * 100).round(0).astype(int)
    view = pd.DataFrame({
        "game": preds["away"] + " @ " + preds["home"],
        "stats%": pct(preds["stats_p"]), "roster%": pct(preds["roster_p"]),
        "market%": pct(preds["market_p"]), "blend%": pct(preds["blend_p"]),
        "inj": preds["injury_edge"].round(1),
        "h2h": preds["h2h_edge"].round(1),
        "new_coach": preds["new_coach"],
        "pick": preds["pick"], "conf%": pct(preds["pick_p"]),
        "flag": preds["roster_vs_market"].map({True: "<-- roster vs Vegas", False: ""}),
    })
    print(f"First-game predictions (% = home win probability), new-coach penalty = {penalty} pts")
    print(view.to_string(index=False))

    # Sensitivity: which picks would change if the new-coach effect is real?
    print("\nNew-coach sensitivity (picks that change vs. no penalty)")
    base = predict(0.0)
    for test in [0.5, 1.0, 1.5, 2.0, 3.0]:
        alt = predict(test)
        flips = [f"{a} @ {h}: {b} -> {n}" for a, h, b, n in
                 zip(alt["away"], alt["home"], base["pick"], alt["pick"]) if b != n]
        print(f"  {test:>3} pts: " + ("; ".join(flips) if flips else "no picks change"))
    (HERE / "results").mkdir(exist_ok=True)
    preds.to_csv(HERE / "results" / "predictions.csv", index=False)
