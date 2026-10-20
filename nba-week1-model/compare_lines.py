"""
Compare the model's predictions with real betting lines (data/market_lines.csv).

Moneyline -> win probability: a -150 favorite must risk 150 to win 100, so the
implied chance is 150/250 = 60%. Both sides' implied chances add up to a bit
more than 100% (that's the bookmaker's cut, the "vig"), so we scale them to 100%.
Spread: a home spread of -5.5 means the market expects the home team by 5.5.

Run:  python3 compare_lines.py                  table in the terminal
      python3 compare_lines.py --update-readme  also refresh the table in README.md
Always writes results/predictions_vs_market.md.
"""
import argparse
import contextlib
import io
from datetime import date
from pathlib import Path

import pandas as pd

with contextlib.redirect_stdout(io.StringIO()):
    import model as m

HERE = Path(__file__).parent
README_START, README_END = "<!-- PREDICTIONS TABLE START -->", "<!-- PREDICTIONS TABLE END -->"


def implied(ml: float) -> float:
    return -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)


def short(team: str) -> str:
    return "Blazers" if team.startswith("Portland") else team.split()[-1]


# ---- Chunk 1: market probabilities and margins ------------------------------
lines = pd.read_csv(m.DATA / "market_lines.csv").rename(
    columns={"home": "home_abbr", "away": "away_abbr", "date": "line_date"})
lines["home_full"] = lines["home_abbr"].map(m.ABBR)
lines["away_full"] = lines["away_abbr"].map(m.ABBR)
raw_home, raw_away = lines["ml_home"].map(implied), lines["ml_away"].map(implied)
lines["market_home_p"] = raw_home / (raw_home + raw_away)       # vig removed
lines["market_margin"] = -lines["spread_home"]

# ---- Chunk 2: line them up with the model -----------------------------------
preds = m.predict(m.NEW_COACH_PENALTY)
df = preds.merge(lines, left_on=["home", "away"], right_on=["home_full", "away_full"], how="inner")
df["gap"] = df["margin"] - df["market_margin"]                   # + = we like the home team more
df["market_pick"] = df.apply(lambda r: r["home"] if r["market_home_p"] > 0.5 else r["away"], axis=1)
df["market_pick_p"] = df.apply(lambda r: r["market_home_p"] if r["market_pick"] == r["home"]
                               else 1 - r["market_home_p"], axis=1)
df["agree"] = df["pick"] == df["market_pick"]
# Gap from the market favorite's side: negative = we're less sure of the favorite.
fav_sign = df["market_margin"].apply(lambda x: 1 if x >= 0 else -1)
df["gap_vs_fav"] = df["gap"] * fav_sign


def line_text(margin_home: float, home: str, away: str) -> str:
    """+3.5 for the home team -> 'Knicks -3.5' (betting notation)."""
    if abs(margin_home) < 0.05:
        return "Pick'em"
    fav = home if margin_home > 0 else away
    return f"{short(fav)} −{abs(margin_home):.1f}"


def summary() -> dict:
    fav = df["market_margin"].abs() >= 4
    return {
        "agree": int(df["agree"].sum()), "games": len(df),
        "avg_abs_gap": df["gap"].abs().mean(),
        "fav_market": df.loc[fav, "market_margin"].abs().mean(),
        "fav_ours": (df.loc[fav, "margin"] * fav_sign[fav]).mean(),
    }


# ---- Chunk 3: the markdown table (README / results) --------------------------
def markdown_table() -> str:
    rows = ["| Date | Matchup | Model pick | Model win % | Market pick | Market win % "
            "| Model line | Market line | Diff (pts) |",
            "|---|---|---|---:|---|---:|---|---|---:|"]
    for r in df.itertuples():
        flag = "" if r.agree else " ⚠️"
        rows.append(
            f"| {pd.Timestamp(r.date):%b %d} | {short(r.away)} @ {short(r.home)} "
            f"| **{short(r.pick)}**{flag} | {r.pick_p * 100:.0f}% "
            f"| {short(r.market_pick)} | {r.market_pick_p * 100:.0f}% "
            f"| {line_text(r.margin, r.home, r.away)} | {line_text(r.market_margin, r.home, r.away)} "
            f"| {r.gap_vs_fav:+.1f} |".replace("-", "−"))
    s = summary()
    notes = [
        "",
        f"*Model and market pick the same winner in **{s['agree']} of {s['games']}** games. "
        f"Average difference: **{s['avg_abs_gap']:.1f} points**. "
        f"When the market favors a team by 4+, it expects {s['fav_market']:.1f} points on average; "
        f"the model expects {s['fav_ours']:.1f}.*",
        "",
        "*Diff = model margin minus market spread, from the market favorite's side "
        "(negative = the model is less confident in the favorite). ⚠️ = different pick. "
        "Market win % is the moneyline with the bookmaker's margin removed. "
        f"Lines: {lines['source'].iloc[0].split(' ')[0]}, captured before tip-off. "
        f"Generated {date.today():%b %d, %Y}.*",
    ]
    return "\n".join(rows + notes)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args()

    out = pd.DataFrame({
        "date": df["date"].str[5:],
        "game": df["away"].map(short) + " @ " + df["home"].map(short),
        "our pick": df["pick"].map(short),
        "our %": (df["pick_p"] * 100).round(0).astype(int),
        "mkt pick": df["market_pick"].map(short),
        "mkt %": (df["market_pick_p"] * 100).round(0).astype(int),
        "our margin": df["margin"].round(1),
        "mkt margin": df["market_margin"],
        "gap": df["gap"].round(1),
        "": df["agree"].map({True: "", False: "<-- different pick"}),
    })
    print("Margins and gap are from the HOME team's view (gap + = we like home more than the market)\n")
    print(out.to_string(index=False))
    s = summary()
    print(f"\nSame pick in {s['agree']} of {s['games']} games.  Average size of gap {s['avg_abs_gap']:.1f} pts.")
    print(f"Market favorites by 4+: market avg {s['fav_market']:.1f} pts, ours {s['fav_ours']:.1f} pts")

    table = markdown_table()
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "predictions_vs_market.md").write_text(
        "# Model predictions vs. the betting market\n\n" + table + "\n")
    print("\nWrote results/predictions_vs_market.md")

    if args.update_readme:
        readme = HERE / "README.md"
        text = readme.read_text()
        before, rest = text.split(README_START, 1)
        _, after = rest.split(README_END, 1)
        readme.write_text(before + README_START + "\n" + table + "\n" + README_END + after)
        print("Updated the table in README.md")
