#!/usr/bin/env python3
"""Entrant-level habit screen: does a habit track final rating, holding the archetype fixed?

This is the test behind the headline claim. It is deliberately NOT a pooled tier contrast:
the unit of analysis is one (entrant, archetype) pair, so the top-ranked entrant -- who played
2,049 games -- contributes exactly one observation per feature instead of dominating a tier.

Inputs
    team_stats.json   from mine_team_stats.py  (per-entrant decision counters, keyed by team id)
    teams.csv         from mine_tournament.py  (per-entrant final rating, name, archetype)

Output
    elo_signal_screen_all_tests.csv -- EVERY test, not only the survivors. Columns:
        arch, feature, n_teams, rho, p, pts_per_sd, rho_excl_leader, mean, q

    rho              Spearman correlation of the entrant's habit rate with its final rating
    pts_per_sd       OLS slope: rating points per standard deviation of the habit
    rho_excl_leader  same rho with the highest-rated entrant in that archetype dropped
    q                Benjamini-Hochberg FDR across all tests in the family

Read it with a threshold of q < 0.10, and check rho_excl_leader before believing any row:
a survivor that collapses when one entrant is removed is that entrant's policy, not a finding.

Usage:  python screen_elo_signal.py [--team-stats PATH] [--teams PATH] [--out PATH]
Requires pandas, scipy, statsmodels (unlike the miners, which are standard library only).
"""
import argparse
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

MIN_GAMES = 30      # an entrant needs this many games in the archetype to be admitted
MIN_TEAMS = 20      # an archetype needs this many admitted entrants to be tested
MIN_OFFERS = 50     # a decision type must have been offered this often to score its take-rate
MIN_MENUS = 100     # a menu signature must appear this often to score its pick shares
MIN_EXITS = 100     # a directly-follows source must have this many exits to score its edges
MIN_ATTACKS = 50
MIN_SWINGS = 15     # two-prize-behind / two-prize-ahead episodes
MIN_FIRST_PRIZE = 20

# activity letters: P play, A attach, B ability, E evolve, R retreat, K attack, N end-of-turn
ACTIVITIES = "PABERKN"


def features(counters):
    """Turn one entrant's raw counters for one archetype into habit RATES.

    Rates, not counts: a count would be confounded by how many games the entrant played.
    Returns None when the entrant is below the games threshold.
    """
    f = {}
    games = counters.get("games", 0)
    turns = counters.get("turns", 0)
    if games < MIN_GAMES or turns == 0:
        return None

    # how often each activity was chosen when it was legal
    for letter in ACTIVITIES:
        offered = counters.get(f"off|{letter}", 0)
        if offered >= MIN_OFFERS:
            f[f"takes {letter} when offered"] = counters.get(f"take|{letter}", 0) / offered

    # within one menu signature, which option was picked
    for key, n in counters.items():
        if key.startswith("menu|") and n >= MIN_MENUS:
            sig = key[5:]
            for letter in sig:
                f[f"menu {sig} -> {letter}"] = counters.get(f"pick|{sig}|{letter}", 0) / n

    # directly-follows edges, as a share of the source activity's exits
    exits = Counter()
    for key, n in counters.items():
        if key.startswith("dfg|"):
            exits[key[4]] += n
    for key, n in counters.items():
        if key.startswith("dfg|") and exits[key[4]] >= MIN_EXITS:
            f[f"after {key[4]} -> {key[5]}"] = n / exits[key[4]]

    f["actions per turn"] = counters.get("actions", 0) / turns
    f["pass turns share"] = counters.get("pass_turns", 0) / turns
    f["turns per game"] = turns / games
    if counters.get("attacks", 0) >= MIN_ATTACKS:
        f["zero-damage attack share"] = counters.get("zero_dmg_attacks", 0) / counters["attacks"]
    if counters.get("behind2", 0) >= MIN_SWINGS:
        f["comeback from -2"] = counters.get("behind2_won", 0) / counters["behind2"]
    if counters.get("ahead2", 0) >= MIN_SWINGS:
        f["blown +2 lead"] = counters.get("ahead2_lost", 0) / counters["ahead2"]
    if counters.get("first_prize_games", 0) >= MIN_FIRST_PRIZE:
        f["takes first prize"] = counters.get("first_prize_mine", 0) / counters["first_prize_games"]
    f["win rate"] = counters.get("wins", 0) / games
    return f


def build_long_table(team_stats, rating):
    """One row per (entrant, archetype, feature)."""
    rows = []
    for tid, per_arch in team_stats.items():
        for arch, counters in per_arch.items():
            f = features(counters)
            if f is None:
                continue
            team = int(tid)
            if team not in rating or pd.isna(rating[team]):
                continue
            for name, value in f.items():
                rows.append((team, arch, name, value, counters["games"]))
    long = pd.DataFrame(rows, columns=["team", "arch", "feature", "value", "games"])
    long["rating"] = long.team.map(rating)
    return long


def _is_constant(values):
    """Logically-constant within floating-point noise.

    An exact `std == 0` test is NOT enough. A rate like 200/700 is recomputed per entrant,
    so a feature that is constant in substance lands at std ~ 1e-16 rather than 0, sails
    through the guard, and makes Spearman return NaN. One NaN p-value then propagates
    through multipletests and silently wipes out EVERY q in the table.
    """
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return True
    scale = max(1.0, float(np.abs(v).max()))
    return bool(np.ptp(v) <= 1e-12 * scale)


def screen(long):
    res = []
    for (arch, feature), g in long.groupby(["arch", "feature"]):
        if g.team.nunique() < MIN_TEAMS or _is_constant(g.value):
            continue
        r = stats.spearmanr(g.value, g.rating)
        if not np.isfinite(r.pvalue):            # degenerate despite the guard: skip, never emit NaN
            continue
        # rating points per SD of the habit; linregress avoids polyfit's conditioning warning
        z = (g.value - g.value.mean()) / g.value.std()
        pts_per_sd = stats.linregress(z, g.rating).slope

        g2 = g[g.rating < g.rating.max()]        # drop the top-rated entrant
        if g2.team.nunique() >= 3 and not _is_constant(g2.value):
            rho_excl = stats.spearmanr(g2.value, g2.rating).statistic
        else:
            rho_excl = np.nan
        res.append((arch, feature, len(g), r.statistic, r.pvalue,
                    pts_per_sd, rho_excl, g.value.mean()))

    out = pd.DataFrame(res, columns=["arch", "feature", "n_teams", "rho", "p",
                                     "pts_per_sd", "rho_excl_leader", "mean"])
    if out.empty:
        return out.assign(q=[])
    assert np.isfinite(out.p).all(), "non-finite p-value would wipe the whole q column"
    out["q"] = multipletests(out.p, method="fdr_bh")[1]
    return out.sort_values("q")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team-stats", default="team_stats.json")
    ap.add_argument("--teams", default="teams.csv")
    ap.add_argument("--out", default="elo_signal_screen_all_tests.csv")
    a = ap.parse_args()

    team_stats = json.load(open(a.team_stats))
    teams = pd.read_csv(a.teams)
    rating = teams.set_index("team_id").final_rating.to_dict()

    long = build_long_table(team_stats, rating)
    print("admitted entrants per archetype:",
          long.groupby("arch").team.nunique().sort_values(ascending=False).head(8).to_dict())

    out = screen(long)
    out.to_csv(a.out, index=False)
    print(f"{len(out)} tests written to {a.out}; {(out.q < 0.10).sum()} survive FDR 10%")


if __name__ == "__main__":
    main()
