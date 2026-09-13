# Derived tables

Every file here is produced by the scripts in `analysis/` from the competition's official replay
episodes. Nothing here is hand-edited.

| File | Produced by | Notes |
|---|---|---|
| `frozen_tournament_teams.csv` | `mine_tournament.py` | 259 entrants with ≥30 games: rank, games, final rating, win rate, main archetype, tier. Pseudonymous `entrant_id` (`E001`… by rank); leaderboard names and numeric team ids are removed. |
| `elo_signal_screen_all_tests.csv` | `screen_elo_signal.py` | All 696 habit×archetype tests. 72 survive FDR 10 %. |
| `multiturn_screen_all_tests.csv` | `mine_multiturn.py` + the same screen | Turn-pair sequences: a development turn and what followed it. |
| `matchup_residual_pp.csv` | `mine_tournament.py` | Row's win % minus rating-expected win %, against column. Percentage points. |
| `matchup_n_games.csv` | `mine_tournament.py` | Game counts backing each cell above. Read the two together; no cell in the writeup's cycle rests on fewer than 433 games. |
| `budew_dragapult_by_tier.csv` | `mine_tournament.py` | One archetype, four rating tiers: pooled take-rates. Pooled, so read it with the caveat the writeup attaches to pooling. |
| `shipped_agent_flag_enumeration.txt` | `ast.parse` over `agent/` | All 89 switches with file, line and value. |

## Read these before using the tables

**The screen's q-values assume the whole family.** `elo_signal_screen_all_tests.csv` carries every
test so you can see the multiplicity. If you re-screen a subset, recompute the FDR — do not filter
this file and reuse its `q` column.

**Check `rho_excl_leader` before believing any row.** It repeats the correlation with the
highest-rated entrant in that archetype removed. A survivor that collapses under that is one
entrant's policy, not a property of strong play. The writeup keeps one such case as a worked example.

**The corpus over-weights the upper ladder.** The competition publishes top replays per day, so this
is not a uniform sample of all games played.

**No entrant is identifiable here.** `frozen_tournament_teams.csv` was published with leaderboard
display names and numeric team ids; both columns have been removed and replaced with a
rank-ordered pseudonym (`entrant_id`). Even though the source leaderboard is public, republishing a
ranked table of named individuals alongside behavioural analysis is a different act from analysing
it, so the identifiers are gone. The writeup refers to entrants only by rank or archetype.

**Consequence for reproducing.** `screen_elo_signal.py` joins `team_stats.json` to the entrant table
on `team_id`, which this published copy no longer has. Regenerate the table locally with
`mine_tournament.py` — your own copy will carry the real ids and the join will work. The published
copy is for reading the results, not for re-running the join.

## Known limitation

`elo_signal_screen_all_tests.csv` was generated before a latent bug in the screen was found and
fixed: an exact `std == 0` guard let a logically-constant feature through at std ≈ 1e-16, whose NaN
p-value would propagate through Benjamini-Hochberg and blank the entire `q` column. The shipped
table was checked afterwards and is **unaffected** — 0 NaNs in 696 rows, q spanning 0.0003 to 0.993 —
because no feature was constant across entrants on the full fortnight. `screen_elo_signal.py` now
guards with a tolerance and refuses to emit a non-finite p-value, so a rerun on a narrower window
behaves.
