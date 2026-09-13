#!/usr/bin/env python3
"""Process-mine the post-freeze tournament (frozen agents, 2026-08-17..2026-08-30).

Reads ~/ptcg/episodes.db (episodes, game_features, events blobs) and ~/ptcg/epmeta.sqlite
(agents with pre/post rating + reward, canon_arch labels, team_names). Pure stdlib.

Outputs (cwd):
  games.csv        one row per game (ratings, archetypes, outcome, game_features)
  teams.csv        one row per team active post-freeze (final rating, games, main archetype)
  decisions.json   decision-mining aggregates by tier: option uptake, per-turn variants, DFG
  tempo.json       prizes-by-turn, win vs loss, by tier
  run_meta.json    thresholds, counts, timings

NOTE ON IDENTIFIERS: the teams.csv this writes carries the leaderboard display `name` and the
numeric `team_id` for every entrant, because the downstream join needs them. That is fine
locally. Strip both columns before publishing the table anywhere -- the copy in this
repository's data/ directory is pseudonymised for exactly that reason.
"""
import sqlite3, zlib, json, csv, time, os, statistics
from collections import Counter, defaultdict

HOME = os.path.expanduser(os.environ.get("PTCG_HOME", "~/ptcg"))
D0 = os.environ.get("PTCG_D0", "2026-08-17")
D1 = os.environ.get("PTCG_D1", "2026-08-30")
LIMIT = int(os.environ.get("LIMIT", "0"))  # 0 = all games
T0 = time.time()

e = sqlite3.connect(f"file:{HOME}/episodes.db?mode=ro", uri=True)
m = sqlite3.connect(f"file:{HOME}/epmeta.sqlite?mode=ro", uri=True)

# ---------------------------------------------------------------- 1. games table
q = """select ep.id, ep.date, ep.reward0, ep.reward1, ep.n_steps, ep.deck_md5_0, ep.deck_md5_1,
       gf.prizes_taken_0, gf.prizes_taken_1, gf.first_attack_turn_0, gf.first_attack_turn_1,
       gf.mulligans_0, gf.mulligans_1, gf.never_benched_0, gf.never_benched_1,
       gf.time_used_0, gf.time_used_1, gf.n_attacks_0, gf.n_attacks_1, gf.n_attach_0, gf.n_attach_1,
       gf.n_evolve_0, gf.n_evolve_1, gf.dmg_dealt_0, gf.dmg_dealt_1, gf.max_turn, gf.min_deck_0, gf.min_deck_1
       from episodes ep join game_features gf on gf.episode_id=ep.id
       where ep.date between ? and ?"""
if LIMIT: q += f" limit {LIMIT}"
rows = e.execute(q, (D0, D1)).fetchall()
gcols = ["episode_id","date","reward0","reward1","n_steps","deck_md5_0","deck_md5_1",
         "prizes_0","prizes_1","first_attack_turn_0","first_attack_turn_1","mulligans_0","mulligans_1",
         "never_benched_0","never_benched_1","time_used_0","time_used_1","n_attacks_0","n_attacks_1",
         "n_attach_0","n_attach_1","n_evolve_0","n_evolve_1","dmg_0","dmg_1","max_turn","min_deck_0","min_deck_1"]
games = {r[0]: dict(zip(gcols, r)) for r in rows}
ids = list(games)
print("games", len(ids), round(time.time()-T0,1), flush=True)

# agents + archetypes (chunked IN queries)
def chunks(xs, n=900):
    for i in range(0, len(xs), n): yield xs[i:i+n]
ag = {}
arch = {}
for ch in chunks(ids):
    ph = ",".join("?"*len(ch))
    for eid, idx, sub, team, pre, post, rew in m.execute(
        f"select episode_id, idx, submission_id, team_id, initial_score, updated_score, reward from agents where episode_id in ({ph})", ch):
        ag[(eid, idx)] = (sub, team, pre, post, rew)
    for eid, a0, a1 in m.execute(f"select episode_id, arch0, arch1 from canon_arch where episode_id in ({ph})", ch):
        arch[eid] = (a0, a1)
    for eid, ct in m.execute(f"select episode_id, create_time from episodes_meta where episode_id in ({ph})", ch):
        games[eid]["create_time"] = ct
for eid, g in games.items():
    for s in (0, 1):
        sub, team, pre, post, rew = ag.get((eid, s), (None,)*5)
        g[f"sub_{s}"], g[f"team_{s}"], g[f"elo_pre_{s}"], g[f"elo_post_{s}"], g[f"agent_reward_{s}"] = sub, team, pre, post, rew
    g["arch_0"], g["arch_1"] = arch.get(eid, (None, None))
print("joined", round(time.time()-T0,1), flush=True)

# ---------------------------------------------------------------- 2. teams + tiers
team_games = defaultdict(list)   # team -> [(create_time, post_score, arch, win)]
for eid, g in games.items():
    for s in (0, 1):
        t = g[f"team_{s}"]
        if t is None: continue
        win = 1 if (g[f"agent_reward_{s}"] or 0) > 0 else 0
        team_games[t].append((g.get("create_time") or g["date"], g[f"elo_post_{s}"], g[f"arch_{s}"], win, g[f"sub_{s}"]))
names = {}
for ch in chunks(list(team_games)):
    ph = ",".join("?"*len(ch))
    for t, n in m.execute(f"select team_id, name from team_names where team_id in ({ph}) order by last_seen", ch):
        names[t] = n
teams = {}
for t, lst in team_games.items():
    lst.sort(key=lambda x: x[0])
    finals = [x[1] for x in lst if x[1] is not None]
    archs = Counter(x[2] for x in lst if x[2])
    main_arch, main_n = (archs.most_common(1)[0] if archs else (None, 0))
    subs = Counter(x[4] for x in lst)
    teams[t] = {"team_id": t, "name": names.get(t), "n_games": len(lst),
                "final_rating": finals[-1] if finals else None,
                "mean_rating_last10": statistics.mean(finals[-10:]) if finals else None,
                "win_rate": round(sum(x[3] for x in lst)/len(lst), 4),
                "main_arch": main_arch, "main_arch_share": round(main_n/len(lst), 3),
                "n_archs": len(archs), "n_subs": len(subs)}
qual = sorted([v["final_rating"] for v in teams.values() if v["n_games"] >= 30 and v["final_rating"] is not None])
def quant(p): return qual[min(len(qual)-1, int(p*len(qual)))]
thr = {"T1_top5pct": quant(0.95), "T2_top25pct": quant(0.75), "T3_top60pct": quant(0.40)}
def tier_of(r):
    if r is None: return None
    if r >= thr["T1_top5pct"]: return "T1"
    if r >= thr["T2_top25pct"]: return "T2"
    if r >= thr["T3_top60pct"]: return "T3"
    return "T4"
for v in teams.values(): v["tier"] = tier_of(v["final_rating"])
# rank
ranked = sorted([v for v in teams.values() if v["final_rating"] is not None], key=lambda v: -v["final_rating"])
for i, v in enumerate(ranked, 1): v["rank"] = i
print("teams", len(teams), "qualified", len(qual), thr, round(time.time()-T0,1), flush=True)

with open("teams.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(next(iter(teams.values())).keys())); w.writeheader()
    for v in ranked + [v for v in teams.values() if v["final_rating"] is None]: w.writerow(v)
for g in games.values():
    for s in (0, 1):
        t = g[f"team_{s}"]; g[f"tier_{s}"] = teams[t]["tier"] if t in teams else None
        g[f"final_rating_{s}"] = teams[t]["final_rating"] if t in teams else None
gfields = list(next(iter(games.values())).keys())
with open("games.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=gfields, extrasaction="ignore"); w.writeheader()
    for g in games.values(): w.writerow(g)
print("csv written", round(time.time()-T0,1), flush=True)

# ---------------------------------------------------------------- 3. decision mining from blobs
# OptionType codes -> activity letters. A player-turn ends with an ATTACK (K) or END (N).
ACT = {7: "P", 8: "A", 9: "E", 10: "B", 12: "R", 13: "K", 14: "N"}
uptake = defaultdict(Counter)      # (tier, arch_or_ALL, offered_sig) -> Counter(chosen letter)
variants = defaultdict(Counter)    # (tier, turn_bucket) -> Counter(variant string)
dfg = defaultdict(Counter)         # tier -> Counter((a,b))
perturn = defaultdict(lambda: defaultdict(list))  # tier -> metric -> list (one entry per player-turn)
turn_end = defaultdict(Counter)    # tier -> Counter of how turns end / what was left on the table
tempo = defaultdict(lambda: defaultdict(list))    # (tier, win) -> player-turn -> prizes taken so far
timeuse = defaultdict(list); decision_ct = defaultdict(list)
zero_atk = defaultdict(Counter)    # tier -> {'attacks','zero_damage'}
n_blob = 0; n_bad = 0
top_archs = [a for a, _ in Counter([g["arch_0"] for g in games.values()] + [g["arch_1"] for g in games.values()]).most_common(8)]

for ch in chunks(ids, 500):
    ph = ",".join("?"*len(ch))
    for eid, blob in e.execute(f"select episode_id, blob from events where episode_id in ({ph})", ch):
        g = games[eid]
        try:
            d = json.loads(zlib.decompress(blob))
        except Exception:
            n_bad += 1; continue
        n_blob += 1
        # ---- zero-damage attacks from the log stream (ATTACK=15, HP_CHANGE=16 at the same step)
        logs = d.get("logs0") or d.get("logs1") or []
        dmg_at = defaultdict(int)   # (step, victim) -> damage
        atks = []
        for st, ev in logs:
            if not isinstance(ev, dict): continue
            t = ev.get("type")
            if t == 15: atks.append((st, ev.get("playerIndex")))
            elif t == 16 and isinstance(ev.get("value"), int) and ev["value"] < 0:
                dmg_at[(st, ev.get("playerIndex"))] += -ev["value"]
        for st, p in atks:
            tier = g[f"tier_{p}"]
            if tier is None: continue
            zero_atk[tier]["attacks"] += 1
            if dmg_at.get((st, 1-p), 0) == 0: zero_atk[tier]["zero_damage"] += 1
        # ---- decisions -> per-seat activity stream, segmented into player-turns at K/N
        streams = {0: [], 1: []}   # seat -> [(step, chosen, offered_letters)]
        for dec in d.get("decisions") or []:
            try: step, seat, sel, act = dec
            except Exception: continue
            if not isinstance(sel, dict) or not act: continue
            opts = sel.get("option") or []
            try: picked = opts[act[0]]
            except Exception: continue
            ptype = picked.get("type") if isinstance(picked, dict) else None
            letters = sorted({ACT[o.get("type")] for o in opts if isinstance(o, dict) and o.get("type") in ACT})
            if not letters: continue   # sub-menus (card / yes-no picks)
            streams[seat].append((step, ACT.get(ptype, "?"), "".join(letters)))
        for seat in (0, 1):
            tier = g[f"tier_{seat}"]
            if tier is None: continue
            a = g[f"arch_{seat}"]
            win = 1 if (g[f"agent_reward_{seat}"] or 0) > 0 else 0
            decision_ct[tier].append(len(streams[seat]))
            timeuse[tier].append((d.get("over") or [None, None])[seat])
            # segment
            turns = []; cur = []
            for step, ch_, sig in streams[seat]:
                uptake[(tier, "ALL", sig)][ch_] += 1
                if a in top_archs: uptake[(tier, a, sig)][ch_] += 1
                cur.append((step, ch_, sig))
                if ch_ in ("K", "N"):
                    turns.append(cur); cur = []
            if cur: turns.append(cur)
            turn_end_steps = [t[-1][0] for t in turns]
            for ti, t in enumerate(turns, 1):
                letters = [x[1] for x in t]
                bucket = "t1-2" if ti <= 2 else "t3-5" if ti <= 5 else "t6+"
                variants[(tier, bucket)]["".join(letters)] += 1
                for x, y in zip(["START"]+letters, letters+["STOP"]): dfg[tier][(x, y)] += 1
                perturn[tier]["attach"].append(letters.count("A"))
                perturn[tier]["play"].append(letters.count("P"))
                perturn[tier]["evolve"].append(letters.count("E"))
                perturn[tier]["ability"].append(letters.count("B"))
                perturn[tier]["retreat"].append(letters.count("R"))
                perturn[tier]["attacked"].append(1 if letters[-1] == "K" else 0)
                perturn[tier]["actions"].append(len(letters))
                last_sig = t[-1][2]
                offered_any_attack = any("K" in x[2] for x in t)
                turn_end[tier]["turns"] += 1
                if letters[-1] == "N":
                    turn_end[tier]["ended_with_pass"] += 1
                    if "K" in last_sig: turn_end[tier]["pass_with_attack_available"] += 1
                    if "A" in last_sig: turn_end[tier]["pass_with_attach_available"] += 1
                    if "E" in last_sig: turn_end[tier]["pass_with_evolve_available"] += 1
                if offered_any_attack: turn_end[tier]["attack_offered_in_turn"] += 1
                if "A" not in letters and any("A" in x[2] for x in t): turn_end[tier]["turn_without_attach_though_offered"] += 1
            # tempo: prizes taken by end of each of my player-turns
            pr = d.get("prizes") or []
            taken_at = sorted((st, (p0, p1)[seat]) for st, p0, p1 in pr)
            for ti, es in enumerate(turn_end_steps[:25], 1):
                val = 0
                for st, v in taken_at:
                    if st <= es: val = v
                    else: break
                tempo[(tier, win)][ti].append(val)
    print("blobs", n_blob, "bad", n_bad, round(time.time()-T0,1), flush=True)

def summ(xs):
    xs = [x for x in xs if x is not None]
    if not xs: return None
    xs.sort()
    return {"n": len(xs), "mean": round(statistics.mean(xs), 4), "median": xs[len(xs)//2],
            "p90": xs[int(0.9*(len(xs)-1))]}

out = {
  "tiers": thr, "top_archs": top_archs, "n_games_blobs": n_blob,
  "uptake": {f"{t}|{a}|{sig}": dict(c) for (t, a, sig), c in uptake.items()},
  "variants": {f"{t}|{b}": dict(c.most_common(60)) for (t, b), c in variants.items()},
  "variant_totals": {f"{t}|{b}": sum(c.values()) for (t, b), c in variants.items()},
  "dfg": {t: {f"{x}>{y}": n for (x, y), n in c.items()} for t, c in dfg.items()},
  "perturn": {t: {k: summ(v) for k, v in mm.items()} for t, mm in perturn.items()},
  "turn_end": {t: dict(c) for t, c in turn_end.items()},
  "zero_attacks": {t: dict(c) for t, c in zero_atk.items()},
  "timeuse": {t: summ(v) for t, v in timeuse.items()},
  "decisions_per_game": {t: summ(v) for t, v in decision_ct.items()},
}
json.dump(out, open("decisions.json", "w"))
json.dump({f"{t}|{w}": {tn: {"n": len(v), "mean": round(statistics.mean(v), 3)} for tn, v in d_.items()}
           for (t, w), d_ in tempo.items()}, open("tempo.json", "w"))
json.dump({"date_range": [D0, D1], "n_games": len(ids), "n_teams": len(teams), "n_qualified_teams": len(qual),
           "tiers": thr, "n_blobs": n_blob, "n_bad": n_bad, "seconds": round(time.time()-T0, 1)},
          open("run_meta.json", "w"), indent=1)
print("done", round(time.time()-T0, 1))
