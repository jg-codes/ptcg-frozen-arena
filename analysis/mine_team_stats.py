#!/usr/bin/env python3
"""Team-level decision statistics for the Elo-signal screen (frozen tournament 2026-08-17..30).
For every (team, archetype) seat with >=1 game: counts of (menu signature -> chosen), directly-follows
edges, turn-variant strings (first 6 letters), comeback/blown-lead events, zero-damage attacks.
Output: team_stats.json  {team_id: {arch: {...counters...}}}
"""
import sqlite3, zlib, json, time, os
from collections import Counter, defaultdict
HOME = os.path.expanduser(os.environ.get("PTCG_HOME", "~/ptcg"))
D0 = os.environ.get("PTCG_D0", "2026-08-17")
D1 = os.environ.get("PTCG_D1", "2026-08-30"); T0 = time.time()
e = sqlite3.connect(f"file:{HOME}/episodes.db?mode=ro", uri=True)
m = sqlite3.connect(f"file:{HOME}/epmeta.sqlite?mode=ro", uri=True)
ids = [r[0] for r in e.execute("select id from episodes where date between ? and ?", (D0, D1))]
def chunks(xs, n=900):
    for i in range(0, len(xs), n): yield xs[i:i+n]
ag = {}; arch = {}
for ch in chunks(ids):
    ph = ",".join("?"*len(ch))
    for eid, idx, team, rew in m.execute(f"select episode_id, idx, team_id, reward from agents where episode_id in ({ph})", ch):
        ag[(eid, idx)] = (team, rew)
    for eid, a0, a1 in m.execute(f"select episode_id, arch0, arch1 from canon_arch where episode_id in ({ph})", ch):
        arch[eid] = (a0, a1)
ACT = {7: "P", 8: "A", 9: "E", 10: "B", 12: "R", 13: "K", 14: "N"}
S = defaultdict(lambda: defaultdict(Counter))   # team -> arch -> Counter
nb = 0
for ch in chunks(ids, 500):
    ph = ",".join("?"*len(ch))
    for eid, blob in e.execute(f"select episode_id, blob from events where episode_id in ({ph})", ch):
        try: d = json.loads(zlib.decompress(blob))
        except Exception: continue
        nb += 1
        logs = d.get("logs0") or d.get("logs1") or []
        dmg_at = defaultdict(int); atks = []
        for st, ev in logs:
            if not isinstance(ev, dict): continue
            t = ev.get("type")
            if t == 15: atks.append((st, ev.get("playerIndex")))
            elif t == 16 and isinstance(ev.get("value"), int) and ev["value"] < 0: dmg_at[(st, ev.get("playerIndex"))] += -ev["value"]
        streams = {0: [], 1: []}
        for dec in d.get("decisions") or []:
            try: step, seat, sel, act = dec
            except Exception: continue
            if not isinstance(sel, dict) or not act: continue
            opts = sel.get("option") or []
            try: picked = opts[act[0]]
            except Exception: continue
            ptype = picked.get("type") if isinstance(picked, dict) else None
            letters = sorted({ACT[o.get("type")] for o in opts if isinstance(o, dict) and o.get("type") in ACT})
            if not letters: continue
            streams[seat].append((step, ACT.get(ptype, "?"), "".join(letters)))
        pr = sorted(d.get("prizes") or [])
        for seat in (0, 1):
            team, rew = ag.get((eid, seat), (None, None))
            if team is None or rew is None: continue
            a = arch.get(eid, (None, None))[seat] or "?"
            c = S[team][a]; win = 1 if rew > 0 else 0
            c["games"] += 1; c["wins"] += win
            for st, p in atks:
                if p == seat:
                    c["attacks"] += 1
                    if dmg_at.get((st, 1-seat), 0) == 0: c["zero_dmg_attacks"] += 1
            turns = []; cur = []
            for step, ch_, sig in streams[seat]:
                c[f"menu|{sig}"] += 1; c[f"pick|{sig}|{ch_}"] += 1
                for L in sig: c[f"off|{L}"] += 1
                c[f"take|{ch_}"] += 1
                cur.append((step, ch_))
                if ch_ in ("K", "N"): turns.append(cur); cur = []
            if cur: turns.append(cur)
            c["turns"] += len(turns)
            for ti, t in enumerate(turns, 1):
                letters = [x[1] for x in t]
                for x, y in zip(["S"]+letters, letters+["X"]): c[f"dfg|{x}{y}"] += 1
                c[f"var|{''.join(letters)[:6]}"] += 1
                if ti <= 2: c[f"early|{''.join(letters)[:6]}"] += 1
                c["actions"] += len(letters)
                if letters[-1] == "N": c["pass_turns"] += 1
            # prize race
            ends = [t[-1][0] for t in turns]
            behind2 = ahead2 = False; states = []
            for es in ends:
                mine = opp = 0
                for st, p0, p1 in pr:
                    if st <= es: mine, opp = (p0, p1) if seat == 0 else (p1, p0)
                    else: break
                if opp - mine >= 2: behind2 = True
                if mine - opp >= 2: ahead2 = True
                states.append((mine, opp))
            if behind2: c["behind2"] += 1; c["behind2_won"] += win
            if ahead2: c["ahead2"] += 1; c["ahead2_lost"] += 1 - win
            # first prize: who took it
            if pr:
                st, p0, p1 = pr[0]
                mine_first = (p0 > p1) if seat == 0 else (p1 > p0)
                c["first_prize_mine"] += 1 if mine_first else 0
                c["first_prize_games"] += 1
    print("blobs", nb, round(time.time()-T0, 1), flush=True)
json.dump({str(t): {a: dict(c) for a, c in d_.items()} for t, d_ in S.items()}, open("team_stats.json", "w"))
print("done", round(time.time()-T0, 1))
