#!/usr/bin/env python3
"""Balance probes on the frozen tournament: who went first (seat whose first turn-ending decision has the
lowest step), first-player win rate overall / by rating gap / in mirror matches / by deck; game length.
Output: first_player.json"""
import sqlite3, zlib, json, time, os
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
    for eid, idx, team, pre, rew in m.execute(f"select episode_id, idx, team_id, initial_score, reward from agents where episode_id in ({ph})", ch):
        ag[(eid, idx)] = (team, pre, rew)
    for eid, a0, a1 in m.execute(f"select episode_id, arch0, arch1 from canon_arch where episode_id in ({ph})", ch):
        arch[eid] = (a0, a1)
rows = []
for ch in chunks(ids, 500):
    ph = ",".join("?"*len(ch))
    for eid, blob in e.execute(f"select episode_id, blob from events where episode_id in ({ph})", ch):
        try: d = json.loads(zlib.decompress(blob))
        except Exception: continue
        first_end = {0: None, 1: None}; nturn = {0: 0, 1: 0}
        for dec in d.get("decisions") or []:
            try: step, seat, sel, act = dec
            except Exception: continue
            if not isinstance(sel, dict) or not act: continue
            opts = sel.get("option") or []
            try: picked = opts[act[0]]
            except Exception: continue
            pt = picked.get("type") if isinstance(picked, dict) else None
            if pt in (13, 14):
                nturn[seat] += 1
                if first_end[seat] is None: first_end[seat] = step
        if first_end[0] is None or first_end[1] is None: continue
        first = 0 if first_end[0] < first_end[1] else 1
        t0, p0, r0 = ag.get((eid, 0), (None, None, None)); t1, p1, r1 = ag.get((eid, 1), (None, None, None))
        if r0 is None or r1 is None: continue
        a0, a1 = arch.get(eid, (None, None))
        rows.append([first, 1 if (r0 if first == 0 else r1) > 0 else 0, (p0 - p1) if first == 0 else (p1 - p0), a0 if first == 0 else a1, a1 if first == 0 else a0, nturn[0] + nturn[1], len(d.get("prizes") or [])])
    print("games", len(rows), round(time.time()-T0, 1), flush=True)
json.dump({"cols": ["first_seat", "first_won", "first_minus_second_rating", "first_arch", "second_arch", "player_turns", "prize_events"], "rows": rows}, open("first_player.json", "w"))
print("done", round(time.time()-T0, 1))
