#!/usr/bin/env python3
"""Multi-turn sequence mining on the frozen tournament (2026-08-17..30).
Interleave both seats' decision streams by step; for every one of MY turns t record
  - my turn signature (letters, collapsed: development = A/E/R present; attacked = ends with K; passed = ends with N)
  - what happened on the opponent's next turn (did they take a prize?)
  - what happened on MY next turn (did I take a prize? did I attack?)
Per (team, arch) counters -> multiturn.json. Also two-turn variant tokens (my turn -> my next turn), collapsed
alphabet: for each turn keep set-letters in fixed order among A,E,R,B,P plus ending K/N; e.g. 'AEN>AK'.
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
S = defaultdict(lambda: defaultdict(Counter)); nb = 0
def sig(letters):
    s = "".join(L for L in "AERBP" if L in letters); return s + letters[-1]
for ch in chunks(ids, 500):
    ph = ",".join("?"*len(ch))
    for eid, blob in e.execute(f"select episode_id, blob from events where episode_id in ({ph})", ch):
        try: d = json.loads(zlib.decompress(blob))
        except Exception: continue
        nb += 1
        streams = {0: [], 1: []}
        for dec in d.get("decisions") or []:
            try: step, seat, sel, act = dec
            except Exception: continue
            if not isinstance(sel, dict) or not act: continue
            opts = sel.get("option") or []
            try: picked = opts[act[0]]
            except Exception: continue
            pt = picked.get("type") if isinstance(picked, dict) else None
            if not any(isinstance(o, dict) and o.get("type") in ACT for o in opts): continue
            streams[seat].append((step, ACT.get(pt, "?")))
        # turns per seat: (start_step, end_step, letters)
        turns = {0: [], 1: []}
        for seat in (0, 1):
            cur = []; st0 = None
            for step, L in streams[seat]:
                if st0 is None: st0 = step
                cur.append(L)
                if L in ("K", "N"): turns[seat].append((st0, step, cur)); cur = []; st0 = None
        pr = sorted(d.get("prizes") or [])   # (step, p0, p1)
        def prizes_at(step, seat):
            v = 0
            for st, p0, p1 in pr:
                if st <= step: v = (p0, p1)[seat]
                else: break
            return v
        for seat in (0, 1):
            team, rew = ag.get((eid, seat), (None, None))
            if team is None or rew is None: continue
            a = arch.get(eid, (None, None))[seat] or "?"; c = S[team][a]; win = 1 if rew > 0 else 0
            c["games"] += 1; c["wins"] += win
            mine = turns[seat]; opp = turns[1-seat]
            for i, (s0, s1, letters) in enumerate(mine[:-1]):
                n0, n1, nletters = mine[i+1]
                # opponent turn between s1 and n0
                opp_between = [t for t in opp if t[0] > s1 and t[1] < n0]
                my_prize_now = prizes_at(s1, seat) - prizes_at(s0-1, seat)
                opp_prize_between = prizes_at(n0-1, 1-seat) - prizes_at(s1, 1-seat)
                my_prize_next = prizes_at(n1, seat) - prizes_at(n0-1, seat)
                dev = any(L in letters for L in "AER"); ended = letters[-1]
                key = ("dev" if dev else "nodev") + "_" + ("atk" if ended == "K" else "pass")
                c[f"t|{key}"] += 1
                if opp_prize_between > 0: c[f"t|{key}|opp_prize_next"] += 1
                if my_prize_next > 0: c[f"t|{key}|my_prize_next"] += 1
                if nletters[-1] == "K": c[f"t|{key}|attack_next"] += 1
                if my_prize_now > 0: c[f"t|{key}|prize_now"] += 1
                # two-turn variant token
                tok = sig(letters) + ">" + sig(nletters)
                c[f"v2|{tok}"] += 1
                c["pairs"] += 1
                # conversion after a passed development turn: prize on next turn and not punished in between
                if key == "dev_pass":
                    if my_prize_next > 0 and opp_prize_between == 0: c["setup_converted_clean"] += 1
                    if opp_prize_between > 0: c["setup_punished"] += 1
    print("blobs", nb, round(time.time()-T0, 1), flush=True)
json.dump({str(t): {a: dict(c) for a, c in d_.items()} for t, d_ in S.items()}, open("multiturn.json", "w"))
print("done", round(time.time()-T0, 1))
