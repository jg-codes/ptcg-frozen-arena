"""Phase-B heuristic pilot — clean-room, uses the official cg.api (Linux only).

Real decisions from board state + static tables:
  - attack: prefer a knockout (damage >= target HP, weakness x2), else max damage
  - energy: attach only while the attacker can't yet pay an attack (energy discipline,
            avoids the check_agent over-fill flag)
  - evolve / ability / play: advance the board; retreat only when stuck; end last
  - forced sub-selects: sane defaults (place Basics, grab attackers, pitch Energy)

Robust: every path is wrapped; on any error returns the always-legal first-minCount.
Tested via src/eval/match_runner.py on the VPS.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

from cg.api import (  # type: ignore
    AreaType,
    CardType,
    EnergyType,
    OptionType,
    SelectContext,
    all_attack,
    all_card_data,
    to_observation_class,
)

GO_FIRST = True
# SV-era resistance: the engine applies a flat -30 when the defender's resistance type matches the
# attacker's energy type. THIS heuristic does not price resistance itself; the constant is the pairing
# contract for frontier-lineage search agents, whose leaves read bare H.RESISTANCE_VALUE (their
# import-time PAIRING GUARD asserts it exists).
RESISTANCE_VALUE = 30
# B13 lead-protection: retreat a 2-prize ex the opponent can KO next turn (when not behind and a
# benched attacker is ready) -> convert a 2-prize swing into a 1-prize loss + keep the ex alive.
# VALIDATED NEUTRAL (mirror 0.501 @ N=1000, league flat/-) -> default OFF; kept as a documented
# lever. The ceiling is structural, not in single retreat rules. See report/EXPERIMENTS.md B13.
LEAD_PROTECT = False
# B33 mechanic comprehension: parse abilities/trainers (card.skills) for draw/search/energy-accel/
# heal so the agent values SETUP correctly instead of a flat constant. OFF by default (neutral);
# A/B-gated via the #29 gauntlet. The measured ceiling is the value fn being BLIND to what cards do.
COMPREHEND = False

# B35 smart damage-counter placement (Phantom-Dive-style spread → maximize KOs, not spray). Only
# active when COMPREHEND is on; SMART_PLACE exists to A/B the placement effect in isolation.
SMART_PLACE = True

# --- GRIMM-v3 R5/R6 (mirrored from sub_grimmsnarl_v3; flag-gated, default OFF) -----------------
# R5 DECK_TUTOR (config "deck_tutor"): deck-search selections (stadium/item/supporter tutors ->
# TO_HAND; Poffin-style search-to-bench -> TO_BENCH) present options as (area=DECK, index) where
# index points into obs.select.deck — the FULL revealed searched-deck list. _resolve has no DECK
# branch, so every such pick scored a flat 8.0 and _choose took the lowest index: an effectively
# RANDOM fetch. Measured on the grimmsnarl 60 (report/grimm_v3/, 2026-07-02): tonakaiiii fetches
# line pieces (GS 94 / Impidimp 84 / Morgrem 74 / Munkidori 54 over 47 eps; Morgrem online his
# t2 in 46/47), while our 15 v2 SETUP-FAILURE losses had 2-10 multi-choice blind deck searches
# before Grimmsnarl ever went active (14/15 games). When ON, deck-searched cards resolve via
# select.deck and score by line-completion need (_tutor_value) — deck-generic logic.
DECK_TUTOR = False
# R6 SETUP_PRIORITY (config "setup_priority"): SETUP_ACTIVE_POKEMON options scored a flat 50.0 ->
# first index. Prefer the deep evolution-line starter (+12) > energy-gated-ability basic (+6) >
# 1-stage line basic (+2). tonakaiiii opens Impidimp 24/47; 4/4 when holding Impidimp+Munkidori.
SETUP_PRIORITY = False

# --- PROCESS-MINING-DERIVED leaf levers (A/B-gated via src/eval/ab_gauntlet.py; default OFF) ---
# SITU_AGGRO: the scorer was prize-state-BLIND. Mining (perturn_situational): winners pressure
# (ATTACK/EVOLVE, END less) when BEHIND and convert leads when AHEAD. Modulate the chip-attack base
# + overfill penalty by the prize differential pd = len(opp.prize)-len(me.prize) (pd<0 == behind).
SITU_AGGRO = False
# PATIENCE: the #1 data-grounded pro-vs-us divergence (wf pro-skill mining, 8k top-player eps): our
# leaf is STAGE-BLIND and over-attacks — when AHEAD pros attack 0.187 vs our 0.385 (gap 0.198, z=40.5,
# N=16270); the gap WIDENS late (+0.077 early → +0.204 late); develop-first 0.751 (N=10307).
# MECHANISM (a HOLD, not a damp): when AHEAD or late, a NON-threatening chip returns PATIENCE_HOLD so
# the agent develops / passes / protects the lead instead of chipping; a threatening (near-KO) chip is
# always taken; lethal (10000) is UNTOUCHED. A multiplicative damp was tried and rejected — our chip
# (~40) sits isolated far above the only competing options when we'd attack (END=1.0, overfill=-20;
# the positive setups attach 200 / evolve 400 are already chosen when available), so a damp is bimodal
# (no-op or collapse to 0).
# ⚠️ STAGE-A SCREEN (2026-06-27, stage_a.py, N=7299) FALSIFIED this leaf fix: it LOWERS leaf-vs-pro
# agreement (0.296→0.283 at every THREAT_FRAC 0.5–0.95) and only nudges attack-rate (0.328→0.306) — a
# blunt hold suppresses chips pros DO take. The over-attack gap is REAL but NOT leaf-capturable (which
# chip to hold is context-specific: multi-turn line / bench backup / exposure) → the fix belongs in the
# PLAN / learned-policy layer, not a leaf rule. Kept GUARDED OFF + documented (cf. TEMPO); ladder judges.
PATIENCE = False
PATIENCE_THREAT_FRAC = 0.50   # active_dmg >= this fraction of the target's HP => "threatening" => still taken
PATIENCE_HOLD = -2.0          # score of a held weak chip (below END=1.0 => develop/pass wins)
# EVOLVE_TEMPO (2026-08-01, measured spec in the EVOLVE band comment + task #12): phase-declining
# evolve ordering toward the strong-player take-rates. Demotions sized against the band map —
# evolve base 400, attach ~200, END 1: MID drops evolve below attach (400-250=150) so develop wins
# the tie; LATE drops it near END-adjacent (400-320=80); ACTIVE_EARLY keeps it above attach
# (400-120=280) — matching 0.50-vs-0.70. Escape-evolve (2000) and lethal are never demoted.
EVOLVE_TEMPO = False
EVOLVE_TEMPO_MID = 250.0
EVOLVE_TEMPO_LATE = 320.0
EVOLVE_TEMPO_ACTIVE_EARLY = 120.0
# WALL-PIVOT + DEAD-ATTACK (2026-08-01, probe_attack_mix.py + the wall census): 20.5% of our Mega
# Kangaskhan attacks deal 0 — 94% into Crustle-family walls (Rock Inn) — in streaks up to 14 turns
# with RETREAT offered in 92% of those frames; games containing them win 0.200 [0.129,0.297] vs
# 0.502 elsewhere. The immunity guard below zeroes active_dmg but the chip BASE survives, so the
# 0-conversion attack still outranks END(1.0)/RETREAT(-30) and the agent attacks the wall forever.
# Distinct from the refuted PIVOT_VALUE (an S-leaf posture term): these are H ordering rules —
# DEAD_ATTACK demotes a no-value attack below END; WALL_PIVOT gives RETREAT a rule band when the
# active CAN pay but converts 0 while a FUELED bench mon converts >0 (also lifts the pivot into the
# search's TOP_M candidate window at exactly these frames).
DEAD_ATTACK = False
DEAD_ATTACK_SCORE = -10.0     # below END(1.0): a no-value turn-ender loses to develop/pass
WALL_PIVOT = False
WALL_PIVOT_BAND = 900.0       # above every develop band; below protect-band urgency (CLASS-A 3000)
# WALL-AWARE ATTACH. wall_pivot treats the SYMPTOM (retreat once already stuck behind a wall) and
# was measured firing in 0.9% of the frames it targets — 55 of 6,423 — because its precondition
# demands a bench Pokemon that is ALREADY fuelled, and the fuelled one is the ex attacker the wall
# blocks. This treats the CAUSE: stop putting energy on a walled attacker in the first place.
WALL_ATTACH = False
WALL_ATTACH_BAND = 120.0      # > TEMPO's max 80.0, so a walled attacker cannot win on "closest to
                              # online"; small enough that a genuine lethal band still outranks it
# ROCK_INN_SCOPED (2026-08-09) — the wall is ATTACK-scoped, not POKEMON-scoped. `_ex_immune_vs`
# answers "is the attacker an ex in the ability's named scope", which the whole threat model then
# reads as "this Pokemon can do us no damage". The engine disagrees: 11 of the 1,556 attacks in the
# card DB carry "This attack's damage isn't affected by ... any effects on your opponent's Active
# Pokemon" and go straight through the wall. Six owners are ex (Cornerstone Mask Ogerpon ex 117,
# Tatsugiri ex 231, Dudunsparce ex 306, Keldeo ex 583, Mega Lopunny ex 849, Mega Starmie ex 1031).
# The decisive proof of attack-scope sits inside ONE card: measured in ocel ATTACK->HP_CHANGE
# windows into Crustle, Mega Starmie ex's Jetting Blow never once landed its printed 120 in 8,654
# windows, while its Nebula Beam landed exactly 210 in 3,571 (tests/test_ex_immunity.py E3;
# scripts/probes/probe_rock_inn_windows.py). The detector for the clause already exists in this
# module (`_attack_pierces_effects`, used as the DEAD_ATTACK exemption) — the codebase recognised
# the concept and routed around the hole instead of closing it.
# Flag OFF => every helper below returns exactly what `_ex_immune_vs` / `_strongest_attack` returned
# (same call count, no extra regex work), so the incumbent is bit-for-bit preserved.
# FIRES counters (ship_gate 5d): DIAG rock_inn_scoped (scoped check ran against an engaged wall) /
# rock_inn_pierced (a clause-carrying attack diverged from the card-scoped answer); ON branch only.
ROCK_INN_SCOPED = False
# PROMO_BANDS (2026-08-09, playbook Rule B) — the promotion BAND ARITHMETIC in `_promotion_value`
# promotes a 70-80hp chaff Basic over the wall/attacker already sitting on our bench. Measured on the
# DECISION (ladder_replays; SWITCH(3)/TO_ACTIVE(4) menus whose every option is one of OUR bench slots
# and whose menu holds >=1 of {Crustle 345, Mega Kangaskhan ex 756} AND >=1 of {Dwebble 344,
# Shaymin 343}): we take the real body 0.6337 [0.6191, 0.6480] (episode-clustered [0.6172, 0.6503],
# n=4,256 / 1,687 seats) vs teacher seats 0.9171 [0.9128, 0.9212] (n=16,439) and ladder peers INSIDE
# the same corpus 0.8554 [0.7641, 0.9153] (n=83) — so the gap is not an ingestion artefact. The
# situation arises 1.89x/game, 0.69 addressable picks/game. Fueled/unfueled stratified (US-weighted
# standardized gap +0.272), so "our attacker was unusable anyway" does not explain it, and it is not
# an index-0 artefact (we pick idx 0 in 46.2%, teachers 52.5%).
# NOT the prize-exposure penalty — that hypothesis is REFUTED: Crustle is a 1-prize Stage 1, so the
# `-1000 x prizes` term cannot reach the Crustle cells at all (~58% of the US-weighted gap), and our
# deficit is WORST exactly where the penalty does NOT fire (Crustle unfueled, Dwebble at 0 energy, no
# Mega: 0.4845 [0.4280, 0.5420] n=291 when the opponent's active canNOT KO Crustle vs 0.8462
# [0.7910, 0.8890] n=208 when it can, against a flat 0.963/0.963 for teachers).
# TWO ARITHMETIC DRIVERS, both fixed here:
#   (1) the line-seed +150 outranks the hp tiebreak, so on the live deck (88cc3fe1811d)
#         Dwebble 3050 + 150 + 0.5*70 = 3235  >  Crustle 3050 + 0 + 0.5*150 = 3125
#       and 3235 also beats the Mega's 3050 + 0 + 0.5*300 = 3200. ON: in the to_active contexts the
#       seed bonus becomes PROMO_SEED_EPS = 1.0, strictly BELOW the smallest hp step (0.5 * 10hp =
#       5.0) — a seed still breaks an EXACT-hp tie, and can never again outrank a bigger body.
#       TO_FIELD (benching into play) keeps the full +150: developing the line off the bench is what
#       the bonus was FOR, and Rule B's situation is SWITCH/TO_ACTIVE only. The "suppress the bonus
#       when an evolved attacker shares the menu" variant was rejected on the engine: Mega
#       Kangaskhan ex has evolvesFrom=None (it is a Basic), so that variant is blind to the
#       Mega-only cell — 42% of the gap.
#   (2) `_can_pay_any` does not require damage > 0, so a Dwebble holding one energy earns the +8000
#       "fueled attacker" band through its 0-damage Ascension (478, 0 dmg, 1 colorless). In those
#       menus we take the real body 0.0395 [0.0180, 0.0830] n=152 vs 0.4742 with an unenergised
#       Dwebble (teachers 0.5796 vs 0.9042). ON: the fueled band — and the wall-denial "unfueled"
#       test that mirrors it — reads `_can_pay_damaging`.
#       RESIDUAL, deliberately NOT fixed: the separate `+120 x energies` fueling-progress band is
#       untouched, so an energised Dwebble (3050+120+1+35 = 3206) still outranks a 0-energy Crustle
#       (3125) and a 0-energy Mega (3200). That band is a third mechanism, unverified by the Rule B
#       census, and inventing a fix for it here would smuggle an unmeasured lever into a measured
#       one. Pinned as an exact identity in test_promobands_byte_identical.py check 6.
# The prize-exposure DISCRETION is deliberately left alone: teachers promote a full-hp 300 Mega only
# 0.7991 [0.7870, 0.8110] (n=4,361; 0.8341 once it is already damaged), so the `-1000 x prizes` and
# wall-denial terms that keep a 3-prize body off the front when it can be KO'd are the RIGHT
# behaviour and are not part of the defect. This lever must not drive the Mega branch to 1.0.
# SCOPE EXTENSION (2026-08-10): the census family is {SETUP_ACTIVE(1), SWITCH(3), TO_ACTIVE(4)}
# and the two drivers above covered only 3/4 — ctx 1 (587 decisions = 12% of the family, all
# mega_only, us 0.4787 vs teachers 46/46) is closed in _score's SETUP branch by scoring the setup
# pick with the same _promotion_value ladder via _SetupShim (hand Cards carry no hp/energies),
# GUARDED to menus offering a >=2-prize body (= the measured cell; unguarded it also flipped 60
# unmeasured no-Mega picks against the teachers' seed preference). See that branch.
# FIRES counters (ship_gate 5d): promo_bands / promo_bands_seed / promo_bands_zero_dmg /
# promo_bands_setup in DIAG — a dead flag on the ladder must be measurable, not invisible.
# Flag OFF => the seed bonus is the literal 150.0 and the fueled test is the literal `_can_pay_any`
# call (same call, same count, no new work), so the incumbent is bit-for-bit preserved.
PROMO_BANDS = False
PROMO_SEED_EPS = 1.0
# ATTACH_PRIZE (2026-08-09, playbook Rule A) — the ATTACH branch is PRIZE-BLIND. POLICY_W encodes
# exactly three attach terms (attach 200.0 / attach_active 50.0 / overfill -20.0) and not one of
# them reads CardData.ex / CardData.megaEx, so the agent cannot prefer fuelling the body that wins
# the game (3-prize Mega Kangaskhan ex 756) over the body that does not (1-prize Crustle 345,
# Dwebble 344, Shaymin 343). The target identity is ALREADY computed one line above the terms that
# ignore it (`_energy_opt_ref` / `_resolve` -> CARDS.get), so this adds no state.
# THE BLINDNESS PROOF (why this is a defect, not a style): our attach-to-active rate is 0.3966 when
# the active is the 3-prize body and 0.3974 when it is a 1-prize body — delta 0.0008, flat, which is
# exactly what a POLICY_W with only attach/attach_active/overfill predicts. Teacher seats modulate:
# 0.5148 vs 0.3234, delta 0.191.
# MEASURED (fraction of these menus where the attach lands on a >=2-prize body): US ladder_replays
# 0.3698 [0.3615, 0.3781], episode-clustered [0.3588, 0.3813], n=12,981 decisions / 2,047 seats;
# TEACHERS episodes_full 0.5346 [0.5304, 0.5387], clustered [0.5254, 0.5441], n=56,045 / 9,261.
# Crude gap -0.1648; standardised to the teacher BOARD MIX (composition-free) -0.0949 — quote the
# standardised one. Tier gradient rho +0.765 p<0.001 (16 teams, n>=200; +0.714 without Budew, +0.839
# without one high-volume entrant), against a matched prize-FREE control (attach_propensity on the same menus)
# rho -0.006 p=0.983, so the instrument is not generic "good players attach more". Corpus control
# over the 8 decks present in BOTH sources: mean ladder-minus-full delta +0.0144, far short of
# -0.165, so the ingestion difference does not explain it. The gap WIDENS under filters that blunt
# reverse causation (not-ahead-on-prizes -0.197; turn<=6 -0.224; non-overfill attaches only -0.181).
# THE ACTIONABLE STRATUM is where the fix has to bite: active SMALL, big body BENCHED — 53.6% of our
# menus, US 0.2351 vs TCH 0.4690 (gap -0.234, n=6,962). When the active IS the big body the gap is
# only -0.037, i.e. already nearly covered by attach_active +50.
# SHAPE: a PENALTY on the 1-prize option, not a bonus on the >=2-prize one. Same order INSIDE the
# attach options, but the attach band's CEILING is unchanged, so nothing can leak up past
# ability(300)/play(350)/evolve(400). Flat (not scaled by 2 vs 3 prizes) because the containment
# inequalities below are exact only for a flat term — the wall band leaves no room for a 2x version.
# THE PENALTY IS WITHHELD in three cases, each of which is an existing rule's territory:
#   * OVERFILLED alternative — `_can_pay_strongest` already returns overfill(-20) for it, so
#     steering toward it would be steering at a target the incumbent refuses; and the whole term
#     sits AFTER that early return, so POLICY_W["overfill"]'s sign (UNRESOLVED: rho -0.573 p=0.009
#     against rho +0.371 p=0.161 on different corpora) is untouched.
#   * WALLED alternative — `_wall_blocks_all(opp_active, alt)`: WALL_ATTACH (which SHIPS ON) exists
#     to move the attach OFF a Mega whose damage the opponent's Rock Inn prevents. Withholding the
#     penalty when the only big body is walled makes the two rules disjoint by construction rather
#     than by arithmetic, and reading the same helper means ROCK_INN_SCOPED's attack-scoped
#     redefinition flows through automatically instead of desynchronising.
#   * DOOMED-ACTIVE SINK — the Class-A LOSS_AUTOPSY_2026-07-04 rule already flips the active's bonus
#     to -150; stacking a second penalty double-counts and would push the attach under END(1.0).
# BAND DERIVATION, both bounds forced by live constants (asserted in the test, not chosen by taste):
#   lower  BAND > POLICY_W["attach_active"] + FUEL_CAP = 50.0 + 6.0 = 56.0
#          => the benched >=2-prize body outranks the 1-prize ACTIVE even in the worst fuel split,
#             which is the actionable stratum;
#   upper  BAND < WALL_ATTACH_BAND - POLICY_W["attach_active"] - FUEL_CAP = 120.0 - 56.0 = 64.0
#          => a WALLED big body (200 + 50 - 120 + 6 = 136) can never outrank the cheapest penalised
#             1-prize option (200 + 0 + 0 - BAND = 140), i.e. wall_attach can never be INVERTED.
#   60.0 is the midpoint of (56, 64).
# KNOWN RISK, reported not resolved (it points the WRONG WAY and no patch here can settle it):
# within our OWN seats, turn<=6 any-big-attach win rate 0.4490 [0.4139, 0.4847] vs no-big-attach
# 0.5182 [0.4816, 0.5546]. Seat-level and confounded by board composition (a seat that HAS a Mega to
# fuel is a different seat), so it cannot convict — but it is not evidence for the rule either.
# And the actionable stratum's own tier gradient is the weak half of the claim: rho +0.333 p=0.396
# (9 teams) — a positive point estimate under the relaxed bar, nothing more.
# 🔴 MEASURED 2026-08-09 ON THE REAL LADDER DECISIONS — READ BEFORE ENABLING THIS FLAG.
# scripts/probes/probe_attachprize_materiality.py, 13,251 frames / 2,027 episodes, both arms twice.
# The instrument reproduces the brief's headline EXACTLY (us 4800/12981 = 0.3698 [0.3615, 0.3781],
# same n to the digit), which is what makes the two corrections below credible.
#  (1) THE GAP IS MOSTLY A DECK CONFOUND. The -0.1648 "teachers" gap reproduces only against seats
#      on OTHER DECKS (any-deck peers 0.4766 [0.4723, 0.4808], n=53,272 / 8,580 episodes across 138
#      decklists). Restricted to peers piloting OUR EXACT 60, the overall gap VANISHES: peers 0.3625
#      [0.3259, 0.4008] (n=629 / 124 eps, episodes_full) and 0.4185 [0.3612, 0.4781] (n=270 / 35 eps,
#      ladder_replays) against our 0.3698 — we are AT PARITY, point estimate slightly above. This is
#      the same defect class the ABILITY lever was retracted for (project_ptcg_ability_divergence:
#      the "same decklist" control was FALSE).
#      What SURVIVES deck control is the actionable stratum alone: big_benched us 0.1268 [0.1199,
#      0.1341] vs same-60 peers 0.2284 [0.1837, 0.2801] — disjoint, but a ~-0.10 gap, not -0.234.
#  (2) THE PATCH OVERSHOOTS THAT TARGET BY ~4x. Flag ON, matched denominator (both arms attached,
#      n=11,386): big_benched 0.0768 [0.0695, 0.0840] -> 0.6616 [0.6457, 0.6765] against a
#      deck-controlled peer benchmark of 0.2284 and the brief's own teacher target of 0.4690; ALL
#      0.3006 -> 0.7093 against peers 0.3625 and the brief's teacher 0.5346. Decision change is
#      real and large — NET +0.3604 [0.3473, 0.3725] over arm-specific floors 0.0374 / 0.0428 — and
#      it lands squarely in the cell risk (b) flags: turn<=6 big-attach rate 0.2684 -> 0.6430.
#      AND THE BAND CANNOT DIAL IT BACK. The scores it competes with are exact constants, so 56 <
#      band < 64 flips EVERY qualifying frame and no value in between yields an intermediate rate.
#      The 33.8% of big_benched that stays put is entirely the withholding conditions and search
#      re-ranking (measured: 32.4% every big alternative WALLED, 2.1% every big alternative
#      OVERFILLED, 65.5% tie / non-attach / sink), not near-threshold frames.
#      The only evidence-shaped way to scope it further would be to withhold in the big_active
#      stratum — where we already sit at 0.8170 against same-60 peers' 0.4765, i.e. there is no
#      deficit to close and the flag still adds +0.0776. That is a SECOND, unmeasured lever and is
#      deliberately NOT added here.
# CONCLUSION: mechanically correct, byte-identical off, and directionally aimed at a real
# deck-controlled deficit — but over-scaled against every benchmark that survives deck control. Do
# not enable without deciding that the overshoot is acceptable.
# MEASURED RESIDUAL, scoped and NOT fixed here (test_attachprize_byte_identical.py check 6/7): with
# WALL_ATTACH **off** — which is NOT the ship, config_minepilot.json turns it on — the incumbent
# scores a walled and an unwalled >=2-prize body IDENTICALLY (both bare `attach`), so removing the
# 1-prize option that sat on top of both drops the argmax onto whichever has the lower option index.
# 5 of 2,100 captured menus, only in the (rock_inn_scoped ON, wall_attach OFF) combination, and 0 of
# them moved away from an already->=2-prize choice. The tie is wall_attach's absence, not this rule;
# duplicating wall_attach's discipline inside Rule A would conflate two levers, so it is pinned as a
# test-reported count instead. At wall_attach ON the count is 0 in every combination.
# Flag OFF => `_attach_prize_alt` is never called and `score` is the incumbent expression verbatim.
# FIRES counters (ship_gate 5d): DIAG attach_prize_checked (eligible 1-prize attach scanned) /
# attach_prize (penalty landed); ON branch only.
ATTACH_PRIZE = False
ATTACH_PRIZE_BAND = 60.0
# TEMPO: mining (perturn_firstattack) — winners get an attacker ONLINE ~0.4 turns sooner. Reward
# the ATTACH that crosses a Pokemon to "can pay an attack now" (completes an attacker vs spreading).
# (A/B'd 2026-06-27: HARMFUL −0.245 — crude fuel-concentration backfires. Kept OFF as documented.)
TEMPO = False
# PLAN (multi-turn pilot, step 1): focus development on the deck's WIN-CONDITION line instead of
# greedily developing any Pokemon. _compute_plan() finds the primary attacker (highest-value attack,
# ex/megaEx weighted) + its in-deck evolution chain; _score boosts EVOLVE/ATTACH/PLAY toward that
# line. Targets the combo-piloting hole (deck_compare: heuristic pilots combo decks at ~0.33).
PLAN = False
# ATTACH_FIRST (2026-07-16 T3 diagnosis, report/PLAYBOOK.md gap 1a; evidence ~/ptcg/_t3diag/):
# the energy attach is a FREE, once-per-turn, EXPIRING resource, and it does NOT end the turn — but a
# lethal ATTACK returns 10000+dmg (:_score ATTACK lethal) which dominates the attach band (200/250), so
# on **27.8%** [25.5, 30.2] of attach-legal turns we take the KO and simply throw the attach away.
# 132/141 sampled declines were a lethal, and **98.0%** [95.4, 99.1] of them occur in games that CONTINUE
# — attach-then-attack is legal (observed 116x/40 games), so both plays fit in the same turn.
# MECHANISM: rank a BENEFICIAL attach above the lethal band. Because attach never ends the turn and is
# once-per-turn, the agent attaches, the ATTACH options then disappear, and the lethal is taken on the
# very next option — the KO is deferred by one decision, never lost.
# SCOPE (deliberately narrow — these two exclusions are the whole safety argument):
#   * OVERFILL attaches never reach here (the `_can_pay_strongest` early-return fires first), so a
#     fully-fueled target is not elevated;
#   * a DOOMED-ACTIVE energy sink (`sink` below, the Class-A LOSS_AUTOPSY_2026-07-04 pattern) is NOT
#     elevated — pumping energy into a mon that dies next turn must not jump the lethal.
# Relative order WITHIN the attach options is preserved (the band is a constant), so the bench-vs-active
# preference and the PLAN/TEMPO shaping are unchanged.
# ⚠️ Behaviour-changing => GUARDED OFF; the ladder judges (offline win-rate is anti-predictive). Bounded
# upside: the teacher wastes its OWN attach on 16.7% [13.0, 21.3] of its attacks. This does NOT address
# the larger bench-TARGETING gap (PLAYBOOK gap 1b) — that needs a prune change too.
ATTACH_FIRST = False
# Must exceed the lethal band's maximum (10000 + active_dmg; active_dmg is bounded by the biggest
# hand-scaled attack, Resentful Refrain 50x opp hand ~= a few hundred). 12000 clears it with room.
ATTACH_FIRST_BAND = 12000.0

# Hand-set POLICY priorities — these define the agent's whole playstyle (when to develop vs attach
# vs attack) and were NEVER tuned. Exposed for league/self-play optimisation (the untested lever
# that targets the STRONG component: the policy, not the thin search layer on top).
POLICY_W = {
    "attack_chip": 40.0,      # non-lethal attack base (vs setup actions)
    "attack_dmg": 0.1,        # chip damage scaling
    "attach": 200.0,          # fuel an attacker
    "attach_active": 50.0,    # bonus for fueling the ACTIVE attacker
    "overfill": -20.0,        # penalty once the line's top attack is paid
    "evolve": 400.0,
    "ability": 300.0,
    "play": 350.0,            # develop board / play trainers before attacking
    "retreat_stuck": 120.0,   # retreat a fuel-less active when a fueled benched exists
    "protect_ex": 500.0,      # B13: retreat a threatened 2-prize ex to deny the swing (outranks develop, below a KO)
    "yes_activate": 20.0,
}

# --- OPTIONAL per-archetype POLICY profiles (arena/policy_db.json) ------------------------------
# BACKWARD-COMPATIBLE: nothing below runs unless main.py is told (cfg.policy_profile) to load a
# profile. With no profile loaded POLICY_W is the shipped dict above verbatim, so SUB-1..5 reproduce
# byte-identical. A profile is a PARTIAL override merged onto POLICY_W (keys it omits keep their
# shipped value, e.g. yes_activate). "default" == champion_genome.json['pw'] (the tuned champion).
POLICY_PROFILE = None     # name of the active profile (None => shipped POLICY_W, untouched)
_POLICY_BASE = dict(POLICY_W)   # immutable snapshot of the shipped weights (for re-load / reset)


def _resolve_policy_db_path(db_path):
    """Find policy_db.json across the dev (~/ptcg/arena) and Kaggle (/kaggle_simulations) layouts."""
    cands = [db_path] if db_path else []
    cands += ["arena/policy_db.json", "policy_db.json",
              "/kaggle_simulations/agent/policy_db.json",
              "/kaggle_simulations/agent/arena/policy_db.json"]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cands += [os.path.join(here, "policy_db.json"),
                  os.path.join(here, "arena", "policy_db.json")]
    except NameError:
        pass
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def load_policy_profile(profile="default", db_path=None) -> bool:
    """Merge the named POLICY profile from policy_db.json onto POLICY_W (partial override).

    Returns True if a profile was applied, False otherwise (missing db / unknown profile / error)
    — on any failure POLICY_W is left exactly as the shipped dict, so the agent degrades to baseline.
    Profiles may ALIAS another profile by storing its name as a string (e.g. vs-aggro -> vs-lucario_v3).
    Called ONLY from main.py when cfg.policy_profile is set; never at import => baseline is untouched.
    """
    global POLICY_PROFILE
    if not profile:
        return False
    try:
        import json
        p = _resolve_policy_db_path(db_path)
        if not p:
            return False
        with open(p) as f:
            db = json.load(f)
        prof = db.get(profile)
        resolved = profile
        seen = set()
        while isinstance(prof, str) and prof not in seen:   # resolve alias chain
            seen.add(prof)
            resolved = prof          # track the RESOLVED target so POLICY_PROFILE names the applied weights, not the alias
            prof = db.get(prof)
        if not isinstance(prof, dict):
            return False
        merged = dict(_POLICY_BASE)                          # start from the SHIPPED weights...
        for k, v in prof.items():
            if k.startswith("_"):
                continue
            try:
                merged[k] = float(v)                         # ...override only the keys the profile sets
            except (TypeError, ValueError):
                continue
        POLICY_W.clear()
        POLICY_W.update(merged)
        POLICY_PROFILE = resolved
        return True
    except Exception:
        # never let a bad db break the agent — keep the shipped POLICY_W
        return False


# --- THE DECK-PLAYING LAYER: auto-pilot the deck WE hold from its own playbook -------------------
# The policy profiles above are per-OPPONENT (how froslass plays vs X). This layer is per-OWN-DECK:
# it detects which deck we are piloting and loads that deck's playbook (pw overrides + capability
# flags COMPREHEND/SMART_PLACE) so a non-froslass deck is no longer mis-piloted by froslass-tuned
# weights. BACKWARD-COMPATIBLE: only runs when main.py sets cfg.autopilot; absent the db or a
# confident match it leaves POLICY_W/flags exactly as shipped (froslass byte-identical).
PILOT_ARCHETYPE = None     # the own-deck archetype we auto-piloted (None => no autopilot applied)


def _resolve_pilot_db_path(db_path):
    cands = [db_path] if db_path else []
    cands += ["arena/pilot_db.json", "pilot_db.json",
              "/kaggle_simulations/agent/pilot_db.json",
              "/kaggle_simulations/agent/arena/pilot_db.json"]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        cands += [os.path.join(here, "pilot_db.json"), os.path.join(here, "arena", "pilot_db.json")]
    except NameError:
        pass
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def detect_archetype(deck, db, min_overlap=0.55):
    """Which playbook's canonical deck best matches `deck`? Multiset-overlap |deck ∩ canon| / 60.

    Returns (archetype, score) for the best match with score >= min_overlap, else (None, 0.0). The
    deck WE pilot is usually an exact copy of one playbook's canonical deck (score ~1.0); the overlap
    tolerates minor tech swaps. Self-contained (ships in pilot_db.json) so it works in the container."""
    mine = Counter(int(c) for c in deck)
    n = sum(mine.values()) or 60
    best, best_score = None, 0.0
    for name, pb in db.items():
        if name.startswith("_") or not isinstance(pb, dict):
            continue
        canon = pb.get("deck")
        if not canon:
            continue
        theirs = Counter(int(c) for c in canon)
        inter = sum((mine & theirs).values())
        score = inter / n
        if score > best_score:
            best, best_score = name, score
    return (best, best_score) if best_score >= min_overlap else (None, best_score)


def autopilot_from_deck(deck=None, db_path=None, min_overlap=0.55) -> str | None:
    """Detect the archetype of the deck we hold and apply ITS playbook (pw overrides + flags).

    Returns the archetype applied, or None (no db / no confident match / error) — in which case the
    agent is left exactly as shipped. Called ONLY from main.py when cfg.autopilot is set."""
    global PILOT_ARCHETYPE, COMPREHEND, SMART_PLACE
    try:
        import json
        p = _resolve_pilot_db_path(db_path)
        if not p:
            return None
        with open(p) as f:
            db = json.load(f)
        arch, _score = detect_archetype(deck if deck is not None else MY_DECK, db, min_overlap)
        if not arch:
            return None
        pb = db.get(arch) or {}
        pw = pb.get("pw")
        if isinstance(pw, dict):
            merged = dict(_POLICY_BASE)
            for k, v in pw.items():
                if k.startswith("_"):
                    continue
                try:
                    merged[k] = float(v)
                except (TypeError, ValueError):
                    continue
            POLICY_W.clear()
            POLICY_W.update(merged)
        if "comprehend" in pb:
            COMPREHEND = bool(pb["comprehend"])
        if "smart_place" in pb:
            SMART_PLACE = bool(pb["smart_place"])
        PILOT_ARCHETYPE = arch
        return arch
    except Exception:
        return None


# --- ENH 3+4: OPPONENT-CONDITIONAL piloting (classifier -> per-archetype POLICY_W shift) ---------
# The policy profiles above are loaded STATICALLY from a config string. This layer makes them
# DYNAMIC: once the learned archetype classifier (arena/index_db/archetype_clf.py, ~0.90 acc by t3)
# is confident, we merge that opponent's policy_db profile onto POLICY_W mid-game — the "consume the
# opponent ID" play. FLAG-OFF by default (OPPONENT_ADAPT=False) => byte-identical; the per-opponent
# profiles are HAND-SET SEEDS, UNPROVEN until a classifier-ON field A/B clears them (ENHANCEMENT_PLAN
# enh 5). The classifier + card_index ship under index_db/; if absent the wire degrades to a no-op.
OPPONENT_ADAPT = False
OPP_ADAPT_MIN_TURN = 3        # classifier reaches ~0.90 archetype acc by turn 3 (model heldout table)
OPP_ADAPT_TAU = 0.6           # min MAP confidence to apply a shift (else keep the current weights)
# enh 4: explicit classifier-class -> policy_db key map. Fixes the silent no-op — the classifier
# emits 'lucario_v3' but the db key is 'vs-lucario_v3'. An UNMAPPED class => no shift (keep default),
# so classes with no tuned profile (bellibolt/chandelure/megastarmie/mewtwo/typhlosion/froslass/other)
# never force a bogus lookup.
_ARCH_TO_PROFILE = {
    "lucario_v3": "vs-lucario_v3",
    "alakazam": "vs-alakazam",
    "alakazam_mist": "vs-alakazam",
    "dragapult": "vs-dragapult",
    "trevenant": "vs-trevenant",
}
_clf_predict = None                                   # lazily-bound archetype_clf.predict (False=failed)
_OPP_ADAPT_STATE = {"applied": None}                  # archetype currently merged (avoid re-merge churn)


def _load_clf():
    """Lazily import the productionized archetype classifier from index_db/. Returns the predict fn,
    or False if unavailable (no model / no card_index — e.g. local dev) so the wire degrades to no-op."""
    global _clf_predict
    if _clf_predict is not None:
        return _clf_predict
    import sys
    here = None
    try:
        here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index_db")
    except NameError:
        pass
    for d in (here, "index_db", "/kaggle_simulations/agent/index_db", "arena/index_db"):
        if d and os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        import archetype_clf
        _clf_predict = archetype_clf.predict
    except Exception:
        _clf_predict = False
    return _clf_predict


def _opp_seen_ids(obs) -> list:
    """Opponent card ids revealed so far INCLUDING attached energies (enh 4 energy-tell fix: the
    classifier keys partly on energy ids that a pk.id-only collector misses)."""
    st = obs.current
    opp = st.players[1 - st.yourIndex]
    seen: list[int] = []
    for zone in ((opp.active or []), (opp.bench or []), (getattr(opp, "discard", None) or [])):
        for pk in zone:
            if pk is None:
                continue
            cid = getattr(pk, "id", None)
            if cid is not None:
                seen.append(int(cid))
            for e in (getattr(pk, "energies", None) or []):       # attached energies = the tell bits
                try:
                    seen.append(int(e))
                except (TypeError, ValueError):
                    pass
    return seen


def apply_opponent_shift(obs, turn: int = 99, db_path=None) -> str | None:
    """Merge the classifier-MAP opponent's policy_db profile onto POLICY_W. No-op (None) unless
    OPPONENT_ADAPT, turn >= MIN_TURN, a confident MAP (conf >= TAU), and a MAPPED profile. Re-merges
    only when the archetype CHANGES (load_policy_profile restarts from _POLICY_BASE each time, so a
    flip is clean). Never raises into the agent — degrades to the current weights on any failure."""
    if not OPPONENT_ADAPT or turn < OPP_ADAPT_MIN_TURN:
        return None
    pred = _load_clf()
    if not pred:
        return None
    try:
        seen = _opp_seen_ids(obs)
        if not seen:
            return None
        res = pred(seen, turn)
        arch = res.get("archetype")
        if res.get("confidence", 0.0) < OPP_ADAPT_TAU:
            return None
        key = _ARCH_TO_PROFILE.get(arch)
        if not key:
            return None                                # unmapped class -> keep current weights
        if _OPP_ADAPT_STATE["applied"] == key:
            return key                                 # already merged this archetype -> skip churn
        if load_policy_profile(key, db_path):
            _OPP_ADAPT_STATE["applied"] = key
            return key
    except Exception:
        return None
    return None


# static tables (loaded once)
try:
    CARDS = {c.cardId: c for c in all_card_data()}
    ATTACKS = {a.attackId: a for a in all_attack()}
except Exception:
    CARDS, ATTACKS = {}, {}

_COLORLESS = int(EnergyType.COLORLESS)
_WILD = {int(EnergyType.RAINBOW)}  # provides any type


MY_DECK_PATH = None       # the deck.csv _deck() actually read (None => no deck found).
                          # Auditable on purpose: "which deck is this process playing?" was not
                          # answerable from inside the running agent, which is how the substitution
                          # below survived every green preflight.


def _deck_candidates() -> list:
    """deck.csv search order, most specific first. The CWD entry is LAST BY CONTRACT.

    🔴 THE SILENT-SUBSTITUTED-DECK BUG (measured 2026-08-09, the sibling of main.py's config bug).
    This list used to start with the bare "deck.csv", i.e. the CALLER'S CWD. `scripts/preflight.py`
    and every offline harness run from ~/ptcg, where a stale ~/ptcg/deck.csv holds the EXTINCT
    froslass list — the very list build_bundle.sh's own comment says invalidated ~3 weeks of A/Bs.
    Measured directly (sys.addaudithook on the "open" event, real cabt game, bundle sub_minepilot):
    the harness-built agent opened <workdir>/deck.csv and set MY_DECK to md5 00df71b9be13
    (froslass) while the bundle shipped 88cc3fe1811d (crustle_teacher). preflight check 2 only
    asserted `len(deck) == 60`, so the gate was green on an agent playing a deck the bundle does
    not contain.

    Order mirrors main.py._config_candidates() exactly — module dir, the explicit Kaggle bundle
    root, CWD last — and for the same reason: the deck a module plays is a property of WHERE THAT
    MODULE LIVES, not of who invoked it. The CWD candidate is demoted, not deleted, so a harness
    that drops a deck.csv beside a module-dir-less copy still resolves.

    NOTE: kaggle_environments EXECs main.py without defining __file__; this module is IMPORTED, so
    __file__ normally exists — the guard stays because the SUB-1 error was a NameError right here.
    """
    cands = []
    try:
        cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv"))
    except NameError:
        pass
    cands.append("/kaggle_simulations/agent/deck.csv")
    cands.append(os.path.join(os.getcwd(), "deck.csv"))
    seen, out = set(), []
    for p in cands:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _deck() -> list[int]:
    global MY_DECK_PATH
    for p in _deck_candidates():
        try:
            with open(p) as f:
                cards = [int(x) for x in f.read().split() if x.strip().isdigit()][:60]
        except OSError:
            continue
        MY_DECK_PATH = p
        return cards
    return []


MY_DECK = _deck()


def _payable(have, cost) -> bool:
    """Can `have` (list[EnergyType]) pay `cost` (list[EnergyType])?"""
    havec = Counter(int(e) for e in have)
    need_colorless = 0
    for c in cost:
        ci = int(c)
        if ci == _COLORLESS:
            need_colorless += 1
        elif havec.get(ci, 0) > 0:
            havec[ci] -= 1
        elif any(havec.get(w, 0) > 0 for w in _WILD):
            for w in _WILD:
                if havec.get(w, 0) > 0:
                    havec[w] -= 1
                    break
        else:
            return False
    return sum(havec.values()) >= need_colorless


def _can_pay_any(pkmn) -> bool:
    cd = CARDS.get(pkmn.id)
    if not cd:
        return False
    return any(_payable(pkmn.energies, ATTACKS[a].energies) for a in cd.attacks if a in ATTACKS)


def _can_pay_damaging(pkmn) -> bool:
    """`_can_pay_any` restricted to attacks that actually DEAL damage (PROMO_BANDS / Rule B).

    Target-free on purpose: it answers "is this body an attacker at all", which is what the
    promotion's +8000 band means. The target-aware question ("does its damage survive THIS
    defender") is `_best_payable_dmg(pk, opp) > 0`, already owned by WALL_PIVOT — keeping them
    separate stops one flag from silently acquiring the other's ex-immunity semantics.
    Called ONLY from `_promotion_value` under PROMO_BANDS; unreferenced when the flag is off."""
    cd = CARDS.get(pkmn.id)
    if not cd:
        return False
    return any(ATTACKS[a].damage > 0 and _payable(pkmn.energies, ATTACKS[a].energies)
               for a in cd.attacks if a in ATTACKS)


_EFFECT_CACHE: dict = {}

# Corrected damage model lives engine-free in core/damage.py (unit-tested locally), bundled by
# build_bundle.sh into the container. Prefer it; fall back to the inline logic if core isn't bundled
# (no-regression). Fixes the verified Resentful Refrain + megaEx mismodels (MANUAL_POSTMORTEM_82108349).
try:
    from core.damage import (attack_effect_bonus as _core_attack_effect,
                             dynamic_attack_bonus as _core_dynamic,
                             gives_multiprize as _core_multiprize,
                             prizes_for_ko as _core_prizes,
                             ability_damage_boost as _core_ability_boost,
                             bench_damage_prevented as _core_bench_prevent)
    _CORE_DAMAGE = True
except Exception:
    _CORE_DAMAGE = False

# Resistance reduction: cabt is the Scarlet&Violet / Mega-era TCG, where Resistance is a flat -30 applied
# when the DEFENDER's resistance type matches the ATTACKER's energy type. The engine applies it; the agent
# ignored it entirely (a deck-general correctness gap — matters whenever WE attack a resistor or face one,
# and for piloting other decks). The reduction VALUE is not exposed in CardData (only the type), so -30 is
# the SV-era standard, flagged for empirical confirmation against the live engine.
RESISTANCE_VALUE = 30


# DEAD_ATTACK's exemption: an attack whose TEXT carries non-damage value (Dwebble's Ascension =
# evolve accelerator; status/denial/search attacks) must stay rankable when 0-damage attacks are
# demoted. Conservative direction: falsely KEEPING an attack rankable costs one turn; falsely
# demoting one deletes a line.
_SIDE_VALUE_RE = re.compile(
    r"evolve|search your deck|attach|draw|heal|asleep|paralyz|confus|poison|burn"
    r"|discard an? [^.]{0,40}energy from your opponent|can.t attack|prevent"
    # adversarial review F9 (2026-08-01, measured on the real DB): 92 zero-damage attacks carry
    # real value the first regex missed — direct Knock Outs, "you win this game", prize-takers,
    # forced switches, counter placement, copy-attacks. False-exempt costs one turn; false-demote
    # deletes a win line, so additions stay broad.
    r"|knock(?:ed|s)? ?out|win(?:s)? this game|take [^.]{0,30}prize|switch"
    r"|damage counters? on|use it as this attack|discard the top"
    # F9 second pass (47 residuals re-measured on the DB): discard-pile recovery, hand reveal,
    # defensive buffs, deck-look utility.
    # 🔴 `does \d+ damage` was here and was REMOVED 2026-08-02 (adversarial audit). It exempted
    # attacks whose value IS damage — Raging Bolt's "Bellowing Thunder ... does 70 damage" carries
    # engine damage=0 because the damage is DYNAMIC — and the exemption is tested AFTER the
    # ex-immunity guard has already zeroed active_dmg. So a walled dynamic attacker converted
    # nothing and was still exempted from the demotion, which made DEAD_ATTACK inert on exactly
    # the frames it exists for (measured: 0 of 7 attackers demotable on the kanga_meowth deck).
    # An under-priced DAMAGE attack is a damage-parser gap (_attack_effect/_attack_dynamic), not a
    # side-value one; fixing it here bought inertness. This list stays NON-DAMAGE value only.
    r"|from your discard pile|reveals? their hand|less damage|look at the top",
    re.I)


def _attack_side_value(atk) -> bool:
    txt = (getattr(atk, "text", "") or "")
    return bool(txt and _SIDE_VALUE_RE.search(txt))


# F10 guard: an attack whose text pierces defender-side effects (Superb-Scissors-class wording)
# must not be zeroed-then-demoted by DEAD_ATTACK — the immunity zeroing above it is pre-existing
# always-on behavior and stays untouched; the flag just must not ESCALATE its blindness.
_PIERCE_RE = re.compile(r"isn.t affected by[^.]{0,80}(effects|anything) (?:on|done to)? ?your opponent", re.I)


def _attack_pierces_effects(atk) -> bool:
    txt = (getattr(atk, "text", "") or "")
    return bool(txt and _PIERCE_RE.search(txt))


def _attack_effect(atk):
    """B27 spread/ability parser: recover effective damage from attack TEXT that `damage` misses.
    Returns (active_extra, board_extra): active_extra can KO the ACTIVE (snipe/conditional/opp-hand),
    board_extra is bench/spread pressure. core/damage.attack_effect_bonus routes Resentful Refrain
    (50×opp-hand) to ACTIVE instead of bench spread; the inline body is the fallback."""
    aid = getattr(atk, "attackId", None)
    if aid in _EFFECT_CACHE:
        return _EFFECT_CACHE[aid]
    txt = (getattr(atk, "text", "") or "").lower()
    if _CORE_DAMAGE:
        res = _core_attack_effect(txt)
        _EFFECT_CACHE[aid] = res
        return res
    active_extra = board_extra = 0
    m = re.search(r"(\d+)\s+damage counters?\s+on[^.]*bench", txt)   # spread to bench
    if m:
        board_extra += int(m.group(1)) * 10
    m = re.search(r"does (\d+) damage to 1 of your opponent", txt)   # snipe
    if m:
        if "bench" in txt[m.start():m.start() + 70]:   # bench-snipe (Jetting Blow) -> bench, not active (false-lethal fix)
            board_extra += int(m.group(1))
        else:
            active_extra += int(m.group(1))
    m = re.search(r"(\d+) more damage", txt)                         # conditional bonus (count half)
    if m:
        active_extra += int(m.group(1)) // 2
    # opp-hand scaling (Resentful Refrain) is STATE-DEPENDENT — deferred to the decision site (uses the
    # real opp hand); a flat state-free estimate fabricated false lethals vs a non-hoarding opponent.
    if not re.search(r"(\d+)\s+damage for each card in your opponent.?s hand", txt):
        m = re.search(r"(\d+) damage for each", txt)                 # other (non-opp-hand) variable (×2)
        if m:
            board_extra += int(m.group(1)) * 2
    res = (active_extra, board_extra)
    _EFFECT_CACHE[aid] = res
    return res


def _attack_dynamic(atk, my_hand_n, opp_active, opp_hand_n=0) -> float:
    """B33: STATE-dependent attack damage from text. opp_hand_n = the TARGET controller's hand size
    (our attack: opponent's hand; opp burst: ours) — drives Resentful Refrain (50 × opp hand), which
    the shipped model was blind to. Prefers core/damage.dynamic_attack_bonus."""
    txt = (getattr(atk, "text", "") or "").lower()
    if _CORE_DAMAGE:
        oe = len(getattr(opp_active, "energies", []) or []) if opp_active else 0
        return _core_dynamic(txt, my_hand_n=my_hand_n, opp_hand_n=opp_hand_n, opp_energy_n=oe)
    extra = 0.0
    m = re.search(r"place (\d+) damage counters? on[^.]*for each card in your hand", txt)
    if m:
        extra += int(m.group(1)) * 10.0 * my_hand_n          # 2 counters/card -> 20 dmg/card
    m = re.search(r"(\d+)\s+damage for each card in your opponent.?s hand", txt)
    if m:
        extra += int(m.group(1)) * opp_hand_n                # Resentful Refrain (50 × opp hand)
    m = re.search(r"(\d+) more damage for each energy attached to your opponent", txt)
    if m and opp_active:
        extra += int(m.group(1)) * len(getattr(opp_active, "energies", []) or [])
    return extra


# REFRAIN_AWARE: value Resentful Refrain (Mega Froslass ex) = "50 damage FOR EACH CARD IN YOUR
# OPPONENT'S HAND" as 50 x REAL opp hand as ACTIVE (KO-capable) damage. The generic _attack_effect
# "(\d+) damage for each" regex mis-tags it as a flat board_extra += 50*2 = +100 bench chip — so the
# agent misvalues its OWN win-con (project_ptcg_resentful_refrain_misvalue; opp_hand_w unwired). Flag
# default OFF => inert/byte-behaviour-identical. Isolated froslass A/B pooled +0.053 (600/arm),
# consistently positive; also a correctness fix (frame-verified). See tests/test_refrain.py.
_REFRAIN_RE = re.compile(r"(\d+)\s+damage for each card in your opponent(?:['’])?s hand")
REFRAIN_AWARE = False


def _refrain_active_dmg(atk, opp):
    """(active_add, board_remove) for opponent-hand-scaling attacks. active_add = N x opp_hand;
    board_remove undoes _attack_effect's flat N*2 bench misvalue for the same text."""
    try:
        txt = (getattr(atk, "text", "") or "").lower()
        m = _REFRAIN_RE.search(txt)
        if not m:
            return (0.0, 0.0)
        n = int(m.group(1))
        oh = getattr(opp, "handCount", None)
        if oh is None:
            oh = len(getattr(opp, "hand", []) or [])
        return (float(n * (oh or 0)), float(n * 2))
    except Exception:
        return (0.0, 0.0)


# NEBULA_AWARE: attacks whose text says the damage "isn't affected by Weakness or Resistance"
# (Mega Starmie ex Nebula Beam = flat 210) must NOT get the weakness x2 the leaf applies
# unconditionally — else the agent over-values it vs Weak opponents (thinks 420 KO when it's 210)
# and commits a failed line. Flag default OFF => inert. NARROW exposure (only vs same-type-weak
# opponents, e.g. Water-weak Cinderace ~5% of meta) — correctness fix; validate with the right
# opponent in the mix before claiming a winrate (the froslass/alakazam/archaludon A/B has no
# Water-weak opponent, so it cannot measure this). Runtime uses a curly apostrophe (isn’t).
NEBULA_AWARE = False
_IGNORE_WEAK_RE = re.compile(r"isn['’]?t affected by weakness")


def _ignores_weakness(atk):
    try:
        return bool(_IGNORE_WEAK_RE.search((getattr(atk, "text", "") or "").lower()))
    except Exception:
        return False


def _strongest_attack(cd):
    atks = [ATTACKS[a] for a in cd.attacks if a in ATTACKS]
    return max(atks, key=lambda a: a.damage + sum(_attack_effect(a))) if atks else None


def _is_ex(pk) -> bool:
    """True if KOing this Pokémon yields the opponent MORE THAN ONE prize — a Pokémon ex OR a Mega
    Evolution Pokémon ex. The old check tested only `.ex`, misclassifying our own Mega Froslass/Starmie
    ex (megaEx = 3 prizes, api.py:477) as a 1-prize basic. Exact count: core/damage.prizes_for_ko."""
    cd = CARDS.get(pk.id) if pk else None
    if not cd:
        return False
    is_ex = bool(getattr(cd, "ex", False))
    is_mega = bool(getattr(cd, "megaEx", False))
    return _core_multiprize(is_ex=is_ex, is_mega_ex=is_mega) if _CORE_DAMAGE else (is_ex or is_mega)


def _prizes_for_ko(pk) -> int:
    """EXACT prize count the opponent takes for KOing this Pokémon (api.py:476-477): Mega ex = 3,
    plain ex = 2, else 1. The bool `_is_ex` answers "multiprize?"; this answers "how many?" — every
    prize-MAGNITUDE site must use this, not `2 if _is_ex else 1`, or a 3-prize Mega ex is undercounted."""
    cd = CARDS.get(pk.id) if pk else None
    if not cd:
        return 1
    is_ex = bool(getattr(cd, "ex", False))
    is_mega = bool(getattr(cd, "megaEx", False))
    if _CORE_DAMAGE:
        return _core_prizes(is_ex=is_ex, is_mega_ex=is_mega)
    return 3 if is_mega else (2 if is_ex else 1)


# ABILITY-AWARENESS (state-space audit #1): the engine exposes cd.skills (abilities), already loaded into
# CARDS, but the agent read them nowhere on the opponent. These gather a player's IN-PLAY ability texts and
# extract the threat-relevant effects (damage boosters, bench-damage prevention) so the threat envelope +
# bench valuation can finally see them. Engine-free (regex) with a core/ fallback.
_RE_AB_BOOST = re.compile(r"(?:do(?:es)?|deal[s]?)\s+(\d+)\s+more damage")  # boosters only (not "takes N more")


def _inplay_skill_texts(player) -> list:
    """All ability (skill) texts on a player's in-play Pokémon (active + bench)."""
    out = []
    for pk in (list(getattr(player, "active", None) or []) + list(getattr(player, "bench", None) or [])):
        if not pk:
            continue
        cd = CARDS.get(getattr(pk, "id", None))
        for sk in ((getattr(cd, "skills", None) or []) if cd else []):
            out.append(getattr(sk, "text", "") or "")
    return out


def _ability_boost(player) -> int:
    """Flat +N damage from the player's in-play boost abilities (Victini +10 / Hop's Snorlax +30) — added
    to the opponent's burst so the threat envelope isn't blind to ability-driven lethal."""
    texts = _inplay_skill_texts(player)
    if _CORE_DAMAGE:
        try:
            return _core_ability_boost(texts)
        except Exception:
            pass
    return sum(int(m.group(1)) for t in texts if (m := _RE_AB_BOOST.search((t or "").lower())))


def _bench_dmg_prevented(player) -> bool:
    """True if the player has an in-play ability that prevents bench damage (Shaymin Flower Curtain) — our
    bench-snipe (Mega Starmie ex Jetting Blow) is then worth ~0 on their bench."""
    texts = _inplay_skill_texts(player)
    if _CORE_DAMAGE:
        try:
            return _core_bench_prevent(texts)
        except Exception:
            pass
    return any(("prevent all damage" in (t or "").lower() and "bench" in (t or "").lower()) for t in texts)


# EX-IMMUNITY WALL (state-space audit; Crustle/Sylveon "Mysterious Rock Inn"): an in-play ability that
# prevents ALL damage from the opponent's Pokémon-ex attacks. UNMODELED before — the agent scored its own
# Crustle as fully vulnerable to the ex-heavy field (Alakazam ~39% + mirror ~18% + Mega Starmie ex),
# defaulting to a generic prize race instead of walling. Fires ONLY when OUR active holds the ability AND
# the opp attacker is an ex in the named scope (Crustle/Sylveon = any ex; Farigiraf = Basic ex only) =>
# byte-identical for froslass / any deck without such an active. Always-on correctness, like
# _bench_dmg_prevented (mirror). Tight text match (excludes Shaymin bench-prevention + trainer/stadium text).
def _ex_immune_vs(my_active, opp_active) -> bool:
    """True if my_active's in-play ability prevents ALL of opp_active's attack damage because opp_active is
    a Pokémon ex in the ability's named scope. Deliberately conservative: an over-loose match would make us
    think a vulnerable active is safe and get it KO'd, so require the exact 'prevent all damage ... {ex}'
    self-clause and a real ex attacker."""
    if not (my_active and opp_active) or not _is_ex(opp_active):
        return False
    cd = CARDS.get(getattr(my_active, "id", None))
    for sk in ((getattr(cd, "skills", None) or []) if cd else []):
        t = (getattr(sk, "text", "") or "").lower()
        if not ("prevent all damage" in t and "{ex}" in t and "this" in t and "bench" not in t):
            continue
        if "basic pok" in t:                       # Farigiraf: only a BASIC (non-Mega, non-evolved) ex is blocked
            ocd = CARDS.get(getattr(opp_active, "id", None))
            # NB: Mega-ex cards carry stage=None (not a numeric stage), so a `stage != 0` test misclassifies
            # a Mega ex as Basic — use megaEx/evolvesFrom, which are reliable, to identify a true Basic ex.
            is_basic = (ocd is not None and not getattr(ocd, "megaEx", False)
                        and not getattr(ocd, "evolvesFrom", None))
            if not is_basic:
                continue                           # Mega / evolved ex is NOT blocked by a Basic-only wall
        return True
    return False


# --- ATTACK-SCOPED Rock Inn (ROCK_INN_SCOPED; see the flag block) --------------------------------
# Three shapes, because the call sites ask three different questions and conflating them is exactly
# how the card-scoped predicate went wrong:
#   _wall_blocks_attack(holder, attacker, atk) — "does the wall stop THIS attack?"  (damage sites)
#   _wall_blocks_all(holder, attacker)         — "does the wall stop EVERY attack?" (early returns,
#                                                 bench/threat valuation, the wall_* rule guards)
#   _strongest_unwalled_attack(cd, holder, attacker) — the correct form of the _opp_can_ko guard:
#                                                 "strongest UNWALLED attack", not "is the strongest
#                                                 attack unwalled".
# All three short-circuit on the flag FIRST so the OFF path does no work the incumbent didn't do.
def _wall_blocks_attack(holder, attacker, atk) -> bool:
    """True if `holder`'s prevent-all-damage-from-ex ability stops `atk` (an attack of `attacker`)."""
    if not ROCK_INN_SCOPED:
        return _ex_immune_vs(holder, attacker)
    if not _ex_immune_vs(holder, attacker):
        return False
    # FIRES counters (ship_gate 5d): same style as leaf_v3/sym_dedupe — a dead flag on the ladder
    # must be measurable in DIAG, not invisible (the CARD_EFFECTS failure mode). ON branch only,
    # so the OFF path stays byte-identical with zero extra work. `rock_inn_scoped` counts scoped
    # evaluations against an ENGAGED wall; `rock_inn_pierced` counts the divergences (a
    # clause-carrying attack goes through where the card-scoped incumbent said "blocked").
    DIAG["rock_inn_scoped"] = DIAG.get("rock_inn_scoped", 0) + 1
    if _attack_pierces_effects(atk):
        DIAG["rock_inn_pierced"] = DIAG.get("rock_inn_pierced", 0) + 1
        return False
    return True


def _wall_blocks_all(holder, attacker) -> bool:
    """True if `holder`'s wall stops EVERY damaging attack `attacker` has — the only condition under
    which the card-scoped shortcut ("this Pokemon can do us no damage") is sound.

    Conservative on the damage test (`damage > 0`, matching `_has_unwalled_attacker`): a piercing
    attack with 0 printed damage leaves the incumbent's answer standing rather than resurrecting a
    threat we cannot size. All 11 clause-carrying attacks in the current DB print > 0, so the
    conservatism is currently inert; it only bounds a future rotation."""
    if not ROCK_INN_SCOPED:
        return _ex_immune_vs(holder, attacker)
    if not _ex_immune_vs(holder, attacker):
        return False
    DIAG["rock_inn_scoped"] = DIAG.get("rock_inn_scoped", 0) + 1   # FIRES counter (ship_gate 5d)
    cd = CARDS.get(getattr(attacker, "id", None))
    for a in ((getattr(cd, "attacks", None) or []) if cd else ()):
        atk = ATTACKS.get(a)
        if atk is not None and (getattr(atk, "damage", 0) or 0) > 0 and _attack_pierces_effects(atk):
            DIAG["rock_inn_pierced"] = DIAG.get("rock_inn_pierced", 0) + 1   # scoped != card-scoped
            return False
    return True


def _strongest_unwalled_attack(cd, holder, attacker):
    """`_strongest_attack` restricted to the attacks `holder`'s wall does NOT prevent.

    Verifier scope-correction #2: the naive guard ("is the strongest attack unwalled") happens to
    agree with this on all six clause-carrying ex cards TODAY — Mega Lopunny's Gale Thrust scores
    60+85=145 < Spiky Hopper 160, Mega Starmie's Jetting Blow 120+50=170 < Nebula Beam 210 — and
    diverges the moment a card's biggest attack is the walled one, which is a rotation away."""
    if not ROCK_INN_SCOPED:
        return _strongest_attack(cd)
    atks = [ATTACKS[a] for a in cd.attacks if a in ATTACKS]
    if _ex_immune_vs(holder, attacker):
        DIAG["rock_inn_scoped"] = DIAG.get("rock_inn_scoped", 0) + 1   # FIRES counter (ship_gate 5d)
        atks = [a for a in atks if _attack_pierces_effects(a)]
        if atks:
            DIAG["rock_inn_pierced"] = DIAG.get("rock_inn_pierced", 0) + 1   # a pierce survived the wall
    return max(atks, key=lambda a: a.damage + sum(_attack_effect(a))) if atks else None


def _has_unwalled_attacker(me, opp_active, exclude=None) -> bool:
    """True if we hold a Pokémon (active or bench, other than `exclude`) whose damage the opponent's
    active does NOT prevent and which has a damaging attack at all.

    Deliberately IGNORES current energy. This decides where the next energy should GO, so demanding
    an already-fuelled alternative would reproduce the exact defect that made wall_pivot inert: its
    guard required a pre-fuelled bench and fired in 0.9% of the frames it targeted (measured
    2026-08-06, 55 of 6,423 mirror frames), because the fuelled bench Pokémon is the ex attacker the
    wall blocks."""
    for pk in ((list(me.active[:1]) if getattr(me, "active", None) else [])
               + list(getattr(me, "bench", None) or [])):
        if pk is None:
            continue
        if exclude is not None and getattr(pk, "serial", None) == getattr(exclude, "serial", None):
            continue
        if _wall_blocks_all(opp_active, pk):     # flag-off == _ex_immune_vs (card-scoped)
            continue
        cd = CARDS.get(getattr(pk, "id", None))
        if not cd:
            continue
        for a in (getattr(cd, "attacks", None) or []):
            atk = ATTACKS.get(a)
            if atk is not None and getattr(atk, "damage", 0) > 0:
                return True
    return False


def _opp_can_ko(opp_active, my_active, attacker_blocked: bool = False) -> bool:
    """True if the opponent's active, if fueled, would KO my_active right now (weakness/resistance-aware).
    attacker_blocked=True (the opponent's active is Asleep/Paralyzed) => it can't attack next turn => no
    KO threat (audit #4: special conditions in the threat predicate)."""
    if attacker_blocked or not (opp_active and my_active):
        return False
    if _wall_blocks_all(my_active, opp_active):  # Rock Inn wall: our active takes 0 from opponent ex
        return False                            # attackers (ROCK_INN_SCOPED: unless an attack pierces)
    cd = CARDS.get(opp_active.id)
    if not cd:
        return False
    atk = _strongest_unwalled_attack(cd, my_active, opp_active)   # flag-off == _strongest_attack
    if not atk or atk.damage <= 0:
        return False
    dmg = atk.damage
    cdm = CARDS.get(my_active.id)
    if cdm and cdm.weakness is not None and cd.energyType == cdm.weakness:
        dmg *= 2
    if cdm and cdm.resistance is not None and cd.energyType == cdm.resistance:   # SV-era -30
        dmg = max(0, dmg - RESISTANCE_VALUE)
    return dmg >= (getattr(my_active, "hp", 0) or 0) and _payable(opp_active.energies, atk.energies)


# Evolution forward-map within MY_DECK: card NAME -> in-deck CardData that evolve from it.
# (engine `evolvesFrom` holds the predecessor's NAME, not its cardId).
_EVOLVES_TO: dict = {}
for _cid in set(MY_DECK):
    _cd = CARDS.get(_cid)
    _ef = getattr(_cd, "evolvesFrom", None) if _cd else None
    if _ef:
        _EVOLVES_TO.setdefault(_ef, []).append(_cd)


def reconfigure_deck(cards):
    """League hook: re-point the agent at a different 60-card deck and rebuild the deck-derived
    evolution map, so a candidate deck plays with tables matched to ITS cards (not froslass's)."""
    global MY_DECK, _EVOLVES_TO
    MY_DECK = [int(c) for c in cards][:60]
    _EVOLVES_TO = {}
    for cid in set(MY_DECK):
        cd = CARDS.get(cid)
        ef = getattr(cd, "evolvesFrom", None) if cd else None
        if ef:
            _EVOLVES_TO.setdefault(ef, []).append(cd)
    _compute_plan()


# --- PLAN: the deck's win-condition line (primary attacker + its in-deck evolution chain) --------
PLAN_LINE_IDS: set = set()     # card ids on the win-condition line (boosted by _score when PLAN on)
PLAN_ATTACKER_ID = None


def _compute_plan():
    """Identify the win-condition attacker in MY_DECK and the evolution chain that builds it."""
    global PLAN_LINE_IDS, PLAN_ATTACKER_ID
    PLAN_LINE_IDS, PLAN_ATTACKER_ID = set(), None
    pokes = [CARDS[c] for c in set(MY_DECK)
             if CARDS.get(c) and (getattr(CARDS[c], "hp", 0) or 0) > 0]
    if not pokes:
        return
    by_name = {cd.name: cd for cd in pokes}

    def val(cd):
        atk = _strongest_attack(cd)
        if not atk:
            return -1.0
        base = atk.damage + sum(_attack_effect(atk))
        if getattr(cd, "megaEx", False) or getattr(cd, "ex", False):
            base += 200.0          # 2-prize threats are the win-conditions
        return base

    # win-conditions = the CO-PRIMARY 2-prize attackers + their chains. Include an ex line only if its
    # attack value is within 50% of the top attacker's, so a DUAL-attacker deck (froslass: Mega Froslass
    # ex + Mega Starmie ex) keeps BOTH lines / its flexibility, while a focused combo deck (dragapult)
    # keeps ONLY its real attacker and drops low-value SUPPORT ex (Meowth/Latias) that diluted the plan.
    def atkval(cd):
        a = _strongest_attack(cd)
        return (a.damage + sum(_attack_effect(a))) if a else 0.0
    ex = [cd for cd in pokes if getattr(cd, "megaEx", False) or getattr(cd, "ex", False)]
    pool = ex or [max(pokes, key=val)]
    top = max((atkval(cd) for cd in pool), default=0.0) or 1.0
    wincons = [cd for cd in pool if atkval(cd) >= 0.5 * top]
    PLAN_ATTACKER_ID = max(wincons, key=val).cardId
    ids = set()
    for w in wincons:
        chain, cur = [w], w
        for _ in range(3):         # walk back to the basic via evolvesFrom (name-keyed)
            ef = getattr(cur, "evolvesFrom", None)
            if not ef or ef not in by_name:
                break
            cur = by_name[ef]
            chain.append(cur)
        ids |= {c.cardId for c in chain}
    PLAN_LINE_IDS = ids


_compute_plan()   # build the plan for the load-time MY_DECK (reconfigure_deck rebuilds it per deck)


def _line_cards(cd, seen=None):
    """cd plus all of its transitive in-deck evolutions."""
    if seen is None:
        seen = set()
    if cd is None or cd.cardId in seen:
        return []
    seen.add(cd.cardId)
    out = [cd]
    for nxt in _EVOLVES_TO.get(cd.name, []):
        out += _line_cards(nxt, seen)
    return out


def _line_top_cost(cd) -> int:
    """Max strongest-attack energy COUNT over this card AND its in-deck evolutions.

    Fixes the over-fill bug: a Basic whose own cheap attack is paid still needs MORE
    energy for its costlier evolution (Snorunt 1e -> Mega Froslass ex 3e). Mirrors the
    reference check_agent `_line_top_cost`; count-based (the engine validates types)."""
    best = 0
    for c in _line_cards(cd):
        atk = _strongest_attack(c)
        if atk:
            best = max(best, len(atk.energies))
    return best


def _can_pay_strongest(pkmn) -> bool:
    """True once the Pokémon is fully fueled for its evolution LINE's top attack
    (= stop fueling). Line-aware so we pre-load a Basic toward its evolution rather
    than stopping at its own cheap attack."""
    cd = CARDS.get(pkmn.id)
    if not cd:
        return False
    top = _line_top_cost(cd)
    return top > 0 and len(pkmn.energies) >= top


def _best_payable_dmg(pk, target) -> int:
    """Strongest damage `pk` could deal NEXT TURN with its CURRENT energy, weakness x2 /
    resistance -30 adjusted vs `target`. 0 if no damaging attack is payable. Feeds the
    promotion rule's KO-back band."""
    cd = CARDS.get(getattr(pk, "id", None))
    if not cd:
        return 0
    # EX-IMMUNITY (Rock Inn): the TARGET walls pk if it holds the ability and pk is an ex — 0 damage.
    # DRY guard for every _best_payable_dmg caller (gust-KO value, promotion KO-back + threat bands).
    if _wall_blocks_all(target, pk):
        return 0
    walled = ROCK_INN_SCOPED and _ex_immune_vs(target, pk)   # some (not all) attacks are prevented
    cdt = CARDS.get(target.id) if target is not None else None
    have = getattr(pk, "energies", []) or []
    best = 0
    for a in cd.attacks:
        atk = ATTACKS.get(a)
        if not atk or atk.damage <= 0 or not _payable(have, atk.energies):
            continue
        if walled and not _attack_pierces_effects(atk):
            continue
        dmg = atk.damage
        if cdt is not None and cdt.weakness is not None and cd.energyType == cdt.weakness:
            dmg *= 2
        if cdt is not None and cdt.resistance is not None and cd.energyType == cdt.resistance:
            dmg = max(0, dmg - RESISTANCE_VALUE)
        best = max(best, dmg)
    return int(best)


def _attach_enables_koback(pk, opp_active) -> bool:
    """LOOSE check for the doomed-active attach rule: could ONE more energy (color-blind) make any
    of `pk`'s attacks BOTH payable and lethal vs `opp_active` (weakness x2 / resistance -30)?
    Loose on purpose — when in doubt keep fueling the active (never block a possible win line)."""
    cd = CARDS.get(getattr(pk, "id", None))
    cdo = CARDS.get(getattr(opp_active, "id", None))
    if not cd or opp_active is None:
        return False
    # EX-IMMUNITY (Rock Inn): if THEIR active walls OUR attacker (their Rock Inn vs our ex), NONE of pk's
    # ex-attacks can KO it — so a KO-back is NOT possible (don't keep pumping a doomed active on a false hope).
    if _wall_blocks_all(opp_active, pk):
        return False
    walled = ROCK_INN_SCOPED and _ex_immune_vs(opp_active, pk)   # only the non-piercing attacks die
    ohp = getattr(opp_active, "hp", 0) or 0
    if ohp <= 0:
        return True                      # unknown target HP: assume lethal possible (conservative)
    have_n = len(getattr(pk, "energies", []) or [])
    for a in cd.attacks:
        atk = ATTACKS.get(a)
        if not atk or atk.damage <= 0 or len(atk.energies) > have_n + 1:
            continue                     # not payable even with one more energy (color-blind)
        if walled and not _attack_pierces_effects(atk):
            continue
        dmg = atk.damage
        if cdo is not None and cdo.weakness is not None and cd.energyType == cdo.weakness:
            dmg *= 2
        if cdo is not None and cdo.resistance is not None and cd.energyType == cdo.resistance:
            dmg = max(0, dmg - RESISTANCE_VALUE)
        if dmg >= ohp:
            return True
    return False


def _is_basic_pokemon(cd) -> bool:
    """Benchable body: a Pokémon card with no evolvesFrom (engine semantics for Basic)."""
    return (cd is not None and getattr(cd, "cardType", None) == CardType.POKEMON
            and not getattr(cd, "evolvesFrom", None))


def _hand_has_benchable_basic(me) -> bool:
    """True if OUR visible hand holds a Basic Pokémon. Unseen/absent hand -> False (conservative:
    the A1 emergency-fetch slice then still grabs a body — a spare Basic is never the wrong pick
    on a benchless board)."""
    for c in (getattr(me, "hand", None) or []):
        cid = getattr(c, "id", None) or getattr(c, "cardId", None)
        if _is_basic_pokemon(CARDS.get(cid)):
            return True
    return False


def _gust_target_value(tpk, my_active) -> float:
    """G1 GUST TARGETING (Class-A facts): rank OPPONENT pokemon we may drag active (Boss's
    Orders / gust effects). Exact arithmetic, constants not POLICY_W:
      +8000 + 500*prizes   our active KOs it THIS turn (weakness-aware payable damage) — take
                           the guaranteed prizes, biggest bounty first
      +3000                it cannot pay any attack (drag a passenger: free tempo wall for us)
      +0.5 * (missing hp)  softer targets tiebreak
      -1000                it KOs OUR active back next turn (never gust their killer in for free)
    The pubArchaludon loss anatomy: we never sniped bench Duraludons pre-evolution — an unfueled
    basic dragged active is both a KO candidate and a tempo freeze."""
    v = 50.0
    hp = getattr(tpk, "hp", 0) or 0
    cd = CARDS.get(getattr(tpk, "id", None))
    mx = (getattr(cd, "hp", 0) or 0) if cd else 0
    if my_active is not None and 0 < hp <= _best_payable_dmg(my_active, tpk):
        v += 8000.0 + 500.0 * _prizes_for_ko(tpk)
    payable = (getattr(tpk, "energies", None) is not None and _can_pay_any(tpk))
    if not payable:
        v += 3000.0
    if my_active is not None and _opp_can_ko(tpk, my_active):
        v -= 1000.0
    v += 0.5 * max(0, mx - hp)
    return v


# --- GUST_SNIPE_PRIORITY hook (2026-07-23, anti-Hariyama; flag + gate live on search_agent) ------
# search_agent owns the flag/weights (registry "S" keys = the coevo genome axis) and arms the
# per-decision cell below (seat, w_preevo, w_engine) ONLY on a confident Hariyama-family belief
# read; H stays belief-free. Cell None (the default, reset every decision) => the gust branch in
# _score returns the untouched _gust_target_value => byte-identical. Seat-scoped: inside search
# rollouts the SIMULATED OPPONENT's gust choices keep the exact-arithmetic bands (skewing the MIN
# node's model with OUR priorities would corrupt the worst-case read).
_GS_ARMED = [None]      # None | (root seat index, preevo bonus, engine bonus)
_GS_LAST: dict = {}     # id(option) -> applied bonus within the CURRENT _choose (armed only;
                        # read by _choose's changed-target DIAG — armed vs ACTUALLY-changed)
_ONEVOLVE_GUST_PRE: dict = {}    # normalized pre-evolve NAME -> [evolved CardData with an
_ONEVOLVE_GUST_TRIED = [False]   #   on-evolve gust skill] (card-DB-derived, no hardcoded names)


def _norm_card_name(s) -> str:
    """Apostrophe-normalized lowercase card name (the curly-quote lesson: engine text uses ’)."""
    return str(s).replace("’", "'").strip().lower()


def _onevolve_gust_pre() -> dict:
    """Card-DB map for the VERIFIED on-evolve gust mechanic (Hariyama 674 'Heave-Ho Catcher':
    'when you play this Pokémon from your hand to evolve 1 of your Pokémon … Switch in 1 of your
    opponent's Benched Pokémon to the Active Spot'): pre-evolve name -> its on-evolve-gust
    evolutions. Built lazily ONCE over the full DB (deck-general — any future card with the same
    text shape joins automatically). Empty on any error (levers stay inert)."""
    if not _ONEVOLVE_GUST_TRIED[0]:
        _ONEVOLVE_GUST_TRIED[0] = True
        try:
            for cd in CARDS.values():
                ef = getattr(cd, "evolvesFrom", None)
                if not ef:
                    continue
                for sk in (getattr(cd, "skills", None) or []):
                    txt = _norm_card_name(getattr(sk, "text", "") or "")
                    if ("evolve" in txt and "switch in 1 of your opponent" in txt
                            and "benched" in txt):
                        _ONEVOLVE_GUST_PRE.setdefault(_norm_card_name(ef), []).append(cd)
                        break
        except Exception:
            pass
    return _ONEVOLVE_GUST_PRE


def _gust_snipe_bonus(tpk, opp_side) -> float:
    """Additive archetype-conditional gust-target term (GUST_SNIPE_PRIORITY; called ONLY when the
    armed cell matches — see _GS_ARMED). Priority vs the Hariyama family (engine-verified intel,
    search_agent flag block): pre-evolve snipe (+w_preevo: kill Makuhita BEFORE the evolve = no
    more Heave-Ho pulls AND — Rock Inn verified — the wall goes invincible) > draw-engine piece
    (+w_engine: Lunatone's draw skill; Solrock via the named-partner coupling) > Hariyama itself
    (+0; energy-denial on the family is low value — Mega Lucario ex Aura Jab refuels from
    discard). Deck-general detectors, no hardcoded names. Never raises (any error => 0)."""
    ga = _GS_ARMED[0]
    if ga is None:
        return 0.0
    try:
        cd = CARDS.get(getattr(tpk, "id", None))
        if cd is None:
            return 0.0
        nm = _norm_card_name(getattr(cd, "name", ""))
        if nm in _onevolve_gust_pre():
            return float(ga[1])                              # pre-evolve of the gust evolver
        for sk in (getattr(cd, "skills", None) or []):       # draw-text skill (Lunar Cycle)
            t = (getattr(sk, "text", "") or "").lower()
            if "draw" in t and "card" in t:
                return float(ga[2])
        for pk in (opp_side.active + opp_side.bench):        # named-partner coupling (Solrock:
            if pk is None or pk is tpk:                      #  named in Lunatone's skill text)
                continue
            c2 = CARDS.get(getattr(pk, "id", None))
            if c2 is None:
                continue
            texts = [(getattr(sk, "text", "") or "") for sk in (getattr(c2, "skills", None) or [])]
            texts += [(getattr(ATTACKS.get(a), "text", "") or "") for a in (c2.attacks or [])
                      if ATTACKS.get(a) is not None]
            if any(nm in _norm_card_name(t) for t in texts if t):
                return float(ga[2])
        return 0.0
    except Exception:
        return 0.0


def _promotion_value(pk, opp_active, to_active: bool = True) -> float:
    """CLASS-A PROMOTION RULE (report/PILOTING_POLICY_CLASSES.md §3): rank candidates for a
    forced/effect promotion — SWITCH(3) post-KO survivor pick, TO_ACTIVE(4) promote,
    TO_FIELD(6) into play. Previously every candidate scored a FLAT 50.0/5.0, so the new
    active was chosen ARBITRARILY (the 'promote an unpowered attacker' bug).

    Lexicographic bands, deliberately CONSTANTS (not POLICY_W): a strictly-dominant ordering
    must not be invertible by a tuned profile. Sub-band terms are capped so they can never
    cross a band boundary (energies <=360 + evolve 150 + hp <=190 < the 1000 band gap):
      +8000            can pay a damaging attack NOW (fueled attacker)
      +4000            its strongest PAYABLE attack KOs their active back (weakness-aware)
      -1000 x prizes   (to_active only) they can KO it next turn -> exposes 1/2/3 prizes
      +120 x energies  fueling progress (capped at 3)
      +150             a deck card evolves from it (line potential)
      +0.5 x hp        survivability tiebreak
    PROMO_BANDS (Rule B, flag-off byte-identical) rewrites two of those bands in the to_active
    contexts only: the +150 line seed becomes PROMO_SEED_EPS (below the hp step, so it can no
    longer outrank a bigger body), and the +8000 fueled band requires a DAMAGING payable attack.
    See the flag block for the decision census behind both.
    TO_FIELD carve-out — MEASURED UNREACHABLE from ctx 6 (2026-08-10, _sc_ctx6_force/_test on the
    VPS): the only put-into-play chooser in the card DB is the Palafin line (106 Zero to Hero ->
    107), and forcing it in the live engine shows every real ctx-6 option is a DECK-area(1) ref
    with a select.deck payload; _resolve handles ACTIVE/BENCH/HAND/DISCARD only, so the ctx-6
    call site above never resolves a Pokemon and this function is never entered with
    to_active=False from a real frame (0 ctx-6 selects in 16k+ archived episodes, both seats;
    runtime coverage: 190k calls, all to_active=True). The to_active=False path is kept and
    suite-tested (synthetic exhaustive + 3 real forced frames, ON==OFF) as rotation insurance.
    BASE=3050 absorbs the worst-case -3000 penalty so every candidate scores >0: _choose's
    `s > 0` optional-pick filter then keeps the OLD always-accept behaviour on minCount=0
    switches — this fix changes the RANKING only, never whether we take the pick. Known
    narrow limit: the KO-back band is turn-parity-blind (assumes we attack next, true for
    the common post-KO promotion; an end-of-our-turn recoil promote mis-credits it)."""
    cd = CARDS.get(getattr(pk, "id", None))
    if not cd:
        return 5.0
    v = 3050.0
    # a TO_FIELD pick can resolve to a HAND card object (no .energies) — treat as unpowered,
    # never raise inside _score (the I5 crash class)
    fueled = getattr(pk, "energies", None) is not None and _can_pay_any(pk)
    # PROMO_BANDS driver (2): a payable 0-DAMAGE attack is not an attacker. Dwebble's Ascension
    # (478: 0 dmg, 1 colorless) hands a 1-energy Dwebble the +8000 band ahead of an unfueled
    # Crustle — and in exactly those menus we take the real body 0.0395 vs 0.4742 with an
    # unenergised Dwebble. Flag-off: `fueled` is already final above, so no extra call is made.
    # FIRES counters (ship_gate 5d, leaf_v3 style): `promo_bands` = Rule B's scope was reached at
    # all (flag ON, to_active promotion evaluated); `promo_bands_zero_dmg` = driver (2) actually
    # demoted a 0-damage "attacker". `_can_pay_damaging` is still called only when PROMO_BANDS and
    # to_active and fueled — the same predicate set as the single-line form this replaces.
    if PROMO_BANDS and to_active:
        DIAG["promo_bands"] = DIAG.get("promo_bands", 0) + 1
        if fueled and not _can_pay_damaging(pk):
            fueled = False
            DIAG["promo_bands_zero_dmg"] = DIAG.get("promo_bands_zero_dmg", 0) + 1
    # WALL-PIVOT second leg (review F12/F5, flag-gated): a candidate that is payable but WALLED
    # by the known defender (its damage into them is 0 — _best_payable_dmg's ex-immunity guard)
    # is NOT a fueled attacker for this promotion; without this, retreating walled Kanga #1 could
    # promote walled Kanga #2 straight back into the wall (real-card arithmetic: 11560 vs the
    # fueled Crustle's 11485). Flag-off byte-identical.
    if (fueled and WALL_PIVOT and to_active and opp_active is not None
            and _best_payable_dmg(pk, opp_active) <= 0):
        fueled = False
    if fueled:
        v += 8000.0
    if to_active and opp_active is not None:
        ohp = getattr(opp_active, "hp", 0) or 0
        if 0 < ohp <= _best_payable_dmg(pk, opp_active):
            v += 4000.0
        if _opp_can_ko(opp_active, pk):
            v -= 1000.0 * _prizes_for_ko(pk)
        elif (getattr(pk, "energies", None) is not None
              # PROMO_BANDS: mirror the fueled test above — "a FUELED candidate keeps its rank"
              # only means anything if both sites agree on what fueled IS. Flag-off this is the
              # literal `not _can_pay_any(pk)`, same call, same count.
              and not (_can_pay_damaging(pk) if PROMO_BANDS else _can_pay_any(pk))
              and 0 < (getattr(pk, "hp", 0) or 0) <= 2 * _best_payable_dmg(opp_active, pk)):
            # WALL-DENIAL (2026-07-06 pubArchaludon study): an UNFUELED multi-prize body promoted
            # into a wall that 2HKOs it is pure prize food (their plan: 220/turn into our 3-prize
            # Megas = 6 prizes in ~4 attacks while we chip a 300hp+Lab wall). Feed walls 1-prize
            # bodies; a FUELED candidate keeps its rank (it can fight while it lives).
            # -1000*(prizes-1): 1-prize candidates unaffected; a Mega drops -2000 — below a plain
            # basic's band, above the -3000 BASE absorption (band arithmetic unchanged).
            v -= 1000.0 * (_prizes_for_ko(pk) - 1)
    v += 120.0 * min(3, len(getattr(pk, "energies", []) or []))
    nm = getattr(cd, "name", None)
    if nm and nm in _EVOLVES_TO:
        # LINE-PROTECTION (grimm-aggro mining 2026-07-07, 929 pairs: LOSSES feed the line seeds
        # — Impidimp/Morgrem promoted into KOs, breaking the line AND donating the body; WINS
        # feed non-line bodies and keep the line developing on the bench). A line seed's +150
        # was unconditional, actively steering seeds INTO the feed. Now survival-conditional:
        # a seed promoted into a KO is the WORST equal-prize body to feed (-300). Sub-band
        # (<1000): prize-exposure ordering still dominates; this breaks equal-prize ties.
        if (to_active and opp_active is not None and _opp_can_ko(opp_active, pk)):
            v -= 300.0
        elif PROMO_BANDS and to_active:
            # PROMO_BANDS driver (1): demote the seed bonus BELOW the hp tiebreak on a promotion
            # (SWITCH/TO_ACTIVE). 1.0 < 0.5 * 10hp = the smallest hp step in the card DB, so a seed
            # still wins an EXACT-hp tie and never again outranks a bigger body (Dwebble 3235 >
            # Crustle 3125 / Mega 3200 -> 3086 < 3125 / 3200). TO_FIELD keeps the full +150: the
            # bonus exists to develop the line off the BENCH, which Rule B does not touch.
            # (Restructured from a ternary for the FIRES counter; values bit-identical both ways.)
            v += PROMO_SEED_EPS
            DIAG["promo_bands_seed"] = DIAG.get("promo_bands_seed", 0) + 1   # FIRES: driver (1) hit
        else:
            v += 150.0
    v += 0.5 * (getattr(pk, "hp", 0) or 0)
    return v


class _SetupShim:
    """PROMO_BANDS ctx-1 (SETUP_ACTIVE) adapter: a setup candidate is a HAND `Card` (id, serial,
    playerIndex — the engine dataclass carries NO hp and NO energies; cg/api.py:333), so scoring it
    with `_promotion_value` directly would zero the hp tiebreak and hand every menu to the seed.
    The shim carries the card DB's printed hp and an EMPTY energy list (a setup card genuinely has
    no attached energy — a 0-cost damaging attack still counts as fueled, same as on a promotion).
    Constructed ONLY under PROMO_BANDS at ctx 1; never mutates the shared obs (a `pk0.hp = ...`
    write onto the dataclass would leak into every later option pass over the same observation)."""
    __slots__ = ("id", "serial", "hp", "energies")

    def __init__(self, cid, serial, hp):
        self.id, self.serial, self.hp, self.energies = cid, serial, hp, []


def _resolve(obs, area, index, player_index):
    """Resolve an (area, index, playerIndex) option ref to a Pokemon/Card, or None."""
    st = obs.current
    if st is None or area is None or index is None:
        return None
    pl = st.players[player_index if player_index is not None else st.yourIndex]
    try:
        if area == AreaType.ACTIVE:
            return pl.active[index]
        if area == AreaType.BENCH:
            return pl.bench[index]
        if area == AreaType.HAND and pl.hand:
            return pl.hand[index]
        if area == AreaType.DISCARD:
            return pl.discard[index]
    except (IndexError, TypeError):
        return None
    return None


_SKILL_CACHE: dict = {}



# --- ENERGY_KERNEL (2026-07-26): read the fields the option ACTUALLY carries ------------------
# MEASURED DEFECT, not a new feature. The ENERGY/ENERGY_CARD/ATTACH branch below opens with
#     target = _resolve(obs, opt.inPlayArea, opt.inPlayIndex, ...) or my_active
# but ENERGY-cluster frames (contexts 21,22,23,26,28,30,31,32,33) hand us OptionType ENERGY(6) /
# ENERGY_CARD(5) options that populate area / index / energyIndex / playerIndex and leave
# inPlayArea and inPlayIndex as None — verified on live frames by
# scripts/probes/probe_energy_fields.py: 25 of 25 options across 9 frames carried area+index and
# ZERO carried inPlayArea. _resolve returns None when area is None (:1064), so `or my_active`
# fires and EVERY option in the frame resolves to the same body. The branch then returns the same
# number for all of them — a tie created by reading the wrong field, with no evaluator involved.
#
# That is why the ENERGY cluster measures tie_plain == tie_leafmax == 1.000: no leaf is ever
# consulted (non-MAIN frames never reach the search — _candidates returns None at
# search_agent.py:3371 and the frame goes to H._choose at :4147), so no leaf change could ever
# have fixed it. Measured on the same frames: resolving via area/index yields 1.44 distinct
# targets where the current arguments yield 1.00.
#
# ATTACH(8) is the one type in this branch that genuinely carries inPlayArea/inPlayIndex (2,874
# observations), so the dispatch is on type rather than a blanket swap — a blanket swap would
# break the MAIN attach path, which is the one part of this branch that currently works.
ENERGY_KERNEL = False


def _energy_opt_ref(obs, opt):
    """The body an ENERGY-family option refers to, reading the fields ITS type populates."""
    st = obs.current
    if st is None:
        return None
    if getattr(opt, "type", None) == OptionType.ATTACH:
        return _resolve(obs, getattr(opt, "inPlayArea", None),
                        getattr(opt, "inPlayIndex", None), st.yourIndex)
    return _resolve(obs, getattr(opt, "area", None), getattr(opt, "index", None),
                    getattr(opt, "playerIndex", None))


def _attach_prize_alt(obs, opt, opp_active, my_active) -> bool:
    """ATTACH_PRIZE (Rule A): does THIS energy select offer a STRICTLY better-prized body that the
    agent could fuel instead?

    True iff some other option in the same select resolves to a Pokemon that
      (1) is worth >= 2 prizes when KO'd (`_prizes_for_ko`, i.e. CardData.ex / megaEx), AND
      (2) is not already fuelled for its line's top attack (`_can_pay_strongest`) — the incumbent
          scores that option overfill(-20), so it is not a target we may steer toward, AND
      (3) is not walled by the opponent's active (`_wall_blocks_all`) — WALL_ATTACH's territory,
          and reading the same helper keeps ROCK_INN_SCOPED's attack-scoped answer in sync.

    Target resolution MIRRORS the ATTACH branch exactly, including the `or my_active` fallback, so
    the scan sees precisely the bodies `_score` would score. Called only under ATTACH_PRIZE; the
    byte-identical test asserts a zero call count on the flag-off pass."""
    st = obs.current
    if st is None:
        return False
    for o in (obs.select.option or []):
        if o is opt:
            continue
        if getattr(o, "type", None) not in (OptionType.ENERGY, OptionType.ENERGY_CARD,
                                            OptionType.ATTACH):
            continue
        alt = (_energy_opt_ref(obs, o) if ENERGY_KERNEL
               else _resolve(obs, getattr(o, "inPlayArea", None), getattr(o, "inPlayIndex", None),
                             st.yourIndex)) or my_active
        if alt is None or _prizes_for_ko(alt) < 2:
            continue
        if _can_pay_strongest(alt):
            continue
        if opp_active is not None and _wall_blocks_all(opp_active, alt):
            continue
        return True
    return False


# --- EVOLVE_REF_FIX (2026-07-26): recover the evolution CARD's identity ------------------------
# Second instance of the same defect class, different mechanism. At EVOLUTION-context frames
# (18,19,20,37,45) the EVOLVE branch's two differentiating bonuses — ESCAPE-EVOLVE and the
# PLAN-line bonus — both key on `opt.cardId`, and MEASURED at ctx=37 that field is absent on every
# option (5 of 5 carry area/inPlayArea/inPlayIndex/index, none carry cardId). So neither bonus can
# fire and every option returns the same POLICY_W["evolve"]: EVOLUTION ties 1.000 [0.91,1.00] at
# n=37 frames.
#
# Note the mechanism differs from the ENERGY case even though the symptom is identical. Here
# inPlayArea IS populated, but all options point at the SAME body — you are choosing WHICH
# evolution card to apply, not which body to apply it to (measured: 1.00 distinct target via
# inPlayArea vs 2.50 via area/index). The discriminator is the card in hand, so the fix recovers
# the card identity rather than the target.

# --- FUEL_SHORT (2026-07-26): reward an attach that COMPLETES a cost, not one that adds a body ---
# C1 "ATTACH completes cost (any target)" is our single largest measured blunder class: 57.2%
# against a ~1305-Elo teacher's 96.9% over 7,166 availabilities = 1.489 fixable blunders per game
# (report/our_compliance_full.log). Reading the attach branch, NOTHING rewards completing a cost —
# the only cost-related check is `_can_pay_strongest -> POLICY_W["overfill"]`, which is a PENALTY
# for over-filling.
#
# TEMPO looks like it covers this and does not: it adds 80 * (have + 1) / need, a pure COUNT of
# attached energies. It is colour-blind. Crustle's cost is {G} + 2 colorless, and our 13 energies
# are 8 colorless-providing (Mist, Spiky) against only 5 that satisfy the {G} slot (Grow Grass x4,
# Basic {G} x1) — so "one more energy" and "one step closer to attacking" are different facts, and
# TEMPO scores them identically.
#
# The deficit is returned DECOMPOSED, which is the whole trick. From an empty board a Mist and a
# Grow Grass both leave 2 units short, so any scalar shortfall ties them; as vectors they differ
# (1 colour + 1 colorless vs 0 colour + 2 colorless) and the scarce slot wins.
FUEL_SHORT = False
FUEL_COLOR_W = 6.0     # closing a COLOUR slot: 5 of 13 energies can
FUEL_CLESS_W = 2.0     # closing a colorless slot: 13 of 13 can
FUEL_CAP = 6.0         # provably below the smallest live adjacent band: TEMPO's step is
                       # 80/_line_top_cost = 26.7 on this deck, attach_active is 50. So this can
                       # only break ties WITHIN a band and can never reorder one.


def _color_short(have, cost):
    """(coloured units still missing, colorless units still missing) — same greedy as _payable."""
    havec = Counter(int(e) for e in have)
    need_colorless = missing_colour = 0
    for c in cost:
        ci = int(c)
        if ci == _COLORLESS:
            need_colorless += 1
        elif havec.get(ci, 0) > 0:
            havec[ci] -= 1
        elif any(havec.get(w, 0) > 0 for w in _WILD):
            for w in _WILD:
                if havec.get(w, 0) > 0:
                    havec[w] -= 1
                    break
        else:
            missing_colour += 1
    return missing_colour, max(0, need_colorless - sum(havec.values()))


# FUEL_ATTACH_ONLY (2026-07-27): third instance of the index-space defect class, and the first one
# inside a lever that had already been ESTABLISHED. _score's energy branch (:1624) covers ENERGY(6),
# ENERGY_CARD(5) and ATTACH(8), and calls _fuel_gain for ALL THREE with no dispatch — but this
# function reads me.hand[opt.index], and opt.index is only a HAND index for ATTACH. For ENERGY /
# ENERGY_CARD it is a BOARD slot, which is exactly why _energy_opt_ref above dispatches on type
# ("reading the fields ITS type populates"). So those two types scored a colour bonus off an
# unrelated card in our hand.
# MEASURED before the fix (scripts/probes/probe_fuel_indexspace.py, n=8 games, live config):
#   type 8 ATTACH      248451 calls, 0 fabricated       <- correct, index IS a hand index
#   type 6 ENERGY        7139 calls, 956 fabricated
#   type 5 ENERGY_CARD    126 calls,  25 fabricated
# 981 fabricated reads, mean +2.0 (cap 6.0) — and FUEL_CAP is deliberately sized to break ties
# within a band, so a fabricated value lands where it can decide. Blast radius is small (0.38% of
# calls) but the value is not merely noisy, it is meaningless.
# WHY THE McNEMAR MISSED IT: fuel_short was established on C1 frames, and C1 is cost-completing
# ATTACH — the ENERGY/ENERGY_CARD path was never in the validation set. tests/test_fuel_short.py
# exercises _color_short on hand-built vectors and never calls this function.
# A move/discard of existing energy is not an attach, so the right answer is to score no fuel gain
# for it at all rather than to read a different field.
FUEL_ATTACH_ONLY = False


def _attach_energy_type(obs, opt):
    """The EnergyType an ATTACH option would add, or None if it cannot be identified."""
    st = obs.current
    if st is None:
        return None
    try:
        me = st.players[st.yourIndex]
        # me.hand[opt.index] is only meaningful for ATTACH; see FUEL_ATTACH_ONLY above.
        if FUEL_ATTACH_ONLY and getattr(opt, "type", None) != OptionType.ATTACH:
            return None
        idx = getattr(opt, "index", None)
        if idx is None or not me.hand or idx >= len(me.hand):
            return None
        cd = CARDS.get(getattr(me.hand[idx], "id", me.hand[idx]))
        if cd is None or getattr(cd, "cardType", None) not in (CardType.BASIC_ENERGY,
                                                              CardType.SPECIAL_ENERGY):
            return None
        return getattr(cd, "energyType", None)
    except Exception:
        return None


def _fuel_gain(obs, opt, target):
    """Bounded reward for an attach that reduces the target's cost deficit, colour-aware."""
    et = _attach_energy_type(obs, opt)
    if et is None or target is None:
        return 0.0
    cd = CARDS.get(getattr(target, "id", None))
    atk = _strongest_attack(cd) if cd else None
    if not atk or not getattr(atk, "energies", None):
        return 0.0
    have = list(getattr(target, "energies", []) or [])
    c0, l0 = _color_short(have, atk.energies)
    c1, l1 = _color_short(have + [et], atk.energies)
    return min(FUEL_CAP, FUEL_COLOR_W * (c0 - c1) + FUEL_CLESS_W * (l0 - l1))


def _opt_card_id(obs, opt):
    """The cardId an option refers to, falling back to the card at (area, index) when absent."""
    cid = getattr(opt, "cardId", None)
    if cid is not None:
        return cid
    card = _resolve(obs, getattr(opt, "area", None), getattr(opt, "index", None),
                    getattr(opt, "playerIndex", None))
    return getattr(card, "id", None) if card is not None else None


EVOLVE_REF_FIX = False

# TO_HAND_NEED (2026-07-26): let _tutor_value price a TO_HAND fetch even when the card
# is already identified. See the block in _score's TO_HAND handler for the measurement.
TO_HAND_NEED = False


def _skill_value(cd) -> float:
    """B33: parse a card's skills/ability/trainer text into an approximate SETUP value (draw, search,
    energy-accel, heal, retreat-free). Replaces the flat constant so the agent values its engine."""
    if not cd:
        return 0.0
    key = (getattr(cd, "cardId", None), CARD_EFFECTS)   # flag in the key: a mid-process flip
    if key in _SKILL_CACHE:                              # must not return a stale score
        return _SKILL_CACHE[key]
    sk = getattr(cd, "skills", None)
    if not sk:
        return 0.0
    txt = " ".join((getattr(s, "text", "") or "") for s in sk).lower()
    v = 0.0
    m = re.search(r"draw (\d+) card", txt)
    if m:
        v += int(m.group(1)) * 22.0
    elif "draw" in txt and "card" in txt:
        v += 30.0
    if "search your deck" in txt:
        v += 45.0 if ("pokémon" in txt or "pokemon" in txt) else 25.0
    if re.search(r"attach .*energy", txt):
        v += 45.0
    if "heal" in txt:
        v += 20.0
    if "no retreat cost" in txt or "rare candy" in txt:
        v += 18.0
    if CARD_EFFECTS:
        # TYPED card-effect model (card_effects.json, built by scripts/build_card_effects.py).
        # MAX, never replace: the table can only RAISE a score, so a card the keyword scorer already
        # handled keeps its value and the POLICY_W band is not re-scaled — the fix targets the cards
        # that scored ZERO (measured: 70.6% of OUR OWN deck, incl. Boss's Orders / Hero's Cape /
        # Crustle's own ex-damage-prevention). Cards outside the table are unchanged.
        fx = _card_fx_entry(cd)
        if fx:
            v = max(v, float(fx.get("setup_value") or 0.0))
    _SKILL_CACHE[key] = v
    return v


# --- TYPED CARD EFFECTS (2026-07-25, memory project_ptcg_comprehension_gap) ------------------------
# WHY: _skill_value above is a regex KEYWORD scorer and returns 0.0 for 44 of the 84 effect-bearing
# cards in our deck + the real meta gauntlet — including the gust (Boss's Orders), the +100 HP tool
# (Hero's Cape) and OUR OWN win condition (Crustle's "prevent all damage from opponent's {ex}"). With
# those effects unknown we cannot certify dominance, compute hits-to-KO, or model anti-piloting; every
# lever needing them was inert for a reason no A/B could surface. This layer exposes the effects as
# TYPED data so those consumers become possible. Flag-off => byte-identical (loader never runs).
# DEFAULT ON (user directive 2026-07-25 "our best agent uses ALL functionality; better is better").
# Safe by construction: the table combines with max() so it can only RAISE a setup score, never lower
# one, and cards outside the table keep the keyword scorer. Set False only to reproduce a pre-07-25 run.
CARD_EFFECTS = True
_CARD_FX: dict = {}
_CARD_FX_TRIED = [False]


def _card_fx_load() -> dict:
    """Load card_effects.json once. Multi-candidate paths (the /kaggle_simulations flat-bundle
    lesson); any failure leaves the table empty so the agent silently keeps the keyword scorer."""
    if not _CARD_FX_TRIED[0]:
        _CARD_FX_TRIED[0] = True
        try:
            here = os.path.dirname(os.path.abspath(__file__))
        except NameError:                    # kaggle_environments execs the source without __file__
            here = os.getcwd()
        for p in ("card_effects.json", os.path.join(here, "card_effects.json"),
                  "src/agent/card_effects.json", "/kaggle_simulations/agent/card_effects.json"):
            try:
                with open(p) as f:
                    _CARD_FX.update((json.load(f) or {}).get("cards") or {})
                break
            except Exception:
                continue
    return _CARD_FX


def _card_fx_name(cd) -> str:
    """Apostrophe/case-normalised card name — MUST match build_card_effects.norm() (the curly-quote
    lesson: the DB ships 'Boss’s Orders', configs and notes write "Boss's Orders")."""
    return str(getattr(cd, "name", "") or "").replace("’", "'").replace("‘", "'").strip().lower()


def _card_fx_entry(cd):
    """Table row for this card, or None. Never raises."""
    try:
        return _card_fx_load().get(_card_fx_name(cd))
    except Exception:
        return None


def card_effects(cd) -> list:
    """PUBLIC accessor: the card's TYPED effects, e.g. [{"type":"gust","basic_only":True}] or
    [{"type":"hp_buff","hp":100}]. Empty list when unknown or the flag is off. This is the query
    surface for dominance predicates, a hits-to-KO value term, and anti-piloting rules."""
    if not CARD_EFFECTS:
        return []
    e = _card_fx_entry(cd)
    return list(e.get("effects") or []) if e else []


def card_has_effect(cd, etype: str) -> bool:
    return any(e.get("type") == etype for e in card_effects(cd))


def _option_card(obs, opt):
    """Resolve the CardData behind an ability/trainer/play option (tries hand + in-play refs)."""
    for area, idx, pl in (
        (getattr(opt, "area", None), getattr(opt, "index", None), getattr(opt, "playerIndex", None)),
        (getattr(opt, "inPlayArea", None), getattr(opt, "inPlayIndex", None), None),
    ):
        o = _resolve(obs, area, idx, pl)
        if o is not None and getattr(o, "id", None) is not None:
            return CARDS.get(o.id)
    return None


# --- GRIMM-v3 R5/R6 helpers (mirrored from sub_grimmsnarl_v3; only reached when the flags are ON,
# except _ability_energy_req which is pure/gated by its callers) ---------------------------------
_ABILITY_ENERGY_RE = re.compile(r"if this pok[eé]mon has (?:any )?\{(\w)\} energy attached")
_LETTER_ETYPE = {"G": int(EnergyType.GRASS), "R": int(EnergyType.FIRE),
                 "W": int(EnergyType.WATER), "L": int(EnergyType.LIGHTNING),
                 "P": int(EnergyType.PSYCHIC), "F": int(EnergyType.FIGHTING),
                 "D": int(EnergyType.DARKNESS), "M": int(EnergyType.METAL)}
_ABILITY_ENERGY_CACHE: dict = {}


def _ability_energy_req(cd):
    """EnergyType (int) this card's ability needs ATTACHED to function ('if this Pokémon has any
    {D} Energy attached' — Munkidori's Adrena-Brain), else None. Cached per cardId."""
    if cd is None:
        return None
    key = getattr(cd, "cardId", None)
    if key in _ABILITY_ENERGY_CACHE:
        return _ABILITY_ENERGY_CACHE[key]
    req = None
    for sk in (getattr(cd, "skills", None) or []):
        m = _ABILITY_ENERGY_RE.search((getattr(sk, "text", "") or "").lower())
        if m:
            req = _LETTER_ETYPE.get(m.group(1).upper())
            break
    _ABILITY_ENERGY_CACHE[key] = req
    return req


def _line_depth(cd) -> int:
    """R6: transitive in-deck evolution depth above cd (Impidimp->Morgrem->Grimmsnarl = 2)."""
    if cd is None:
        return 0
    d = 0
    for nxt in _EVOLVES_TO.get(cd.name, []):
        d = max(d, 1 + _line_depth(nxt))
    return d


# LOOKING_ZONE (2026-07-27): _deck_opt_card read ONLY obs.select.deck, so a LOOKING-area option —
# the "look at the top N cards" effects — never resolved and every one of them fell through
# _tutor_value's `cd is None` branch to a flat 8.0. MEASURED before writing this (n=10 games,
# scripts/probes/probe_tutor_buckets.py): of 405 tied ZONE_MOVE options, 30 were LOOKING-area and
# ALL 30 resolve cleanly through State.looking with real card names, while 120 PRIZE-area options
# resolve through nothing at all — prizes are face-down, so THAT tie is correct and is deliberately
# left alone. This flag closes the LOOKING half only.
LOOKING_ZONE = False


def _deck_opt_card(obs, opt):
    """R5: resolve a searched-card select option to CardData, or None.

    DECK area -> obs.select.deck (the engine reveals the searched deck; option.index indexes it).
    LOOKING area -> State.looking, same indexing, behind LOOKING_ZONE.
    PRIZE area is deliberately NOT handled: prize slots carry no usable id because prizes are
    face-down, so there is nothing to rank and the resulting tie is the correct answer.
    """
    try:
        area = getattr(opt, "area", None)
        idx = getattr(opt, "index", None)
        if area == AreaType.DECK:
            src = getattr(obs.select, "deck", None)
        elif LOOKING_ZONE and area == AreaType.LOOKING:
            src = getattr(getattr(obs, "current", None), "looking", None)
        else:
            return None
        if not src or idx is None or not (0 <= idx < len(src)):
            return None
        c = src[idx]
        return CARDS.get(getattr(c, "id", None)) if c is not None else None
    except Exception:
        return None


def _deck_cd_by_name(name):
    """CardData for an in-deck card by NAME (60-card scan; rare tutor frames only)."""
    for cid in set(MY_DECK):
        cd = CARDS.get(cid)
        if cd is not None and cd.name == name:
            return cd
    return None


# TUTOR_TIEBREAK (2026-07-27): _tutor_value is a COARSE BUCKETER — ~14 discrete constants and no
# ordering inside a bucket — so two genuinely different fetches score identically and the pick falls
# to list order. MEASURED (scripts/probes/probe_tutor_buckets.py, n=10 games): of the tied ZONE_MOVE
# frames, 37% are genuinely DIFFERENT cards sharing a bucket. Observed collisions: Dudunsparce vs
# Kadabra at 90.0 and 25.0, Abra vs Dunsparce at 15.0 and 45.0, and Grow Grass / Mist / Spiky Energy
# all at 20.0. (The other 63% are NOT defects — 39% is the same card offered several times, where
# the choice is a no-op, and 24% is face-down PRIZE picks, unrankable by the rules. Do not "fix"
# those.) This adds a bounded within-bucket ordering:
#   * ENERGY -> prefer the colour we are actually SHORT of, reusing _color_short, the same deficit
#     logic fuel_short is built on. That is the one energy question with a right answer.
#   * POKEMON -> prefer the bigger body (HP), a weak but non-arbitrary proxy, normalised to [0,1).
# TIEBREAK_CAP is 1.0 against a SMALLEST adjacent-bucket gap of 2.0 (8.0 vs 10.0), so this can order
# within a band and can NEVER promote a card across one — the same discipline as FUEL_CAP=6.0 sitting
# under TEMPO's 26.7 step. No win-rate is claimed: this makes the agent HAVE an opinion where it had
# none, which is not the same as the opinion being right.
TUTOR_TIEBREAK = False
TIEBREAK_CAP = 1.0


def _tutor_tiebreak(obs, cd) -> float:
    """Bounded [0, TIEBREAK_CAP) ordering INSIDE a _tutor_value bucket. Never crosses a bucket."""
    if not TUTOR_TIEBREAK or cd is None:
        return 0.0
    try:
        ct = getattr(cd, "cardType", None)
        if ct in (CardType.BASIC_ENERGY, CardType.SPECIAL_ENERGY):
            st = obs.current
            me = st.players[st.yourIndex]
            act = (me.active or [None])[0]
            tcd = CARDS.get(getattr(act, "id", None)) if act is not None else None
            atk = _strongest_attack(tcd) if tcd is not None else None
            if atk is None:
                return 0.0
            have = list(getattr(act, "energies", []) or [])
            cost = list(getattr(atk, "energies", []) or [])
            c0, _l0 = _color_short(have, cost)
            c1, _l1 = _color_short(have + [getattr(cd, "energyType", None)], cost)
            if c0 > c1:                       # this energy closes a COLOUR we are short of
                return TIEBREAK_CAP * 0.9
            return 0.0
        if ct == CardType.POKEMON:
            hp = float(getattr(cd, "hp", 0) or 0)
            return TIEBREAK_CAP * min(hp, 340.0) / 400.0     # bounded, strictly < cap
    except Exception:
        return 0.0
    return 0.0


def _tutor_value(obs, cd) -> float:
    """R5: need-based value of fetching `cd` off a deck search (TO_HAND). Line completion
    first (the NEXT missing evolution for a mon we already field), then the energy-gated-
    ability engine basic, then second-wave line starters / bench width, then staples.
    Mirrors tonakaiiii's measured tutor targets on the same 60 (report/grimm_v3/)."""
    if cd is None:
        return 8.0
    st = obs.current
    me = st.players[st.yourIndex]
    inplay = [pk for pk in ((me.active or []) + (me.bench or [])) if pk is not None]
    inplay_names = {CARDS[pk.id].name for pk in inplay if CARDS.get(pk.id)}
    hand_cds = [CARDS.get(c.id) for c in (me.hand or []) if c is not None]
    hand_names = {c.name for c in hand_cds if c}
    if cd.cardType == CardType.POKEMON:
        ef = getattr(cd, "evolvesFrom", None)
        if ef:
            if ef in inplay_names and cd.name not in hand_names:
                return 95.0 if getattr(cd, "stage2", False) else 90.0   # completes a fielded line
            if ef in hand_names and cd.name not in hand_names:
                return 55.0                                             # line piece one step ahead
            return 25.0
        n_same = sum(1 for pk in inplay if CARDS.get(pk.id) and CARDS[pk.id].name == cd.name)
        if _ability_energy_req(cd) is not None and n_same == 0:
            return 60.0                                                 # engine mon (Adrena-Brain)
        if _line_depth(cd) >= 2 and n_same < 2:
            return 65.0                                                 # second-wave line starter
        if len(inplay) <= 2:
            return 45.0                                                 # bench width
        return 15.0
    if cd.name == "Rare Candy":
        for hc in hand_cds:
            if hc is not None and getattr(hc, "stage2", False):
                s1 = _deck_cd_by_name(getattr(hc, "evolvesFrom", None) or "")
                base = getattr(s1, "evolvesFrom", None) if s1 is not None else None
                if base and base in inplay_names:
                    return 85.0                                         # candy line is LIVE
        return 22.0
    if cd.cardType == CardType.BASIC_ENERGY:
        return 10.0
    return 20.0


def _bench_tutor_value(obs, cd) -> float:
    """R5: Poffin-style search-to-BENCH pick order (all > 0 so we always fill the bench):
    deep-line starter (pre-stage the second Impidimp — tona benches Impidimp 60/83 Poffin
    puts) > engine basic > any evolvable > duplicate/vanilla."""
    if cd is None:
        return 50.0
    st = obs.current
    me = st.players[st.yourIndex]
    inplay = [pk for pk in ((me.active or []) + (me.bench or [])) if pk is not None]
    n_same = sum(1 for pk in inplay if CARDS.get(pk.id) and CARDS[pk.id].name == cd.name)
    d = _line_depth(cd)
    if d >= 2 and n_same < 2:
        return 78.0
    if _ability_energy_req(cd) is not None and n_same == 0:
        return 70.0
    if d >= 1 and n_same < 1:
        return 62.0
    return 55.0


def _setup_active_bonus(cd) -> float:
    """R6: opening-active preference on top of the flat 50.0."""
    d = _line_depth(cd)
    bonus = 12.0 if d >= 2 else (2.0 if d >= 1 else 0.0)
    if _ability_energy_req(cd) is not None:
        bonus = max(bonus, 6.0)
    return bonus


def _counter_placement_value(obs, opt) -> float:
    """B35 (dragapult Phantom Dive & any 'place damage counters' attack): place each counter to
    MAXIMIZE KOs/prizes instead of spraying them. A counter = 10 dmg; a target is KO'd when its
    remaining hp <= 10. The engine asks one counter at a time (ctx DAMAGE_COUNTER[_ANY], min1 max1
    over the legal targets), so we go greedy: take an immediate KO (weighted by prize value), else
    CONCENTRATE on the lowest-remaining-HP opponent target so the 6 counters compound into KOs
    rather than spreading thin; never counter our own Pokemon if avoidable."""
    st = obs.current
    tgt = _resolve(obs, getattr(opt, "area", None), getattr(opt, "index", None),
                   getattr(opt, "playerIndex", None))
    if tgt is None:
        tgt = _resolve(obs, getattr(opt, "inPlayArea", None), getattr(opt, "inPlayIndex", None),
                       getattr(opt, "playerIndex", None))
    hp = getattr(tgt, "hp", None) if tgt is not None else None
    if hp is None:
        return 1.0
    if getattr(opt, "playerIndex", None) == st.yourIndex:   # our own Pokemon — avoid self-damage
        return -100.0 - hp
    prize = float(_prizes_for_ko(tgt))   # 3 for Mega ex / 2 for ex / 1 else (api.py:476-477)
    if hp <= 10:                       # this counter scores the KO -> a prize
        return 1000.0 * prize
    return (200.0 - hp) + 25.0 * (prize - 1.0)   # concentrate on closest-to-KO; tilt to 2-prize ex


def _stage_from_prizes(me, opp) -> int:
    """Markov game stage from prizes remaining (fewer left = later): 0 early, 1 mid, 2 late.
    Local copy so the leaf policy stays stage-aware without importing search_agent (circular)."""
    try:
        left = min(len(me.prize), len(opp.prize))
    except Exception:
        return 0
    if left >= 5:
        return 0
    if left >= 3:
        return 1
    return 2


def _score(opt, obs) -> float:
    st = obs.current
    ctx = obs.select.context
    me = st.players[st.yourIndex]
    opp = st.players[1 - st.yourIndex]
    my_active = me.active[0] if me.active else None
    opp_active = opp.active[0] if opp.active else None
    t = opt.type
    pd = 0  # prize differential: >0 ahead, <0 behind (mining-derived SITU_AGGRO; prize=remaining)
    if SITU_AGGRO:
        try:
            pd = len(opp.prize) - len(me.prize)
        except Exception:
            pd = 0

    if t == OptionType.YES:
        if ctx == SelectContext.IS_FIRST:
            return 100.0 if GO_FIRST else -100.0
        if ctx == SelectContext.MULLIGAN:
            return -100.0
        return POLICY_W["yes_activate"]  # ACTIVATE / use-effect: usually beneficial (== shipped 20.0; now profile-tunable)
    if t == OptionType.NO:
        if ctx == SelectContext.IS_FIRST:
            return -100.0 if GO_FIRST else 100.0
        if ctx == SelectContext.MULLIGAN:
            return 100.0
        return -5.0

    if t == OptionType.ATTACK:
        # Attacking ENDS the turn, so a non-lethal attack must rank BELOW setup
        # actions (do all develop/attach/evolve first); only a KO jumps the queue.
        atk = ATTACKS.get(opt.attackId)
        if not atk:
            return 45.0
        dmg = atk.damage
        if opp_active and my_active:
            cd_opp = CARDS.get(opp_active.id)
            cd_me = CARDS.get(my_active.id)
            if (cd_opp and cd_me and cd_opp.weakness is not None and cd_me.energyType == cd_opp.weakness
                    and not (NEBULA_AWARE and _ignores_weakness(atk))):
                dmg *= 2
            if cd_opp and cd_me and cd_opp.resistance is not None and cd_me.energyType == cd_opp.resistance:
                dmg = max(0, dmg - RESISTANCE_VALUE)   # SV-era -30
        active_extra, board_extra = _attack_effect(atk)   # B27: spread/snipe/variable from text
        if board_extra and _bench_dmg_prevented(opp):      # audit #1b: Shaymin Flower Curtain etc. => our
            board_extra = 0                                # bench-snipe (Mega Starmie ex Jetting Blow) is dead
        # opp-hand scaling (Resentful Refrain = 50×opp hand) uses the REAL opp hand and is ALWAYS on
        # (accurate; decoupled from the mirage-prone COMPREHEND, which also gates _skill_value/SMART_PLACE).
        # own-hand scaling (Alakazam) stays behind COMPREHEND.
        # MERGE (agent-dev): the frontier always-on opp-hand path (default) and the docs-refactor
        # REFRAIN_AWARE path value the SAME feature — keep BOTH but make them MUTUALLY EXCLUSIVE (defer
        # the always-on opp_hand_n to 0 when REFRAIN_AWARE is on) so enabling the flag never double-counts.
        ohn = getattr(opp, "handCount", 0) or len(getattr(opp, "hand", []) or [])
        hn = (getattr(me, "handCount", 0) or len(getattr(me, "hand", []) or [])) if COMPREHEND else 0
        active_extra += _attack_dynamic(atk, hn, opp_active,
                                        opp_hand_n=(0 if REFRAIN_AWARE else ohn))  # own-hand if COMPREHEND
        if REFRAIN_AWARE and opp is not None:              # docs-refactor alt: Resentful Refrain 50×OPP hand
            _ra, _br = _refrain_active_dmg(atk, opp)
            if _ra > 0:
                active_extra += _ra
                board_extra = max(0.0, board_extra - _br)
        active_dmg = dmg + active_extra                    # what can KO the ACTIVE
        # EX-IMMUNITY (Rock Inn): if THEIR active walls OUR attacker (their Rock Inn vs our ex), this attack
        # does 0 to it — kill the phantom lethal + chip so we DON'T swing an ex into an immune wall (mirror:
        # Mega Kangaskhan ex into their Crustle = 0; attack with a non-ex or develop instead). board_extra
        # (bench spread) is left intact — their non-immune bench still takes it.
        # ROCK_INN_SCOPED: zero only the attacks the wall actually stops. This is the OTHER
        # direction of the fix and the one that matters for the froslass list (00df71b9be13), which
        # runs Mega Starmie ex: card-scoped, its Nebula Beam (210, clause-carrying) was zeroed into
        # a Crustle mirror and the real lethal was thrown away.
        if opp_active is not None and _wall_blocks_attack(opp_active, my_active, atk):
            active_dmg = 0.0
        if opp_active and active_dmg >= opp_active.hp:
            return 10000.0 + active_dmg   # lethal KO — take it now
        # DEAD-ATTACK (flag): converts NOTHING (0 post-guard active damage, no board damage, no
        # non-damage effect) yet ENDS our turn — rank below END(1.0) so develop/retreat/pass win
        # the frame. The wall census measured 333 such attacks in 85 games (wr 0.200 vs 0.502).
        # Review guards (08-01 adversarial hunt): only demote when an END option exists in THIS
        # select (F8: an all-non-positive select with minCount 0 must never return an empty pick),
        # and never demote an effect-piercing attack the immunity zeroing above cannot see (F10).
        if (DEAD_ATTACK and active_dmg <= 0 and board_extra <= 0
                and not _attack_side_value(atk) and not _attack_pierces_effects(atk)
                and any(getattr(o, "type", None) == OptionType.END
                        for o in (obs.select.option or []))):
            return DEAD_ATTACK_SCORE
        # chip ranking includes bench/spread pressure so the agent USES spread attacks
        chip = POLICY_W["attack_chip"] + (active_dmg + board_extra) * POLICY_W["attack_dmg"]
        if SITU_AGGRO and pd < 0:
            chip += 10.0 * (-pd)   # behind on prizes -> pressure (winners END less, attack more)
        if PATIENCE:
            # #1 pro divergence: when AHEAD or late, HOLD a NON-threatening chip (rank below END so the
            # agent develops / passes / protects the lead instead of chipping); a threatening (near-KO)
            # chip is always taken. oh==0 (no opp active) is treated as non-threatening, but a true
            # lethal already returned above (:lethal KO), so only harmless chips reach here.
            ahead = (len(opp.prize) - len(me.prize)) > 0
            oh = getattr(opp_active, "hp", 0) if opp_active else 0
            threatening = oh > 0 and active_dmg >= PATIENCE_THREAT_FRAC * oh
            if not threatening and (ahead or _stage_from_prizes(me, opp) >= 2):
                return PATIENCE_HOLD   # hold a non-threatening chip (ranks < END => develop / pass wins)
        return chip

    if t in (OptionType.ENERGY, OptionType.ENERGY_CARD, OptionType.ATTACH):
        target = (_energy_opt_ref(obs, opt) if ENERGY_KERNEL
                  else _resolve(obs, opt.inPlayArea, opt.inPlayIndex, st.yourIndex)) or my_active
        if target is None:
            return 10.0
        if _can_pay_strongest(target):
            return POLICY_W["overfill"]  # over-fill: target can already pay its strongest attack
        bonus = POLICY_W["attach_active"] if (my_active and target.serial == my_active.serial) else 0.0
        # DOOMED-ACTIVE ATTACH RULE (Class-A, report/LOSS_AUTOPSY_2026-07-04.md move-class #1: both
        # big fresh losses swung on pumping energy into an active that dies next turn — 7/10 deck
        # energies onto a 10hp mon facing a visible 250). If THEIR active KOs ours next turn
        # (weakness-aware, current-energy) AND this attach cannot enable a KO-BACK this turn AND a
        # bench target exists, the active is an energy SINK: flip the bonus to a rule-band penalty
        # so the bench attach (POLICY_W["attach"]) outranks it. Constants, not POLICY_W (a profile
        # must not re-invert it); KO-back check is deliberately LOOSE (color-blind +1 energy) so a
        # maybe-lethal attach keeps the bonus (conservative: never blocks a win line).
        sink = False   # doomed-active energy sink: never elevated by ATTACH_FIRST (see flag comment)
        if (bonus and opp_active is not None and me.bench
                and _opp_can_ko(opp_active, my_active, bool(getattr(opp, "asleep", False)
                                                            or getattr(opp, "paralyzed", False)))
                and not _attach_enables_koback(my_active, opp_active)):
            bonus = -150.0
            sink = True
        score = POLICY_W["attach"] + bonus
        if TEMPO:                         # concentrate fuel to bring an attacker ONLINE sooner
            cd = CARDS.get(target.id)
            atk = _strongest_attack(cd) if cd else None
            need = len(atk.energies) if atk else 0
            if need:
                have = len(getattr(target, "energies", []) or [])
                score += 80.0 * min(1.0, (have + 1) / need)   # reward finishing the closest attacker
        if PLAN and getattr(target, "id", None) in PLAN_LINE_IDS:
            score += 60.0                 # fuel the win-condition line first
        if ATTACH_FIRST and not sink:     # spend the free once-per-turn attach BEFORE any turn-ending
            score += ATTACH_FIRST_BAND    # attack; the lethal is still taken on the next option
        if FUEL_SHORT and not sink:       # colour-aware completion; TEMPO above is count-only
            score += _fuel_gain(obs, opt, target)
        # WALL-AWARE ATTACH (2026-08-06, measured). Our largest behavioural gap is wall_zero: of our
        # attacks INTO a "prevent all damage … {ex}" wall, 38.5% deal ZERO against strong players'
        # 1.7% on the SAME 60 cards — 23x, disjoint CIs. The cause is HERE, not in the retreat: the
        # once-per-turn attach goes onto the Mega Kangaskhan ex, whose damage that wall prevents, so
        # the energy buys damage that cannot land and wall_pivot then has no fuelled legal target.
        # Penalise only when an UNWALLED attacker exists to take the energy instead, so we can never
        # strand ourselves with nowhere to put it.
        if (WALL_ATTACH and opp_active is not None and target is not None
                and _wall_blocks_all(opp_active, target)   # a piercing attacker is NOT a dead end
                and _has_unwalled_attacker(me, opp_active, target)):
            score -= WALL_ATTACH_BAND
        # PRIZE-VALUE ATTACH (ATTACH_PRIZE, Rule A; see the flag block). POLICY_W is prize-blind
        # here — measured: our attach-to-active rate is 0.3966 with a 3-prize active and 0.3974 with
        # a 1-prize one (delta 0.0008) against a teacher delta of 0.191. Demote the 1-prize option
        # when a still-fuelable, non-walled >=2-prize body is on the same menu. Sits AFTER the
        # `_can_pay_strongest` early return, so POLICY_W["overfill"] (sign UNRESOLVED) is untouched;
        # withheld on the doomed-active `sink` so the Class-A rule is not double-counted.
        # FIRES counters (ship_gate 5d, leaf_v3 style): `attach_prize_checked` = an eligible
        # 1-prize attach was scanned at all; `attach_prize` = the penalty actually landed. Split
        # from the single `and` chain so the counters exist — call set and values are identical
        # (`_attach_prize_alt` still runs only when every earlier predicate passed; flag OFF makes
        # zero extra calls, which test_attachprize_byte_identical.py counts).
        if ATTACH_PRIZE and not sink and target is not None and _prizes_for_ko(target) < 2:
            DIAG["attach_prize_checked"] = DIAG.get("attach_prize_checked", 0) + 1
            if _attach_prize_alt(obs, opt, opp_active, my_active):
                score -= ATTACH_PRIZE_BAND
                DIAG["attach_prize"] = DIAG.get("attach_prize", 0) + 1
        return score

    if t == OptionType.EVOLVE:
        s = POLICY_W["evolve"] + (40.0 if (SITU_AGGRO and pd < 0) else 0.0)  # behind -> rebuild
        # ESCAPE-EVOLVE (Class-A facts; grimm-aggro mining 2026-07-07: aggro losses feed the
        # UNEVOLVED active line piece at ~t4 — but the line's own evolution IS the escape when
        # the evolved form leaves their KO range, e.g. Impidimp 70hp -> Morgrem 100 -> Grimmsnarl
        # 320 vs a 90-damage attacker). Evolution KEEPS damage, so the ghost HP = evolved max HP
        # minus damage already taken. Band 2000: above every develop band, below the pivot rule
        # (3000 — a fueled-bench pivot still outranks) and our own lethal (10000).
        if (my_active is not None and opp_active is not None
                and getattr(opt, "inPlayArea", None) == AreaType.ACTIVE
                and _opp_can_ko(opp_active, my_active, bool(getattr(opp, "asleep", False)
                                                            or getattr(opp, "paralyzed", False)))):
            evo = CARDS.get(_opt_card_id(obs, opt) if EVOLVE_REF_FIX
                            else getattr(opt, "cardId", None))
            cd_a = CARDS.get(getattr(my_active, "id", None))
            if evo is not None and cd_a is not None:
                dmg = max(0, (getattr(cd_a, "hp", 0) or 0) - (getattr(my_active, "hp", 0) or 0))
                ghost = type("Pk", (), {})()
                ghost.id = (_opt_card_id(obs, opt) if EVOLVE_REF_FIX
                            else getattr(opt, "cardId", None))
                ghost.hp = max(1, (getattr(evo, "hp", 0) or 0) - dmg)
                ghost.energies = list(getattr(my_active, "energies", []) or [])
                if not _opp_can_ko(opp_active, ghost):
                    return 2000.0
        if PLAN and (_opt_card_id(obs, opt) if EVOLVE_REF_FIX
                     else getattr(opt, "cardId", None)) in PLAN_LINE_IDS:
            s += 120.0                    # advance the win-condition line before off-plan develop
        if EVOLVE_TEMPO:
            # MEASURED SPEC (probe_evolve_hold.py, 55,253 Elo>=1050 frames): strong players' evolve
            # gate is PHASE — P(take|offered) ~0.70 early / ~0.40 mid / ~0.30 late, with the ACTIVE
            # held more than bench even early (0.50 vs 0.70); KO-threat is NEUTRAL (0.423 vs 0.429)
            # so it is deliberately NOT a condition here. Ours measured 0.928 flat — the single
            # largest behavioral deviation on record, and the Grookey pilot screen (2/12 on a deck
            # whose field EV is 0.57) showed its cost on evolution-tempo decks. This demotes the
            # evolve BAND by phase so evolves lose mid/late ties to develop actions instead of
            # auto-winning them — a curve toward the strong rates, never a blunt hold (the PATIENCE
            # refutation): the escape-evolve 2000 return above is untouched (that special IS
            # threat-conditioned and already measured Class-A), and early bench evolves keep the
            # full band. Phase via _stage_from_prizes (the PATIENCE precedent), approximating the
            # spec's turn buckets.
            stg = _stage_from_prizes(me, opp)
            if stg >= 2:
                s -= EVOLVE_TEMPO_LATE
            elif stg == 1:
                s -= EVOLVE_TEMPO_MID
            elif getattr(opt, "inPlayArea", None) == AreaType.ACTIVE:
                s -= EVOLVE_TEMPO_ACTIVE_EARLY
            DIAG["evolve_tempo"] = DIAG.get("evolve_tempo", 0) + 1   # FIRES counter (5d)
        return s
    if t == OptionType.ABILITY:
        # B33: value draw/search/accel abilities by what they DO (else flat — the blindness gap)
        return POLICY_W["ability"] + (_skill_value(_option_card(obs, opt)) if COMPREHEND else 0.0)
    if t == OptionType.PLAY:
        cdp = _option_card(obs, opt)
        # A1 EMERGENCY BENCH (Class-A; live 2026-07-06: 6/8 fresh SUB-32 losses end benchless-KO):
        # bench EMPTY + they can KO our lone active => a KO ends the game (no Pokémon in play), so
        # playing a Basic strictly dominates every develop action. Band 5000 (< lethal 10000).
        # Threat-conditioned like the kernel's BENCH_FORCE soundness rule; heuristic-side twin
        # covers heuristic pilots AND dominion's drained-budget fallback path.
        if (not me.bench and my_active is not None and opp_active is not None
                and _is_basic_pokemon(cdp)
                and _opp_can_ko(opp_active, my_active, bool(getattr(opp, "asleep", False)
                                                            or getattr(opp, "paralyzed", False)))):
            return 5000.0
        # B33: a search-2-Basics / draw-7 trainer ranks above a low-value item when COMPREHEND
        return POLICY_W["play"] + (_skill_value(cdp) if COMPREHEND else 0.0)
    if t == OptionType.RETREAT:
        # CLASS-A PIVOT RULE (report/LOSS_AUTOPSY_2026-07-04.md move-class #2): the active is
        # STUCK (cannot pay any attack) AND DOOMED (they KO it next turn) AND one attach cannot
        # make it lethal-capable, while a fueled bench attacker waits and retreat is offered
        # (offer == payable). Staying donates the body AND our whole attack turn; pivoting
        # strictly dominates. Rule band 3000: above every develop/attach/play band, below a
        # lethal of our own (10000). Couples with the doomed-attach demotion: not-savable ⇒
        # attach −150 and retreat 3000, so the argmax pivots instead of pumping the corpse.
        if (my_active and opp_active is not None and me.bench
                and not _can_pay_any(my_active)
                and _opp_can_ko(opp_active, my_active, bool(getattr(opp, "asleep", False)
                                                            or getattr(opp, "paralyzed", False)))
                and not _attach_enables_koback(my_active, opp_active)
                and any(_can_pay_any(b) for b in me.bench)):
            return 3000.0
        # WALL-PIVOT (flag): the active CAN pay but converts 0 into their active BECAUSE THE
        # DEFENDER WALLS IT (_ex_immune_vs — review F11: 'payable but 0' alone also matches our
        # own Ascension Dwebble, whose only attack has damage 0, and would retreat a pre-loaded
        # evolver instead of evolving; the explicit wall predicate restricts the rule to the
        # census surface and makes own-wall retreat structurally impossible, F13) while a FUELED
        # bench mon converts >0. Staying donates every attack turn (mirror census: zero streaks
        # to 14 turns, retreat offered in 92% of them). Band 900: above develop/attach so the
        # pivot survives the search's TOP_M prune; below CLASS-A(3000)/emergency(5000)/lethal.
        if (WALL_PIVOT and my_active is not None and opp_active is not None and me.bench
                and _wall_blocks_all(opp_active, my_active)   # don't retreat a piercing attacker
                and _can_pay_any(my_active)
                and _best_payable_dmg(my_active, opp_active) <= 0
                and any(_best_payable_dmg(b, opp_active) > 0 for b in me.bench)):
            return WALL_PIVOT_BAND
        # B13 lead-protection: save a 2-prize ex the opponent can KO next turn, when we are not
        # behind on prizes and a benched attacker is ready. A lethal of our own (10000) still
        # outranks this, so we race when we can KO their threat first.
        if (LEAD_PROTECT and my_active and _is_ex(my_active)
                and _opp_can_ko(opp_active, my_active,
                                bool(getattr(opp, "asleep", False) or getattr(opp, "paralyzed", False)))
                and (len(opp.prize) - len(me.prize)) >= 0
                and any(_can_pay_any(b) for b in me.bench)):
            return POLICY_W["protect_ex"]
        if my_active and not _can_pay_any(my_active) and any(_can_pay_any(b) for b in me.bench):
            return POLICY_W["retreat_stuck"]
        return -30.0
    if t == OptionType.END:
        return 1.0

    if t == OptionType.CARD:
        # B35: smart Phantom-Dive-style counter placement (KO-maximizing). Gated by COMPREHEND so
        # the shipped froslass agent is byte-identical; only the comprehension pilots get it.
        if COMPREHEND and SMART_PLACE and ctx in (SelectContext.DAMAGE_COUNTER,
                                                  SelectContext.DAMAGE_COUNTER_ANY):
            return _counter_placement_value(obs, opt)
        # CLASS-A PROMOTION RULE (report/PILOTING_POLICY_CLASSES.md §3): SWITCH(3) post-KO/effect
        # survivor pick, TO_ACTIVE(4) promote, TO_FIELD(6) into play. These previously scored a
        # FLAT 50.0 (TO_ACTIVE) / 5.0 (SWITCH/TO_FIELD fall-through), so the new active was chosen
        # ARBITRARILY — the 'promote an unpowered attacker' bug. Only ranks OUR OWN Pokémon; an
        # effect targeting the opponent's field keeps the old flat value.
        if ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.TO_FIELD):
            own = opt.playerIndex is None or opt.playerIndex == st.yourIndex
            pk = _resolve(obs, opt.area, opt.index, opt.playerIndex) if own else None
            if pk is not None and getattr(pk, "id", None) in CARDS:
                return _promotion_value(pk, opp_active, to_active=(ctx != SelectContext.TO_FIELD))
            if not own and ctx in (SelectContext.SWITCH, SelectContext.TO_ACTIVE):
                # G1 GUST TARGETING (Class-A facts; RESEARCH_TCG_AGENTS G1 — the top-10 recipe is
                # prize-math-gated Boss targeting; previously flat 50 = ARBITRARY gust target):
                tpk = _resolve(obs, opt.area, opt.index, opt.playerIndex)
                if tpk is not None and getattr(tpk, "id", None) in CARDS:
                    gv = _gust_target_value(tpk, my_active)
                    ga = _GS_ARMED[0]                        # GUST_SNIPE_PRIORITY (armed by
                    if ga is not None and ga[0] == st.yourIndex:   # search_agent; None=identical)
                        gb = _gust_snipe_bonus(tpk, opp)
                        if gb:
                            _GS_LAST[id(opt)] = gb
                            return gv + gb
                    return gv
            return 50.0 if ctx == SelectContext.TO_ACTIVE else 5.0
        if ctx == SelectContext.HEAL:
            own = opt.playerIndex is None or opt.playerIndex == st.yourIndex
            pk = _resolve(obs, opt.area, opt.index, opt.playerIndex) if own else None
            cdh = CARDS.get(getattr(pk, "id", None)) if pk is not None else None
            if cdh is not None:   # OUR Pokémon only — an opp-heal effect keeps the flat value
                mx = getattr(pk, "maxHp", None) or (getattr(cdh, "hp", 0) or 0)  # board maxHp sees tools
                dmg_taken = max(0, mx - (getattr(pk, "hp", 0) or 0))
                return 20.0 + dmg_taken * (2.0 if pk is my_active else 1.0)  # heal the damaged ACTIVE first
            return 5.0
        if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.SETUP_BENCH_POKEMON,
                   SelectContext.TO_BENCH):
            # Rule 5d requires every ON flag to demonstrably FIRE at runtime, and these two had
            # no counter at all -- a smoke run reported setup_priority 0 / deck_tutor 0, which
            # meant "no such key", not "never fired". Unverifiable is how a flag becomes inert
            # without anyone noticing (the SUB-28 class), so count the firings.
            #
            # PROMO_BANDS ctx-1 extension (Rule B scope gap, 2026-08-10). The Rule B census family
            # is {SETUP_ACTIVE(1), SWITCH(3), TO_ACTIVE(4)} and the flag covered 3 of 4: ctx 1 is
            # 587 decisions = 12% of the family, ALL in the mega_only cell — a setup active must be
            # a BASIC, so Crustle can never appear on this menu, the only real body is the Mega
            # (also a Basic), and we sit 0.4787 there (teachers 46/46 = 1.0000 [0.9229, 1.0]).
            # Score the setup pick with the SAME band ladder as a promotion (to_active=True): via
            # _SetupShim the hand card gets its printed hp and an empty energy list, so the fueled/
            # KO bands are inert for every normal setup card and the pick reduces to seed-eps +
            # 0.5*hp — Mega 3200 over Dwebble 3086 / Shaymin 3090, Rule B's whole point.
            # SCOPED TO THE MEASURED CELL: the ladder applies only when THIS menu offers a
            # >=2-prize body. Measured 2026-08-10 (_sc_promobands_ctx1.py on 996 ladder ctx-1
            # frames): unscoped, the hp tiebreak also flipped 60 Dwebble->Shaymin picks on the 112
            # no-Mega menus, where teachers prefer the SEED 9/12 — an unmeasured, teacher-opposed
            # side effect; with the guard the no-Mega cell is untouched (0 changes). On the live
            # deck (basics = 344/343/756 only) the guard IS the census family. Precedence: this
            # branch sits ABOVE SETUP_PRIORITY (R6, never shipped) deliberately — R6's +12/+2
            # line-starter bonus is bigger than the 5.0 minimum hp step and would re-invert the
            # band; when both flags are armed Rule B owns the guarded menus (R6 keeps the rest).
            # Flag OFF => branch not entered, zero extra work, incumbent path byte-identical.
            if PROMO_BANDS and ctx == SelectContext.SETUP_ACTIVE_POKEMON:
                big0 = False
                for o2 in (obs.select.option or []):
                    c2 = _resolve(obs, getattr(o2, "area", None), getattr(o2, "index", None),
                                  getattr(o2, "playerIndex", None))
                    if c2 is not None and _prizes_for_ko(c2) >= 2:
                        big0 = True
                        break
                pk0 = _resolve(obs, opt.area, opt.index, opt.playerIndex) if big0 else None
                cd0 = CARDS.get(getattr(pk0, "id", None)) if pk0 is not None else None
                if cd0 is not None:
                    DIAG["promo_bands_setup"] = DIAG.get("promo_bands_setup", 0) + 1   # FIRES (5d)
                    return _promotion_value(
                        _SetupShim(pk0.id, getattr(pk0, "serial", pk0.id),
                                   getattr(pk0, "hp", None) or (getattr(cd0, "hp", 0) or 0)),
                        opp_active, to_active=True)
            if SETUP_PRIORITY and ctx == SelectContext.SETUP_ACTIVE_POKEMON:
                c0 = _resolve(obs, opt.area, opt.index, opt.playerIndex)
                cd0 = CARDS.get(c0.id) if c0 else None
                if cd0 is not None:
                    b0 = _setup_active_bonus(cd0)             # R6
                    DIAG["setup_priority"] = DIAG.get("setup_priority", 0) + 1
                    if b0:
                        DIAG["setup_priority_nonflat"] = DIAG.get("setup_priority_nonflat", 0) + 1
                    return 50.0 + b0
            if DECK_TUTOR and ctx == SelectContext.TO_BENCH:
                cd0 = _deck_opt_card(obs, opt)
                if cd0 is not None:
                    v0 = _bench_tutor_value(obs, cd0)         # R5 (Poffin -> bench)
                    DIAG["deck_tutor"] = DIAG.get("deck_tutor", 0) + 1
                    if v0 != 50.0:
                        DIAG["deck_tutor_nonflat"] = DIAG.get("deck_tutor_nonflat", 0) + 1
                    return v0
            return 50.0
        card = _resolve(obs, opt.area, opt.index, opt.playerIndex)
        cd = CARDS.get(card.id) if card else None
        if ctx in (SelectContext.DISCARD, SelectContext.TO_DECK, SelectContext.TO_DECK_BOTTOM):
            # pitch Basic Energy first, protect Pokemon
            if cd and cd.cardType == CardType.BASIC_ENERGY:
                return 30.0
            if cd and cd.cardType == CardType.POKEMON:
                return -10.0
            return 5.0
        if ctx in (SelectContext.TO_HAND, SelectContext.EVOLVES_FROM, SelectContext.EVOLVES_TO):
            # A1 EMERGENCY FETCH (Class-A; the ep-84204485 t1 game-loser): benchless with NO Basic
            # in hand, and this fetch offers one -> take the body. Previously Staryu and Mega
            # Starmie ex both scored the flat 30.0 and the tie-break took the unbenchable
            # evolution; the lone active was KO'd next turn (benchless auto-loss). TO_HAND is
            # heuristic-routed even under dominion (KERNEL_SPECIFIC), so this closes the live gap.
            if ctx == SelectContext.TO_HAND:
                cdf = cd if cd is not None else _deck_opt_card(obs, opt)
                own = opt.playerIndex is None or opt.playerIndex == st.yourIndex
                if (own and not me.bench and _is_basic_pokemon(cdf)
                        and not _hand_has_benchable_basic(me)):
                    return 5000.0
            # TO_HAND_NEED: use the need-based valuation whenever the card can be identified,
            # not only when it CANNOT. _tutor_value is documented as "need-based value of fetching
            # `cd` off a deck search (TO_HAND); line completion..." — it is the right function for
            # this frame, and gating it on `cd is None` meant that KNOWING which card we were
            # fetching made the agent value it LESS precisely. With cd resolvable the branch
            # collapses to a two-value scale (30.0 for any Pokemon with attacks, 8.0 for everything
            # else), so every attacking Pokemon on offer ties and the pick falls to list order.
            # MEASURED: ZONE_MOVE ties on 0.531 [0.49,0.57] of 610 frames per 30 games, ~320 blind
            # decisions — the largest tie mass outside the searched TURN cluster.
            if DECK_TUTOR and (TO_HAND_NEED or cd is None):
                cd2 = cd if (TO_HAND_NEED and cd is not None) else _deck_opt_card(obs, opt)
                if cd2 is not None:
                    # + a bounded within-bucket ordering so two different fetches sharing a bucket
                    # are not decided by list order (see TUTOR_TIEBREAK).
                    return _tutor_value(obs, cd2) + _tutor_tiebreak(obs, cd2)   # R5 (deck-tutor)
            if cd and cd.cardType == CardType.POKEMON and cd.attacks:
                return 30.0
            return 8.0
        return 5.0

    if t == OptionType.NUMBER:
        nm = opt.number or 0
        # COUNT contexts where MORE is strictly better: draw as many as possible (DRAW_COUNT=38);
        # place as many damage counters as allowed (DAMAGE_COUNTER_COUNT=39 — spread/Phantom-Dive);
        # remove/move as many as allowed (REMOVE_DAMAGE_COUNTER_COUNT=40 — Munkidori Adrena-Brain /
        # heal). SIGN-BUG FIX (port of pokemon-tcg-strategy 14eb91f, report/SELECTION_AUDIT.md): only
        # DRAW_COUNT was whitelisted, so ctx 39/40 fell to -nm and always placed/moved 1 when 3 were
        # legal (confirmed 100% of 302 census frames; capped the Grimmsnarl Adrena-Brain engine +
        # mis-modelled it in the opp node). Conservative WHITELIST (not a default flip): unknown COUNT
        # contexts keep the -nm cost default.
        if ctx in (SelectContext.DRAW_COUNT,
                   SelectContext.DAMAGE_COUNTER_COUNT,
                   SelectContext.REMOVE_DAMAGE_COUNTER_COUNT):
            return float(nm)         # beneficial count -> take the maximum
        return float(-nm)            # otherwise prefer the smallest count (cost default)

    return 1.0


DIAG = {"calls": 0, "choose_err": 0, "obs_err": 0,
        "ctx": Counter(), "chosen_type": Counter(), "main_chosen": Counter()}


# --- LEARNED POLICY (gated, byte-identical when off) --------------------------------------------
# At a MAIN single-select >=2-option decision, score options with the learned blend (anchor*z(_score)
# + blend*top-player type/card nudge) instead of raw _score. ANCHOR keeps the heuristic's hard logic
# (lethal KO at 10000); BLEND adds the top-player proactivity correction (retreat/end less). LEARNED_
# POLICY=False => the line below is unchanged => byte-identical submission. learned_pilot is pure-python.
LEARNED_POLICY = False
try:
    import learned_pilot as _LP  # noqa: E402
except Exception:
    _LP = None


def _choose(obs) -> list[int]:
    sel = obs.select
    opts = sel.option
    n = len(opts)
    DIAG.setdefault("ctx", Counter())[int(sel.context)] += 1  # setdefault: survive a DIAG.clear() in harnesses
    if n == 0:
        return []
    minc = max(0, min(sel.minCount, n))
    maxc = max(minc, min(sel.maxCount, n))
    if maxc == 0:
        return []
    if _GS_ARMED[0] is not None:            # GUST_SNIPE_PRIORITY: fresh per-select bonus trace
        _GS_LAST.clear()                    # (unarmed: never touched — zero overhead, byte-identical)
    if (LEARNED_POLICY and _LP is not None and int(sel.context) == int(SelectContext.MAIN)
            and maxc == 1 and n >= 2):
        try:
            _ls = _LP.score_options(obs, _score, _LP.load_weights())
            scored = sorted(((_ls[i], i) for i in range(n)), key=lambda z: z[0], reverse=True)
        except Exception:
            scored = sorted(((_score(opts[i], obs), i) for i in range(n)), key=lambda z: z[0], reverse=True)
    else:
        scored = sorted(((_score(opts[i], obs), i) for i in range(n)), key=lambda z: z[0], reverse=True)
    if _GS_ARMED[0] is not None and _GS_LAST and maxc == 1 and scored:
        # GUST_SNIPE DIAG (instrumentation directive: count ACTUAL target changes, not armed
        # calls): re-rank this select with the recorded bonuses subtracted (same stable-sort tie
        # semantics: ascending index within equal scores) and compare winners. Pure counters —
        # the CHOSEN move below is untouched by this block.
        DIAG["gust_snipe_sel"] = DIAG.get("gust_snipe_sel", 0) + 1
        _s_by_i = {i: s for s, i in scored}
        _base = sorted(((_s_by_i[i] - _GS_LAST.get(id(opts[i]), 0.0), i) for i in range(n)),
                       key=lambda z: z[0], reverse=True)
        if _base and _base[0][1] != scored[0][1]:
            DIAG["gust_snipe_changed_target"] = DIAG.get("gust_snipe_changed_target", 0) + 1
    chosen: list[int] = []
    for s, i in scored:
        if len(chosen) >= maxc:
            break
        if s > 0 or len(chosen) < minc:
            chosen.append(i)
    if len(chosen) < minc:
        for _, i in scored:
            if i not in chosen:
                chosen.append(i)
            if len(chosen) >= minc:
                break
    if chosen:
        ct = int(opts[chosen[0]].type)
        DIAG.setdefault("chosen_type", Counter())[ct] += 1
        if int(sel.context) == int(SelectContext.MAIN):
            DIAG.setdefault("main_chosen", Counter())[ct] += 1
    return chosen[:maxc]


def _legal_fallback(obs_dict) -> list[int]:
    try:
        sel = obs_dict.get("select") or {}
        n = len(sel.get("option") or [])
        return list(range(max(0, min(int(sel.get("minCount", 0) or 0), n))))
    except Exception:
        return []


def agent(obs_dict) -> list[int]:
    DIAG["calls"] += 1
    # deck-selection / game-over: return the 60-card deck WITHOUT needing the engine
    # parse (must never depend on to_observation_class succeeding here).
    if not isinstance(obs_dict, dict) or obs_dict.get("select") is None:
        return MY_DECK
    try:
        obs = to_observation_class(obs_dict)
        if obs.select is None:
            return MY_DECK
        try:
            return _choose(obs)
        except Exception:
            DIAG["choose_err"] += 1
            return _legal_fallback(obs_dict)
    except Exception:
        DIAG["obs_err"] += 1
        return _legal_fallback(obs_dict)
