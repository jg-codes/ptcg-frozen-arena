"""L3 search agent (MVP) — our-turn rollout lookahead over the engine world-model.

At a MAIN decision we fork the state via the sanctioned `search_begin` API, try each of
the heuristic's top-M candidate first-moves, roll the rest of OUR turn forward with the
heuristic policy, and pick the candidate whose end-of-turn board VALUE is best. Everything
else (setup, forced, multi-select) falls through to the heuristic.

Robust by construction: time-budgeted via obs['remainingOverageTime']; ANY error or low
budget → the plain heuristic. Never worse-than-heuristic on legality.

Reuses heuristic_agent (H) for deck, option scoring, and the rollout policy.
"""
from __future__ import annotations

import json
import math as _mth
import os
import random
import re
import time

from cg import api  # type: ignore
from cg.api import OptionType, SelectContext, to_observation_class  # type: ignore

import heuristic_agent as H  # deck, _score, _choose, _legal_fallback, CARDS, MY_DECK

# ---- PAIRING GUARD (2026-07-02): this module calls H._opp_can_ko(opp_act, my_act, blocked) with
# 3 args at 4 MAIN-frame sites (~869/1084/1185/1706). Paired with a 2-arg heuristic copy, every
# such call raised TypeError inside the leaf, agent()'s blanket except swallowed it, and the agent
# silently played H._legal_fallback (option[0]) on ~85% of searched MAIN frames -- measured
# winrates through that pair were ARTIFACTS (report/GRIMMSNARL_PILOT.md before-columns). Fail
# LOUD at import time instead of crashing silently at decision time.
import inspect as _pg_inspect
_pg_sig = _pg_inspect.signature(H._opp_can_ko)
if len(_pg_sig.parameters) < 3:
    raise ImportError(
        "PAIRING GUARD: search_agent calls 3-arg H._opp_can_ko but the paired heuristic module "
        "%r defines _opp_can_ko%s. Align the heuristic copy (3rd arg with default) or pair the "
        "matching lineage." % (getattr(H, "__file__", "?"), _pg_sig))
if not hasattr(H, "RESISTANCE_VALUE"):
    # Same mixed-lineage attribute skew, next attribute over: three leaf sites price resistance via
    # bare H.RESISTANCE_VALUE; a heuristic copy without it AttributeErrors into the same silent
    # _legal_fallback path at decision time (true_screen 2026-07-02: opp arms FALLBACK_EXC=2).
    raise ImportError(
        "PAIRING GUARD: search_agent prices resistance via H.RESISTANCE_VALUE but the paired "
        "heuristic module %r does not define it. Add RESISTANCE_VALUE = 30 (SV-era -30) to the "
        "heuristic copy or pair the matching lineage." % (getattr(H, "__file__", "?"),))
FALLBACK_EXC = 0          # count of exceptions swallowed into the agent() _legal_fallback path
LAST_FALLBACK_EXC = None  # repr() of the last swallowed exception (harness/test introspection)
LAST_FALLBACK_TB = None   # last two traceback frames — WHERE it was raised, not just what
FALLBACK_BUG = 0          # subset of FALLBACK_EXC that are PROGRAMMING errors (see _BUG_EXC)

# The blanket handler at the bottom of agent() cannot be removed: an exception that escapes loses
# the match outright, so catching everything is correct. What was wrong is that it treated a
# NameError from a bad edit and a transient engine refusal as the same event, and silently returned
# option[0] for both. That is how a brain-pair skew once swallowed ~85% of searched MAIN frames
# (the 3-arg _opp_can_ko class) with nothing but a counter nobody read.
#
# So: still catch everything, still return a legal move, but SEPARATE the two populations. These
# types are only ever OUR bug — the engine cannot cause them — and any nonzero FALLBACK_BUG means
# the agent is running degraded and a ship must be blocked.
_BUG_EXC = (NameError, AttributeError, TypeError, KeyError, IndexError,
            UnboundLocalError, ZeroDivisionError, AssertionError)
import dtrace as trace  # lightweight decision tracing (renamed from `trace` to avoid the
# stdlib `trace` collision flagged in review); no-op unless the harness enables it.
try:
    import belief  # L2 opponent-deck posterior (particle filter over mined meta decks)
except Exception:
    belief = None
try:
    import self_model as _SM  # SelfModel (OwnPilot): deck-general win-condition + Resentful (flat stdlib)
except Exception:
    _SM = None
try:
    import matchup_role  # the generic "Who's the Beatdown?" role layer (engine-free)
except Exception:
    matchup_role = None
try:
    import features  # rich state features for the LEARNED eval
except Exception:
    features = None
try:
    import value_leaf as _VL  # learned P(win) leaf (AUC 0.729) — makes deep search productive
except Exception:
    _VL = None
try:
    # LEAF_V3's coefficient table (SOURCE LITERALS, stdlib-only — see leaf_fe.py's docstring for why
    # it is not a data file). Guarded like every other sibling import rather than hard: an ImportError
    # here would propagate out of `import search_agent`, and main.py:437-439 turns ANY raise in the
    # config applier into the bare POLICY_W heuristic with NO SEARCH for the whole match. Inert beats
    # catastrophic. The three places that make "inert" impossible to ship unnoticed are:
    #   1. build_bundle.sh:30-32 — leaf_fe.py is on the pick() loop, which ABORTS the build if missing
    #      (NOT the `[ -f ] && cp` loop below it, which is exactly how opponent_adapt shipped dead);
    #   2. scripts/test_config_contract.py — refuses a ship config that arms leaf_v3 without the file;
    #   3. DIAG["leaf_v3"] / DIAG["leaf_v3_dead"] — a FIRES counter, so a dead term is measurable.
    import leaf_fe as _LEAF_FE
except Exception:
    _LEAF_FE = None
try:
    # SIB_RANKER's coefficient table + feature extraction (SOURCE LITERALS, stdlib-only — the
    # leaf_fe pattern; see rank_prior.py's docstring). Guarded for the identical reason leaf_fe's
    # import is: an ImportError here would propagate out of `import search_agent`, and main.py's
    # applier turns ANY raise into the bare POLICY_W heuristic with NO SEARCH for the whole match.
    # Inert beats catastrophic; the three loud backstops are the same three as leaf_fe's:
    #   1. build_bundle.sh ships rank_prior.py on the ABORT-on-missing pick() loop;
    #   2. scripts/test_config_contract.py refuses a config arming sib_ranker without the file;
    #   3. DIAG["sib_ranker"] / DIAG["sib_ranker_dead"] — FIRES counters from day one, so a dead
    #      flag on the ladder is measurable (the CARD_EFFECTS failure mode).
    import rank_prior as _RANK_PRIOR
except Exception:
    _RANK_PRIOR = None

TOP_M = 3              # candidate first-moves to search
ROLL_CAP = 120         # max plies in a rollout (raised for the 2-ply horizon)
MIN_BUDGET_S = 2.0     # if remainingOverageTime below this, skip search
PER_MOVE_S = 0.4       # wall-clock cap per decision's search (the FLAT legacy cap; superseded by
                       # the TIME_AWARE adaptive deadline below when TIME_AWARE=True)
# --- TIME-AWARE budgeting -------------------------------------------------------------------
# The flat PER_MOVE_S=0.4 cap is never hit (decisions finish in ~3 ms) so the per-episode overage
# pool (remainingOverageTime, ~600 s shared over ~26 MAIN decisions ≈ ~23 s/decision) goes unspent.
# TIME_AWARE replaces the flat cap with an adaptive per-decision deadline and a round-robin
# sampling loop that keeps drawing rollouts until that deadline — so deeper search (ROLL_TO_END
# Monte-Carlo / belief determinization) actually spends the budget. The deadline is held WELL under
# the per-decision overage share by HARD_WALL_S, so cumulative spend keeps a large margin under the
# pool and we never drift into timeout territory.
TIME_AWARE = False         # master switch: adaptive deadline + sample-to-budget loop
# BUDGET RETUNE (#41): HARD_WALL_S is now a TRUE safety ceiling held ABOVE the per-decision share
# (12 s, not 2.5) so the share term — BUDGET_RESERVE × pool / est_SEARCHABLE_decisions — is what
# actually binds, letting deep search spend the pool (target ~300 s / 600). est_remaining counts
# only the ~prizes_left×3 SEARCHABLE MAIN decisions, not all ~88 raw actions, so the share is real.
HARD_WALL_S = 12.0         # hard per-decision wall (SAFETY ceiling): never exceed; held ABOVE the share
BUDGET_RESERVE = 0.5       # spend only this fraction of the per-decision overage share
DECISIONS_PER_PRIZE = 3    # est. remaining SEARCHABLE OUR decisions ≈ our prizes-left × this
EST_DECISIONS_MIN = 8      # floor on est-remaining-searchable-decisions (caps the per-decision share)
SAMPLE_MIN_PER_CAND = 1    # guarantee ≥ this many samples/candidate even past the deadline
POOL_FLOOR_S = 25.0        # SPEEDRUN floor: when remaining pool < this, bank time (min samples, stop)
STALL_TURN = 40            # SPEEDRUN: a game past this turn count with no prize progress is a stall
STALL_PRIZES_LEFT = 5      # ...and both sides still have ≥ this many prizes left = stall (bank time)
STAKES_LO = 0.4            # _stakes_factor floor (forced/decided turns spend least)
STAKES_HI = 2.0            # _stakes_factor ceiling (near-lethal/branchy/contested spend most)
ROLL_2PLY = False      # if True, roll one OPPONENT reply too and value after it
                       # (penalises moves that lose to the counter — risk-state logic)
ROLL_TO_END = False    # DEEP MONTE-CARLO: roll BOTH sides (heuristic) to game TERMINAL and return
                       # the ACTUAL win/loss — bypasses the coarse leaf eval (the proven bottleneck).
ROLL_END_CAP = 600     # ply cap for a full-game rollout (both turns to terminal)
DETERMINIZATIONS = 1   # for 2-ply: # of sampled opponent hands to AVERAGE over (belief).
                       # 1 = single crude prediction (high variance); K>1 = determinized search.
EARLY_STOP_MARGIN = None  # convergence early-stop: if the leader beats the rest by > this
                          # margin, stop searching and bank time (None = off; pays off in ISMCTS).
RICH_EVAL = False         # richer leaf eval: small potentials (energy tempo / board / hand) BELOW
                          # the HP scale to break the 50% value-ties (review #4). A/B before shipping.
BELIEF = False            # if True, predict the OPPONENT's deck from the L2 belief posterior (a real
                          # meta deck consistent with what we've seen) instead of a self-copy of ours.
# W3 OPPONENT MODEL (2026-07-22, report/PLAN_TOP_ELO.md W3): every depth/belief/k_worlds field test
# was refuted against a SELF-COPY opponent — `_predict` filled the opponent's hidden zones with
# MY_DECK (or, under BELIEF, with overlapping prefix-slices of a raw canonical 60-list that could
# re-spawn cards the opponent has visibly used up). OPP_DECK_MODEL is the consistent fix, DECK
# CONTENTS ONLY: when ON and the belief posterior is a real inference (non-fallback, confidence >=
# OPP_DECK_MODEL_MIN_CONF), each determinized world fills the opponent's UNSEEN zones
# (deck/prize/hand) by sampling WITHOUT replacement from the posterior particle's decklist MINUS
# the opponent's revealed cards (in-play + attached + discard) — a world can never contradict what
# we've observed. Falls back to the existing BELIEF/self-copy path when belief is unconfident, the
# library is missing, or the pool can't cover the hidden-zone sizes. The opponent MOVE POLICY in
# rollouts/MIN-nodes stays OUR heuristic (H._choose) — policy replacement is a separate future lever.
OPP_DECK_MODEL = False
OPP_DECK_MODEL_MIN_CONF = 0.15   # posterior-confidence floor (mirrors the DENY_EVO_MIN_CONF convention)
# BELIEF_VARIANTS (2026-07-23, count-granular list posterior): upgrade the OPP_DECK_MODEL world
# sampler from the modal-particle library to a WEIGHTED POSTERIOR over REAL observed lists
# (deck_variants.json — in-band + encounter-weighted, scripts/mine_deck_variants.py) with a per-card
# HYPERGEOMETRIC count likelihood: "3 seen of a 4-of" narrows to the lists that run 4 (C(4,3)=4 vs
# C(3,3)=1), and the revealed multiset v2 also subtracts PRE-EVOLUTIONS consumed under evolved
# Pokémon. COMPOSES with OPP_DECK_MODEL (only fires inside its branch): when both are ON,
# belief.sample_world(use_variants=True) draws the particle from belief.variant_posterior and falls
# through to the legacy particle path whenever variants can't produce a world (artifact missing,
# fallback/unconfident posterior, short pool) — never weaker than OPP_DECK_MODEL alone. Flag-off
# byte-identical; a stale belief.py without the kwarg raises TypeError into the existing except →
# legacy path, never an error.
BELIEF_VARIANTS = False
BELIEF_VARIANTS_MIN_CONF = 0.15  # variants FAMILY-posterior confidence floor (OPP_DECK_MODEL_MIN_CONF convention)
# THE DOMINION (#42): true minimax over the belief posterior. RECURSE_DEPTH=0 = shipped 1-ply,
# BYTE-IDENTICAL output. d=1 = roll OUR turn, then enumerate the opponent's TOP_Mo H-scored replies
# and take the MIN value from OUR seat (a competent opponent best-responds) — the cheap high-value
# upgrade over greedy ROLL_2PLY. d=2 = then our MAX counter; depths alternate MAX(us)/MIN(opp).
# Each MAIN decision averages the minimaxed value over K_WORLDS belief determinizations (PIMC/ISMCTS).
RECURSE_DEPTH = 0         # 0 = shipped 1-ply (byte-identical); 1 = MIN-over-opp-replies; 2 = +our MAX
TOP_Mo = 2               # opponent reply branching at MIN nodes (H-scored top-Mo pruner)
K_WORLDS = 6             # belief WORLDS averaged per main decision (the K-world expectation loop)
LEARNED_COND = False     # matchup-CONDITIONAL learned blend: at the root, set learned_pilot.BLEND from the
                         # belief archetype (HIGH vs aggro lucario/iono, 0 vs grindy dunsparce/trevenant).
# RICH DECISION TRACING (#3): structured per-decision JSONL. TRACE_LEVEL 0 = off (ZERO runtime cost,
# every trace call guarded). ≥1 logs the decision record (candidates+values, opp MIN-reply line,
# belief guess, chosen move+margin+reason, time, rollouts) AND counts/logs caught rollout errors.
TRACE_LEVEL = 0
STAGE_EVAL = False        # Markov-PHASE-aware leaf eval (early/mid/late) + lethal-range/KO encoding —
                          # the win-condition-aware value fn that breaks the 50% ties (SOTA #1 lever).
LEARNED_EVAL = False      # use the offline-FIT logistic-regression value (learned_eval.json) — the
                          # data picks the weighting over many features instead of hand-tuning.
LEARNED_VALUE = False     # use value_leaf (P(win) AUC 0.729, bc_value.py) as the search leaf — the
                          # strong calibrated value that makes DEEP search productive (vs the blind leaf).
STAGE_W = {               # per-stage [early, mid, late] weights — Optuna-tunable over the pool (#14).
    "tempo": [70.0, 50.0, 25.0],   # fuel the active attacker
    "board": [22.0, 12.0, 5.0],    # board development (benched threats)
    "ko": [15.0, 40.0, 80.0],      # lethal-range / closing
    "hand": 5.0,                   # card advantage (stage-independent)
}

# OPP_THREAT (the alakazam fix): the WHY traces show froslass's ONLY losing matchup is alakazam
# (0.50 wr), where we LOSE 42% of games we'd valued as winning (best_v>=2000) — the value fn is blind
# to the opponent's BURST KO, especially hand-scaling (Powerful Hand = 20×their hand), which the static
# _opp_can_ko misses. This term discounts our value by the prizes we'd concede when the opponent's
# active can KO our active NEXT turn (static + dynamic damage, weakness-aware) — so search protects /
# retreats the threatened attacker instead of over-committing into the comeback. FLAG-OFF by default
# (byte-identical); tunable weight; validate on the FIELD (alakazam matchup + aggregate) before trust.
OPP_THREAT = False
OPP_THREAT_W = 700.0       # < one prize (1000): discounts an exposed lead, never overrides the race

# --- CLOSING_EXCHANGE (T2 FAILED-TO-CLOSE fix; RACE_LEAF-lineage endgame leaf term) ---------------
# Verified failure shape (report/MISPLAY_REVIEW.md T2, 26/118 live losses, 13 at match point): in the
# closing phase the agent takes the greedy attack, leaves a low-HP active exposed with retreat legal,
# and the opponent's RETURN (promote a second attacker / gust) closes the game — e.g. ep 82698289:
# led 5-3, KO'd Mega Lucario, left our 40/310 Mega Froslass ex active, lost to a single 3-prize KO.
# The plain leaf scores prizes+HP only, so the winning-but-exposed line out-values the safe line.
# CLOSING_EXCHANGE adds an endgame-only exchange term: once EITHER side is within closing range
# (we have taken >= CE_TRIGGER_TAKEN prizes, or the opponent has <= CE_TRIGGER_OPP_LEFT left), the
# leaf values the opponent's BEST RETURN on the leaf board (strongest payable-within-one-attach
# return-KO; promote lines via bench attackers; gust lines onto our bench at a discount) and — via
# the candidate comparison at the root — our legal alternatives (retreat the damaged active, switch
# attacker, alternative KO target) whose leaves don't concede that return. Terminal wins still
# short-circuit, so a line that CLOSES NOW is never penalised. OPP_HAND_W wires the Resentful-
# Refrain sizing (damage per card in the DEFENDING player's hand) into the exchange damage model —
# our own win-con was previously sized as a flat attack (sees-but-misvalues). Flag-off =>
# byte-identical. FTC frame suite: 26 live fixtures + gate on VPS ~/ptcg/_ftc/.
CLOSING_EXCHANGE = False   # master flag (repo default OFF; candidate bundles ship ON)
CE_TRIGGER_TAKEN = 4       # closing mode when we have taken >= this many prizes...
CE_TRIGGER_OPP_LEFT = 3    # ...or the opponent has <= this many prizes left: a single Mega-ex KO
                           # takes 3 (api.py:476-477), so opp_left=3 is ALREADY closable — the
                           # verified ep-82698289 root sits at taken=2/opp_left=3 and must trigger
CE_LOSS_W = 5000.0         # opp best return CLOSES the game on this leaf board (> a 3-prize KO
                           # swing +HP+answer ~3700+, so it can flip the greedy take-3-but-lose line)
CE_KO_W = 260.0            # per prize of a non-closing opp return-KO (sub-prize: shapes, never flips)
CE_ANSWER_W = 400.0        # our own next attack can CLOSE from this leaf (keep a live closer)
CE_GUST_DISCOUNT = 0.1     # gust/board-pull lines need an unseen card AND the pull; keep the safe
                           # (retreat) line clearly above the exposed-active line it must beat
# CE_SLOW_ONLY (2026-07-06, report/FTC_CLOSING_EXCHANGE.md unblock path): CE w5000 is field-refuted
# ALWAYS-ON (-0.049 pooled: over-defends prize_race aggro — megastarmie/dreepy/iono) but gained on
# stall/mill tanks (archaludon/dragapult), and w<=3000 fails the 26-frame gate. This gates the CE
# term on the IDENTIFIED opponent archetype being SLOW: matchup_priors.json (BEFORE artifact,
# canonical labels since b06f6a4) wincon in {stall, mill_deckout} or pace rate <= CE_SLOW_RATE —
# the mined pace splits the field-A/B sign exactly (gained <=0.58, lost >=0.65). Unknown/missing
# prior => NOT slow (fail-safe: term stays off vs unidentified opponents). Requires the _OPP_ARCH
# producer gate (the RACE_CLOCK-confirm lesson) — wired below. Flag-off => byte-identical.
CE_SLOW_ONLY = False
CE_SLOW_RATE = 0.6
# Root-decision cache for the slow verdict: _ce_opp_is_slow() does string joins over the priors
# table — calling it PER LEAF ran ~1100×/game and, with the warm-cache exclusion, starved the
# TIME_AWARE search in the treatment arm (the 07-06 A/B compute confound; archaludon -0.087 was
# unattributable between term-value and lost search depth). Computed ONCE per root in agent()
# (where _OPP_ARCH is set); leaves read the bool. False whenever belief/flags are unavailable.
_CE_ROOT_SLOW = [False]
                           # (0.2 left the verified 82773840 retreat 153 below the greedy attack)
OPP_HAND_W = 50.0          # Resentful-Refrain sizing: damage per card in the DEFENDER owner's hand
                           # for "for each card in your opponent's hand" attacks (printed value 50)

# ARCHETYPE-CONDITIONED MULTI-TURN PRIZE-RACE assessment (#new-goal): a board value that looks AHEAD
# several turns analytically as a RACE — who reaches 0 prizes first — conditioned on (a) the opponent
# ARCHETYPE (their prize-taking RATE: aggro/burst race fast, stall/disruption crawl), (b) our UNVEILED
# KNOWLEDGE (belief MAP archetype from revealed cards), and (c) OUR SITUATION (prizes left + whether we
# have a KO / 2-prize-ex KO available this turn = our rate). value += W·(opp_clock − our_clock): positive
# when WE win the race. This is the multi-turn lookahead the flat prize-diff term lacks — it knows a
# 1-prize lead vs a fast aggro deck is LOSING (their clock is shorter) and a slim lead vs stall is WINNING.
# FLAG-OFF by default (byte-identical); rates are SEED priors (tune on the field before trusting).
PRIZE_RACE = False
PRIZE_RACE_W = 120.0        # per turn of race-clock advantage; CLAMPED below a prize so it shapes, not flips
PRIZE_RACE_CLAMP = 800.0    # |race value| ceiling (< one prize 1000) — never override the prize race itself

# THE GENERIC STRATEGY LAYER — "Who's the Beatdown?" (matchup_role.py). The prior intel terms
# (PRIZE_RACE/OPP_THREAT/BELIEF) used the VEIL without first deciding WHICH ROLE we play, so their
# sign was wrong for control matchups (race-clock told froslass to race alakazam => regressed). This
# layer assigns a role per matchup from OUR deck-speed vs the OPPONENT's (belief archetype), gated by
# VEIL confidence, then sets the DIRECTION of the race/survival value. Generic over any deck + any
# unveiled opponent. FLAG-OFF default (byte-identical); validate on field + held-out wall before ship.
ROLE_AWARE = False
ROLE_W = 1.0                # overall magnitude multiplier on the role-directed value (race/survival)
ROLE_SURVIVAL_W = 110.0     # per-unit weight inside _survival_value (kept < a prize, like the race)
ROLE_SURVIVAL_CLAMP = 800.0 # |survival value| ceiling (< one prize 1000) — shapes, never flips the race
OUR_SPEED = None            # our deck's prizes/turn proxy; computed once from MY_DECK (override via config)
_OPP_ROLE = [(0, 0.0)]      # (role, weight) cached per MAIN decision from the real root (set in agent())
# S1 — ROLE_MATCHUP (2026-07-22, report/hariyama_loss_anatomy_20260722.md surgery candidate S1): the
# NARROWER, archetype-CONDITIONAL activation of the role layer above. ROLE_AWARE's always-on
# speed-derived form is field-refuted as an always-on layer (project_ptcg_multistate_wiring /
# feedback_complexity_not_improvement); this flag instead fires the SAME role machinery
# (matchup_role.role_value + _survival_value via _OPP_ROLE — DRY, no second role system) ONLY when
# the MINED matchup_priors entry for the belief-MAP opponent archetype says wincon == "stall"
# (data-driven from the shipped BEFORE artifact, NOT a hardcoded archetype list) AND the posterior
# is a real inference (non-fallback, conf >= ROLE_MATCHUP_MIN_CONF — the OPP_DECK_MODEL/DENY_EVO
# convention; NOTE deliberately below matchup_role.CONF_MIN=0.30, which is why the weight is a
# binary 1.0 on gate-pass rather than _conf_weight's 0.30-0.70 ramp — the sibling binary-gate
# precedents RISK_SLOW_TAU/_deny_evolution_terms, not a second confidence scheme). When it fires the
# posture is CONTROL (role -1): the leaf follows _survival_value (grind/deny-exposure/bench-depth)
# instead of the race — vs the Hariyama/Mega-Lucario family this is the measured win template
# (Crustle-wall attrition: 17.6-turn wins vs 11.7-turn losses; racing = feeding 3-prize one-shots).
# Precedence: when ON it OVERRIDES ROLE_AWARE's speed-derived assignment (it IS the conditional
# narrowing of that layer); _OPP_ROLE is reset EVERY decision (the RISK_SLOW_TAU convention) so a
# stale role can never leak across decisions. Flag-off => byte-identical.
ROLE_MATCHUP = False
ROLE_MATCHUP_MIN_CONF = 0.15  # posterior-confidence floor (OPP_DECK_MODEL_MIN_CONF convention)

# SELF-BLUNDER avoidance (loss review 2026-06-27 + engine loss reasons 2/3, api.py:320). Unlike the
# refuted general value levers (RICH/STAGE/BELIEF/TREE — all flat-or-worse on the field), these fire
# ONLY in the near-TERMINAL region (the one bin where the leaf is known to discriminate, AUC>0.6) and
# defend documented FIELD losses: 5/16 lost benchless (reason 3), several to self-mill (reason 2). Both
# FLAG-OFF by default (byte-identical); magnitudes held below the hard win/loss (1e6), tunable via value_db.
DECKOUT_AWARE = False       # penalise approaching deck-out (our own draw/thin engine milling us out)
DECKOUT_W = 300.0           # per card below the floor (≈0.3 prize at the floor edge; ramps as deck→0)
DECKOUT_FLOOR = 5           # start penalising when our remaining deck ≤ this many cards
# DECK_RACE (2026-07-30, the mirror deficit lever): DECKOUT_AWARE above is SELF-preservation only —
# it knows our own deck is thin but nothing about wanting MORE deck than the opponent. Measured
# (probe_mirror_deficit.py, 156 Elo-band-stratified mirrors since 07-01): our Crustle-mirror
# winrate sits ~10.5pp below our own band-expected rate, z=-2.64, and 25% of mirror losses are OUR
# DECK-OUT — 5x the all-opponents rate — because two copies of the same slow list grind each other
# and the loser of the deck race decks out first. This term values the deck-count DIFFERENTIAL,
# fired only when EITHER deck is inside the grind horizon (deck-out is a live win condition), so a
# normal midgame never pays it: lines that burn OUR deck (excess draw/search) score lower, lines
# that force THEIR draws score higher. Archetype-agnostic by design — the differential is real in
# any grind, the mirror is just where it bites — so no fragile belief gate. Clamped sub-prize
# (600 < 1000): shapes ties, never flips a prize-decisive ordering; under WIN_VALUE it scales by
# the local prize gradient like every sub-prize term. NO ELO CLAIM (offline is uncalibrated); the
# probe sized and localized the target, the ladder verdict concludes. Flag-off => byte-identical.
DECK_RACE = False
DECK_RACE_W = 25.0          # per card of (my_deck - opp_deck) differential inside the horizon
DECK_RACE_TRIGGER = 15      # fires when EITHER deck <= this (deck-out within planning horizon)
DECK_RACE_CLAMP = 600.0     # whole-term ceiling < one prize (1000)
# LEAF_V3 (2026-08-09, the WITHIN-POSITION resource axis): the shipped leaf is prize*1000 + board HP
# and nothing else, and it is BELOW CHANCE at the only contrast a search actually makes. Measured on
# 243,422 end-of-turn rows over 12,814 deck-fixed episodes (deck md5 88cc3fe1811d, episode-disjoint
# split), ranking states WITHIN a position stratum — (turn bucket, both prize counts, both HP
# buckets, both body counts), i.e. the alternatives one decision chooses between:
#     shipped (pz*1000+HP)   within-position AUC 0.4825   (pooled 0.6293)
#     shipped + DECK_RACE                       0.4867    delta +0.0044 [+0.0027, +0.0063]  0/200<=0
#     pooled-fit 13f                            0.4942    delta +0.0122 [-0.0064, +0.0296]  FAILS
#     FIXED-EFFECTS 13f                         0.5091    delta +0.0270 [+0.0089, +0.0441]  0/200<=0
# POOLED AUC SELECTS THE WRONG MODEL — the pooled vector's dk_me is 12x the within-position one and
# its within-turn DELTA slope is negative, so maximizing it prefers plans that DRAW FEWER CARDS.
# Only the FIXED-EFFECTS coefficients ship, and they live in leaf_fe.py as source literals.
#
# THE SUPPORT CONSTRAINT IS THE WHOLE DESIGN. The FE coefficients were fitted HOLDING PRIZES FIXED,
# so they carry no information about trading against a prize and are never allowed to: prize*1000 is
# untouched above, and this enters as a CLAMPED second axis — the same shape DECK_RACE uses at
# ~:2460, with two differences, both deliberate.
#
# (1) The ceiling is STRUCTURAL, not conventional: LEAF_V3_CEIL_FRAC re-clamps against
#     VALUE_BASE["prize"] at every evaluation, so no config edit (and no value_db push) can widen
#     the axis past its share of a prize without editing this module.
# (2) The ceiling is 0.45 of a prize, NOT DECK_RACE's 0.6 — because "clamped below one prize" is
#     the WRONG invariant and DECK_RACE's 600 does not actually satisfy the right one. The ordering
#     a search makes is PAIRWISE: for two candidate states A (one prize ahead) and B, a term worth
#     up to +C on B and -C on A moves their gap by 2C, so a prize-decisive ordering survives only
#     while 2C < VALUE_BASE["prize"]. At C=600 the worst case is a 1200-point swing and the axis CAN
#     flip a full prize; at C=450 the worst case is 900 and it cannot. Realized |score()| on real
#     rows spans ~[-0.27, +0.22], so at W=1000 a typical extreme is ~0.25 prize and the clamp still
#     only bites pathological boards — the tighter ceiling costs nothing we measured.
#     (This says nothing about DECK_RACE, which is a separate shipped lever with its own record;
#     it is noted here only because copying its constant would have quietly broken requirement 2.)
#
# leaf_fe.score() is in P(win) units (change per +1 SD, position held fixed). NO ELO CLAIM: the
# within-position AUC sized the target, the argmax-flip probe and the ladder verdict conclude.
# Flag-off => byte-identical (see src/agent/test_leafv3_byte_identical.py).
LEAF_V3 = False
LEAF_V3_W = 1000.0          # material points per 1.0 of leaf_fe.score() (1 prize = 1000 = 1.0 P(win))
LEAF_V3_CLAMP = 450.0       # whole-term ceiling; 2*450 < one prize => cannot flip a prize ordering
LEAF_V3_CEIL_FRAC = 0.45    # HARD ceiling as a fraction of one prize; re-applied every evaluation
# LEAF_DYN (2026-08-09, the DYNAMIC objective): REFUTED AT ITS OWN PRE-REGISTERED GATE. Built,
# wired, byte-identical off, and it MUST STAY OFF. The record, because a refutation nobody can
# read gets rebuilt by the next session:
#
# THE HYPOTHESIS. LEAF_V3 above is ONE resource vector fitted across every game. But games end
# three different ways, and the corrected census (classifier validated at 99.19% winner agreement,
# n=12,826) says all three are live: PRIZE_OUT 0.6155 / DECK_OUT 0.1874 / BENCH_OUT 0.1548. So
# condition the axis on WHICH terminal condition is approaching, and shift the CLAMPED second
# axis's weights accordingly — never the prize core. That is the ask, and the premise held: the
# same 8 features refitted per condition give three vectors that DISAGREE IN SIGN on 7 of 8
# features (leaf_fe.W_LANE). dk_me is +0.398 in the deck lane and -0.327 in the bench lane.
#
# THE PRE-REGISTERED GATE, per condition, never pooled. Within-position AUC on test rows split by
# the condition their episode actually ended in; 200 episode-clustered bootstrap resamples; PASS
# iff at least one condition's delta CI excludes zero ABOVE and none excludes zero BELOW ("more
# than the measurement resolution" = the CI half-width, so a regression counts only when its CI
# clears zero). DYN vs LEAF_V3-static:
#     PRIZE_OUT  +0.0203  [+0.0120, +0.0319]      improves
#     BENCH_OUT  +0.1516  [+0.1165, +0.1846]      improves
#     DECK_OUT   -0.0930  [-0.1257, -0.0644]      REGRESSES -> GATE FAILS
# Not tuned afterwards, by instruction and on principle: the previous 3-way win-condition gate
# broke its own threshold (BENCH_OUT 0.752 -> 0.674) and shipped anyway.
#
# WHY IT FAILS, measured rather than guessed. The detector may only read leaf-visible state, and
# the lane it must call is the one it calls worst: inside episodes that truly ended in DECK_OUT it
# puts mean weight 0.239 on the deck lane (argmax on 12.1% of rows); in true BENCH_OUT rows, 0.193
# and 0.23%. Blending sign-conflicting vectors at that confidence collapses to the prior mixture —
# the effective dk_me inside true DECK_OUT rows is +0.012 where its own lane wants +0.398, i.e. 3%
# of the intended coefficient, and worse than the global vector's +0.065 which at least has the
# right sign there. An ORACLE detector (true label, unshippable) reaches 0.8336 in the deck cell
# against the fitted blend's 0.6568, so the lanes are right and the detector is the entire gap.
# A better detector is the only thing that could revive this; a re-weighting cannot.
#
# TWO FINDINGS ABOUT LEAF_V3 ITSELF fell out of the same per-condition split, and they matter more
# than this flag does (see also the note above LEAF_V3_W):
#   (a) LEAF_V3 is NOT lane-neutral. Against the shipped leaf it is +0.0208 in PRIZE_OUT and
#       +0.0592 in DECK_OUT but -0.1000 [-0.1386, -0.0571] in BENCH_OUT — a real regression, CI
#       clear of zero, in 15.5% of games. Its global number is a net of a large win and a large
#       loss, not a uniform gain.
#   (b) The +0.0270 that authorised LEAF_V3 was measured on the FE score STANDALONE (a leaf with
#       no prize term and no HP term — _rb_plan_leafrank.py ranks `scF`, not `ship + scF`). The
#       COMPOSITE that actually ships, prize*1000 + HP + clamp(1000*FE), scores +0.0096
#       [-0.0026, +0.0224] on the same instrument: CI straddles zero. Reproduced both in the same
#       run (<workdir>/_c3_composite.py). Within a stratum the prize term is constant but HP is
#       not (the buckets are 150 wide), so the surviving HP differential competes with the axis
#       instead of being replaced by it. This is consistent with C1's independent finding that
#       arming LEAF_V3 changed no decisions (+0.00pp / -0.63pp, both CIs straddling zero).
#
# Provenance: <workdir>/_c3_dynleaf.py (fit + gate), _c3_diag.py (oracle/sign/confidence),
# _c3_composite.py (finding b). Coefficients are SOURCE LITERALS in leaf_fe.py, never a data file.
# Flag-off => byte-identical (src/agent/test_leafdyn_byte_identical.py). PRECEDENCE: LEAF_DYN wins
# over LEAF_V3 when both are armed — the two are alternative fits of the SAME axis and adding them
# would double-count it. scripts/test_config_contract.py refuses a config that arms both.
LEAF_DYN = False
LEAF_DYN_W = 1000.0         # material points per 1.0 of leaf_fe.score_dyn() — same units as LEAF_V3_W
LEAF_DYN_CLAMP = 450.0      # whole-term ceiling; the PAIRWISE bound 2*C < one prize, as above
# EVOLVE_TEMPO leaf half (2026-08-01): the root cause of our measured 0.928 reflexive evolve rate
# is that the leaf REWARDS evolving instantly (evolved form = more board HP) while holding has NO
# represented value — surprise, gust-avoidance and timing are invisible to a material leaf. This
# term gives a held, playable evolution its option value: each hand card whose evolvesFrom names a
# Pokemon on OUR board is worth EVOLVE_TEMPO_HOLD_W, phase-weighted by the measured hold curve
# (early 0 — strong players evolve freely there; mid 1.0; late 1.3). Sub-prize clamp. Shares the
# H-side flag: ordering shapes candidates/rollouts, this shapes the searched VALUE, so "hold now,
# evolve when it buys something" can win a tie on merit instead of by quota.
EVOLVE_TEMPO_HOLD_W = 60.0
EVOLVE_TEMPO_STAGE_W = [0.0, 1.0, 1.3]   # by _game_stage: early, mid, late
EVOLVE_TEMPO_CLAMP = 240.0
BENCHLESS_LETHAL = False    # penalise a benchless board (a KO of our lone active = instant loss, reason 3)
BENCHLESS_W = 3000.0        # benchless WHILE the opponent can KO our active next turn (≈3 prizes ⇒ avoid)

# BENCH_FORCE (anti-brick, 2026-06-28): a benchless board is the instant-loss CLIFF — a KO of the lone
# active = no Pokémon in play = engine loss reason 3. Developing a backup basic is OPPONENT-INDEPENDENT
# (not a mirror mirage), and the soft BENCHLESS leaf terms failed to fix it (the prize+hp leaf credits
# +70 board-hp for a benched basic but the search washes it out / it never enters the TOP_M set). So this
# is a HARD pre-search rule (_force_bench_dev): at OUR MAIN decision, while bench has < BENCH_FORCE_MIN
# Pokémon and a Basic is playable from hand, develop it — BUT only when staying benchless is actually
# DANGEROUS (the soundness rule, user 2026-06-28): no-bench is allowed iff we have a killing blow this turn
# OR we are highly certain the opponent cannot KO our active next turn (_benchless_is_safe). Otherwise we
# MUST bench. Benching is a free action so control returns to the search for the rest of the turn. NOTE:
# the residual deck-brick (drawing only 1 basic all game) is a DECK problem fixed by the pokepad2 deck
# (Poké Pad basic-finder), not by this rule. FLAG-OFF default (byte-identical); on via the ship config.
BENCH_FORCE = False
BENCH_FORCE_MIN = 1         # maintain at least this many benched Pokémon when benchless is unsafe (cap risk)
# --- DOMINANCE_FORCE (2026-07-25, user directive "implement strictly dominant player moves and PLAY
# them instead of relying on probability/evolution based decision"; memory project_ptcg_dominance_layer)
# WHY A ROOT OVERRIDE AND NOT AN ORDERING BAND: two layers decide a MAIN move — H._score ORDERING picks
# which options survive the TOP_M prune, then minimax VALUE picks among them. _state_value scores
# prize(x1000) + board HP and credits energy ONLY under RICH_EVAL, which the shipped config leaves
# FALSE => a free energy attach is worth exactly 0.0 to the search, so a turn-ending lethal always
# out-values it. That is why ATTACH_FIRST (an H._score band of 12000) was MEASURED INERT-BUT-CHURNING
# (569 teacher frames: changed 32% of decisions yet ATTACH share stayed 4%, and it made both the
# teacher-profile L1 and top-1 agreement WORSE) — the band gets the attach SEARCHED, then the
# energy-blind leaf discards it. A dominance fact must therefore be played BEFORE search, exactly like
# the existing BENCH_FORCE ("hard pre-search rule; benching is a free action") and G1_CLOSE ("a found
# multi-step close is a FACT; play it").
# THE DOMINANCE PROOF (from the 100% knowns of OUR OWN board — no probability, no opponent model):
# the energy attach is FREE, once-per-turn and EXPIRING, and it does NOT end the turn. So
# attach-then-X is legal in the same turn for every X (observed 116x/40 games), which makes a
# BENEFICIAL attach weakly dominant over spending the decision on X alone, and strictly dominant
# whenever the game continues (measured 98.0% [95.4,99.1] of the declines). The KO is deferred by one
# decision, never lost.
# SCOPE = THE PROOF'S PRECONDITIONS (inherited verbatim from ATTACH_FIRST's vetted exclusions, and
# detected by DIFFING H._score with the flag flipped so there is ZERO duplicated rule logic):
#   * OVERFILL attaches are excluded (_can_pay_strongest early-return) — a fully fuelled target gains
#     nothing, so the attach is NOT beneficial and dominance does not hold;
#   * a DOOMED-ACTIVE energy sink is excluded (the Class-A LOSS_AUTOPSY_2026-07-04 pattern) — energy
#     into a mon that dies next turn is not free value.
# WHICH attach: among the qualifying ones we keep the H._score argmax, so the bench-vs-active /
# PLAN / TEMPO shaping AND the live attach_to_attacker prune weight still choose the target.
# LOOP SAFETY: capped at ONE force per turn via _DOM_FORCED (attach is once-per-turn by rule, so the
# options should vanish anyway — the cap makes the agent loop-proof even if that rule ever changes).
# Flag-off => byte-identical (helper never called, no DIAG key, ordering and values untouched).
DOMINANCE_FORCE = False
_DOM_FORCED = [None]        # (turn, count) — at most one forced dominant free action per turn
# --- DOMINANCE_CLOSE (2026-07-25, memory project_ptcg_our_blunder_ledger) -------------------------
# THE MEASUREMENT THIS IMPLEMENTS. On 2,162 of OUR OWN real ladder games, scored against classes whose
# dominance was validated by ~1305-Elo TEACHER COMPLIANCE (>=90% rate AND >=90% game-clustered lower
# bound), we violate the SETUP classes badly while our ATTACK play is already at parity:
#     class (teacher rate -> OURS)                      fixable blunders/game
#     ATTACH completes an attack cost   96.9% -> 57.2%          1.489
#     FREE SUPPORTER                    97.3% -> 80.0%          1.000
#     ABILITY available                 98.0% -> 72.2%          0.527
#     ATTACH completes cost ON ACTIVE  100.0% -> 80.0%          0.486
#     FREE BENCH of a 1-PRIZE basic    100.0% -> 81.9%          0.234
#     ATTACH completes a LETHAL        100.0% -> 73.6%          0.191
#     EVOLVE available                 100.0% -> 97.0%          0.053
#     (ANY attack 97.1% -> 97.3%, LETHAL 98.2% -> 98.9% — we are already AT/ABOVE the teacher)
# ~4.0 fixable violations per game, all in SETUP. Root cause is the leaf: _state_value scores
# prize*1000 + board-HP and credits energy/abilities only under RICH_EVAL (shipped FALSE), so a
# cost-completing attach or an ability activation is worth 0.0 to the search.
# 🔴 WHY "CLOSE" AND NOT "FORCE-FIRST" (this is the whole safety argument). Compliance was measured at
# the TURN unit — "did the pilot play this class at SOME point during the turn". Forcing the class at
# the FIRST opportunity would also dictate SEQUENCING, which we have NO evidence for and one direct
# counter-example (draw-before-tutor is 9.6% and REVERSED — the teacher goes tutor-first). The broad
# force-first rule was in fact REFUTED: it drove ATTACH share to 41% vs the teacher's 22%. So this rule
# fires ONLY at the moment the resource would be WASTED: the search has already chosen to END the turn
# while a teacher-mandatory class is still available. It converts a wasted expiring resource into a
# used one and can do nothing else — it never pre-empts an attack, a KO, or any other searched move,
# and it cannot reorder plays within a turn.
# LOOP SAFETY: every forced class consumes a once-per-turn resource (energy attach / supporter) or a
# distinct board slot, so the same class cannot re-offer indefinitely; _DOM_CLOSED additionally caps
# the overrides at DOMINANCE_CLOSE_CAP per (game-)turn.
# Flag-off => byte-identical (helper never called, no DIAG key, decision returned unchanged).
DOMINANCE_CLOSE = False
DOMINANCE_CLOSE_CAP = 3     # max END-overrides per turn (a turn offers at most a few free resources)
# DOM_CLOSE_DECK_GUARD (2026-07-27, adversarial review): class 2 of _dom_close_classes appends EVERY
# free Supporter with NO value test and NO deckCount test — unlike class 3, which gates on
# _skill_value >= HIGH_VALUE_ABILITY_MIN. The block then discards the searched best. On this deck
# that is load-bearing: crustle_teacher runs 14 Supporters over 4 names including 4x Lillie's
# Determination ("shuffle your hand into your deck, then draw 6"), so deck_after = deck + hand - 6,
# deck-NEGATIVE whenever hand < 6 — and the leaf carries DECKOUT_AWARE precisely to rank that line
# last in a grind.
# MEASURED (scripts/probes/probe_dom_close_class.py, n=10 games, live config): DOMINANCE_CLOSE
# overrides the search 3.4x/game; 12 of 34 overrides were class-2 Supporters, and 5 of those 12
# (42%) fired at deckCount <= 10, minimum observed deckCount = 2.
# THE ARGUMENT for this guard is not that free Supporters are bad — the teacher plays them 97.3% and
# we only 80%, which is why DOMINANCE_CLOSE exists. It is that its justification ("the search
# undervalues a free expiring resource") FAILS when deck-out is the binding constraint, because
# there the search has a specific, correct and very large term that this override throws away.
# The threshold is a judgment call, not a measured optimum — hence a flag, so it can be measured.
DOM_CLOSE_DECK_GUARD = False
DOM_CLOSE_MIN_DECK = 8      # below this, trust the search's DECKOUT_AWARE over the free-Supporter rule
_DOM_CLOSED = [None]        # (turn, count)
_ATTACH_TYPES = frozenset(int(v) for v in (
    getattr(OptionType, "ATTACH", 8), getattr(OptionType, "ENERGY", 8),
    getattr(OptionType, "ENERGY_CARD", 8)))
# --- DOMINANCE_ENABLING_ONLY (2026-07-25): NARROW the forced class to ENABLING attaches only.
# WHY: the broad rule above was MEASURED on 569 teacher MAIN frames / 15 games (harness
# <workdir>/_divergence_multi.py, one reference entrant). It FIXED the PLAY share (53% -> 41%
# against the teacher's 40%) and improved the mid/late per-frame L1, but ATTACH OVERSHOT to 41% vs the
# teacher's 22% (per-type error +19 where the live arm's was +4), EARLY-phase L1 got WORSE (47.6 ->
# 62.7) and top-1 agreement was flat-to-down (0.281 -> 0.279) — it fails the pre-registered G-overshoot
# and no-worsened-phase guards. THE ARITHMETIC: one forced attach over ~2.4 MAIN decisions/turn IS
# ~41% ATTACH, so a teacher sitting at 22% DECLINES its free expiring attach roughly half the time.
# "Always spend the free attach" is therefore NOT pro play and the strict-dominance premise above was
# too BROAD: energy is also a HAND resource with option value (ATTACH_FIRST's own comment concedes the
# teacher wastes its OWN attach on 16.7% [13.0, 21.3] of its attacks).
# THE NARROW CLASS (hypothesis, NOT yet measured): the attach must buy IMMEDIATE CAPABILITY rather
# than option value — it must COMPLETE an attack cost, i.e. after this attach the target can pay an
# attack it cannot pay now (need - have == 1). NO cost logic is re-implemented for this: `need` comes
# from _consensus_attack_cost (the SAME detector POLICY_PRIOR_PRUNE's `attach_to_attacker` key uses:
# min energy count over the card's attacks) plus H._strongest_attack's own energies, and the OVERFILL
# end of the range stays with H._can_pay_strongest, which H._score already early-returns on (so an
# exact-completion attach can never also be an overfill). See _dominance_enabling().
# NOT itself a behaviour flag: it only narrows a rule that is gated by DOMINANCE_FORCE (flag-off), so
# it may default True — with DOMINANCE_FORCE False nothing calls it and the kernel stays byte-identical.
DOMINANCE_ENABLING_ONLY = True

# ROBUST reconfiguration-path choice (the user's "drop all paths that are dangerous to us / never hope
# for a blunder"). The refuted BELIEF lever just AVERAGED leaf values over opponent worlds (smearing the
# 50% value-ties). This instead keeps the mean as the primary sort, then BREAKS TIES by the WORST-CASE
# belief world — among moves of ~equal expected value, take the one whose worst plausible opponent reply
# is least bad (maximin). Meaningful only with BELIEF on + K_WORLDS>1 (a spread to be robust over). The
# forced-loss-aware leaf (deckout/benchless above) is what makes a "dangerous" world score as a loss, so
# the maximin actually prunes self-losing lines. FLAG-OFF default (byte-identical).
ROBUST_TIEBREAK = False
ROBUST_BAND = 500.0         # mean-value gap (< half a prize) within which candidates count as a "tie"

# THREAT ENVELOPE (the user's "acknowledge the visible board + anticipate their deck"): judge a benchless
# board lethal against the opponent's FULL fueled next-turn burst (best attack if they attach one energy,
# dynamic+weakness via _opp_burst_dmg) PLUS the archetype's probable hidden booster (_arch_dmg_bonus) —
# not just their current-energy attack. Used inside BENCHLESS_LETHAL only (the fatal reason-3 region), so
# it sharpens that check rather than adding a broad exposure discount (which is the refuted OPP_THREAT).
THREAT_ENVELOPE = False
# opponent prizes-per-turn by archetype, matched by KEYWORD on belief's label (e.g. 'Mega Lucario ex /
# Riolu', 'Spheal / Walrein'). SEED priors: aggro/burst/spread race fast; stall/disruption crawl. First
# matching keyword wins; no match -> 1.0. (belief labels are top-2 Pokemon lines, not our gauntlet keys.)
ARCH_PRIZE_RATE_KW = [
    ("lucario", 1.6),       # aggro — fast 2-prize swings
    ("alakazam", 1.4), ("kadabra", 1.4), ("abra", 1.4),     # Powerful-Hand burst
    ("dragapult", 1.5), ("dreepy", 1.5), ("drakloak", 1.5),  # Phantom-Dive spread
    ("starmie", 1.3), ("abomasnow", 1.1), ("snover", 1.1),   # big-ex
    ("archaludon", 1.1), ("duraludon", 1.1),
    ("crustle", 0.9), ("dwebble", 0.9), ("dunsparce", 0.9), ("dudunsparce", 0.9),
    ("trevenant", 0.7), ("phantump", 0.7),                   # toolbox/midrange
    ("iono", 0.5),                                            # disruption
    ("walrein", 0.4), ("spheal", 0.4),                       # stall
]


# --- RACE_CLOCK (VALUE LEAF v2 core — report/VALUE_LEAF_V2.md): two-sided WIN-CONDITION race ------
# clock as the PRIMARY dynamic signal (up to ±RACE_CLOCK_W prizes, saturated) + a posture controller
# (press when BEHIND the race). Unlike PRIZE_RACE/_race_leaf_terms (clamped < 1 prize, tie-breakers),
# this can refuse a bait prize that loses the race. Flag-off => byte-identical.
RACE_CLOCK = False
RACE_CLOCK_W = 1.5            # max clock-advantage value, in PRIZES (1.5×1000 << terminal 1e6)
RACE_CLOCK_SCALE = 3.0        # turns of clock difference that saturate the signal
POSTURE_PRESS_W = 250.0       # press bonus × _ko_potential when BEHIND (the "start pressing" posture)
POSTURE_RISK_RELIEF = 0.5     # fraction of OWN_PRIZE_RISK forgiven when behind (accept trades)
_MATCHUP_PRIORS: dict = {}    # BEFORE artifact: archetype -> {"rate": prizes/turn,...}; {} = unloaded
_MATCHUP_PRIORS_TRIED = [False]


def _load_matchup_priors() -> dict:
    """Feedforward (BEFORE) hook: per-archetype priors mined offline, shipped in the bundle.
    Graceful everywhere: absent file => {} => keyword-pace fallback (_opp_prize_rate).
    Probes module-dir + src/agent + cwd + container (mirrors _load_arch_amp — the review of
    55b3f19 found the cwd-only probe left priors unreachable in offline harnesses)."""
    if not _MATCHUP_PRIORS_TRIED[0]:
        _MATCHUP_PRIORS_TRIED[0] = True
        try:
            here = os.path.dirname(os.path.abspath(__file__))
        except NameError:                  # kaggle_environments execs the source without __file__
            here = os.getcwd()
        for p in ("matchup_priors.json", os.path.join(here, "matchup_priors.json"),
                  "src/agent/matchup_priors.json", "/kaggle_simulations/agent/matchup_priors.json"):
            try:
                with open(p) as f:
                    _MATCHUP_PRIORS.update(json.load(f) or {})
                break
            except Exception:
                continue
    return _MATCHUP_PRIORS


def reconfigure_deck(cards) -> None:
    """League/harness hook (review of 55b3f19): re-point the PAIR at a new deck and reset every
    deck-derived cache on the SEARCH side too. Without this, H.reconfigure_deck alone leaves
    MY_DECK / _WIN_CONDITION / the priors latch stale — harmless for fresh-module-per-deck
    harnesses (subx_bench), WRONG for long-lived league daemons swapping decks in-process."""
    global MY_DECK
    H.reconfigure_deck(cards)
    MY_DECK = list(cards)[:60]
    _WIN_CONDITION[0] = None
    _MATCHUP_PRIORS_TRIED[0] = False
    _MATCHUP_PRIORS.clear()
    _CE_ROOT_SLOW[0] = False


def _race_clocks(mine, opp):
    """(my_clock, opp_clock): each side's estimated TURNS-TO-WIN under ITS OWN win condition.
    Mine: prize_race via _prize_plan pace; stall via Resentful-Refrain 50×opp-hand vs their active
    HP; mill via their deck-out clock. Theirs: prizes-left / (matchup-prior rate if loaded, else
    archetype keyword pace). The clock DIFFERENTIAL is the scenario map the leaf steers by: did the
    state move toward MY win condition or THEIRS?"""
    wc = _my_win_condition()
    my_left = max(1, len(mine.prize))
    opp_left = max(1, len(opp.prize))
    if wc == "mill_deckout":
        odc = getattr(opp, "deckCount", None)
        my_clock = (odc if odc is not None else 30) / 2.0          # v1 mill pace prior: ~2 cards/turn
    elif wc == "stall":
        hand = getattr(opp, "handCount", 0) or 0
        op_act = opp.active[0] if opp.active else None
        op_hp = (getattr(op_act, "hp", 0) or 150) if op_act is not None else 150
        rate = min(1.5, max(0.2, (50.0 * hand) / max(60.0, float(op_hp))))
        my_clock = my_left / rate
    else:
        _, reaches, our_rate = _prize_plan(mine, opp)
        my_clock = 1.0 if reaches else my_left / max(0.5, our_rate)
    _a3 = _OPP_ARCH[0]
    # A prior-only (fallback) posterior => identity is a GUESS, not a read: do NOT trust the
    # identity-specific mined matchup_priors rate (siblings _deny_evolution_terms + RISK_SLOW_TAU
    # also gate on arch[2]). Embedding matchup_priors made fallback use the MODAL deck's slow mined
    # rate as if inferred (~3x wrong clock); on fallback use the identity-agnostic neutral pace.
    _fb = (not isinstance(_a3, tuple)) or len(_a3) < 3 or bool(_a3[2])
    arch = (_a3 or (None, 0))[0]
    if _fb:
        opp_rate = _opp_prize_rate(None)          # 1.0 neutral: unknown opponent, no identity signal
    else:
        pri = _load_matchup_priors().get(str(arch)) if arch else None
        try:
            opp_rate = float(pri["rate"]) if (pri and pri.get("rate")) else _opp_prize_rate(arch)
        except Exception:
            opp_rate = _opp_prize_rate(arch)
    opp_clock = opp_left / max(0.3, opp_rate)
    return my_clock, opp_clock


def _opp_prize_rate(arch_label):
    """Map a belief archetype label -> opponent prizes-per-turn via keyword (default 1.0)."""
    if not arch_label:
        return 1.0
    s = str(arch_label).lower()
    for kw, rate in ARCH_PRIZE_RATE_KW:
        if kw in s:
            return rate
    return 1.0


# Per-archetype HIDDEN damage cushion (the user's "their deck likely has an X-damage trainer / damage-amp
# support / super-evolution swing — account for it even though we haven't seen it"). Added to the visible
# fueled burst when judging whether a BENCHLESS board is lethal. SMALL + DEFAULT-EMPTY-effective (default
# bonus 0) on purpose: hand-set archetype priors HURT when over-trusted (see PRIZE_RACE regression), so we
# only nudge for archetypes with a well-known booster, keep magnitudes modest, and gate on the field.
ARCH_DMG_BONUS_KW = [
    ("lucario", 40),        # Mega Brave swing + Aura-Jab discard-energy accel / Premium Power Pro tool
    ("dragapult", 30), ("dreepy", 30), ("drakloak", 30),     # Phantom-Dive spread + damage tools
    ("dudunsparce", 20), ("dunsparce", 20),
]
ARCH_DMG_BONUS_DEFAULT = 0.0


def _load_arch_amp():
    """MINED per-archetype off-attack booster (scripts/mine_dmg_amp.py): {label: {amp, booster, cond}}.
    Data-grounded replacement for the hand-set table; empty dict if the artifact isn't present."""
    try:
        here = os.path.dirname(os.path.abspath(__file__))
    except NameError:                      # kaggle_environments execs the source without __file__
        here = os.getcwd()
    for p in ("arch_dmg_amp.json", os.path.join(here, "arch_dmg_amp.json"),
              "src/agent/arch_dmg_amp.json", "/kaggle_simulations/agent/arch_dmg_amp.json"):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            continue
    return {}


_ARCH_AMP = _load_arch_amp()


def _arch_dmg_bonus(arch_label) -> float:
    """Probable extra next-turn damage their ARCHETYPE can add beyond the visible board. Prefers the
    MINED value (exact label, from their real decklist); falls back to the keyword table, then default."""
    if not arch_label:
        return ARCH_DMG_BONUS_DEFAULT
    v = _ARCH_AMP.get(arch_label)                        # exact label (belief + miner share meta_decks)
    if v and v.get("amp"):
        return float(v["amp"])
    s = str(arch_label).lower()
    for kw, bonus in ARCH_DMG_BONUS_KW:
        if kw in s:
            return float(bonus)
    return ARCH_DMG_BONUS_DEFAULT
# PRIZE-PLAN (the user's "do we know what prizes we could take?"): enumerate the KOs available THIS turn
# (active KO = 1 prize, ex = 2) and whether a KO REACHES LETHAL (sum of takeable prizes >= our prizes
# left -> we can win now). Feeds the race (our rate) + a lethal-reach bonus so search SEQUENCES prizes to
# close, and values the 2-prize-ex KO. FLAG-OFF default. Shares ARCHETYPE/PRIZE flags' validation gate.
_OPP_ARCH = [None]         # (archetype, conf) cached per MAIN decision (set in agent(); avoids re-posterior)
_OPP_AMP_LIVE = [True]     # is the opp's archetype damage-booster still playable? (False once all copies seen)
_PP_LEAF_ARMED = [False]  # POLICY_PRIOR: does its LEAF half actually contribute? Recomputed once
                          # per decision (the root-cache convention) because the weights arrive at
                          # config-apply time, after import. See the WARM_START guard below.
_POLICY_PRIOR_ROOT_PRIZES = [None]  # POLICY_PRIOR: (my_prize_left, opp_prize_left) at the search ROOT,
                                   # captured once in agent(); the leaf tiebreaker fires ONLY on a
                                   # leaf state whose BOTH prize counts equal the root (prize-equal
                                   # gate) so it is decisive among prize-tied lines, never flips a prize.
_OPP_CONSIST_IDS = [None]  # DENY_EVOLUTION: union of card-ids across belief-consistent posterior particles,
                           # cached ONCE per MAIN decision (set in agent()); None => not computed this decision

# --- OPTIONAL leaf-VALUE params (arena/value_db.json) -----------------------------------------
# BACKWARD-COMPATIBLE: these are the SHIPPED leaf constants verbatim. _state_value reads them so the
# arena can tune the leaf without touching code. main.py loads value_db.json ONLY when cfg.value_db
# is set; with nothing loaded these equal the hard-coded constants, so SUB-1..5 reproduce
# byte-identical. LATE_CONVERSION (enabled=False by default => no effect) is the NEW term targeting
# the convergent #1 loss: when even/ahead on prizes past mid-game, value PRESSING for the KO/lethal
# over passive board-HP stalling (we stop getting even-then-stalling while the opp takes 3 prizes).
VALUE_BASE = {"prize": 1000.0, "hp": 1.0, "win": 1e6, "loss": -1e6, "draw": 0.0}
# --- WIN_VALUE: the leaf on a WIN-PROBABILITY scale (measured, not asserted) -------------------
# THE DEFECT. VALUE_BASE above is a MATERIAL score, and the search AVERAGES it three times over:
# across K_WORLDS=6 belief determinizations (_search_timeaware), across opponent replies under a
# policy model (_aggregate_opp), and across rollouts. Averaging an unbounded score whose terminal is
# 1e6 against a prize worth 1000 is not an expectation of anything — ONE belief world flipping to
# "win" moves the mean by ~167,000 while a CERTAIN prize moves it by 1,000, so a 1-in-6 fantasy
# outranks a sure prize 167:1. That is a reproduced misplay, not a theory: the agent scored an ATTACK
# in its own win-band and played END instead (scripts/probes/repro_end_over_lethal.py). The same
# mixture is annihilated a third way in _rollout's ROLL_TO_END fallback, which returned
# _state_value(...)/1e6 — turning a three-prize lead into 0.003 against a terminal win of 1.0.
#
# THE FIX. Map the leaf onto P(win). Terminal win/loss/draw -> 1.0/0.0/0.5; otherwise squash. Then
# the K-world mean is a real expectation of win probability, which is also exactly what the rating
# optimises (Elo is binary; margin does NOT affect it — see report/RULES_AND_RATING.md).
#
# WHAT IS AND IS NOT PRESERVED. It is tempting to claim the squash is a monotone transform of the
# material score, so max()/min() commute with it and minimax inside a world is unchanged. That claim
# is FALSE and tests/test_win_value.py refutes it: the measured table says 6-1 and 5-1 differ by
# 0.002 win probability where the material leaf says they differ by a whole prize, so states in
# DIFFERENT prize configurations genuinely reorder. That reordering is the improvement, not a bug —
# it is the flat 1000-per-prize assumption being replaced by measurement. What IS preserved, exactly:
#   * WITHIN one prize configuration the leaf is strictly monotone in the material score, so every
#     sub-prize term keeps its ordering and its relative weight;
#   * a sub-prize term worth X material points is worth EXACTLY X/1000 of a prize in EVERY cell of
#     the table (see the gradient below), which is precisely the invariant ~15 flag comments assert
#     when they say "clamped < one prize so it only shapes ties".
#
# WHY THE SUB-PRIZE TERMS SCALE BY THE LOCAL PRIZE GRADIENT, not by a global constant. The obvious
# implementation — logit = table + material_remainder / WIN_VALUE_SCALE — is wrong, and subtly so:
# WIN_VALUE_SCALE is fitted on PRIZES, and applying it to HP and to every flag weight assumes those
# convert to win probability at the prize exchange rate everywhere. The table itself disproves that
# (a prize is worth 21x more at 2-2 than at 6-2). Scaling instead by the LOCAL value of one prize
# keeps each term at its designed fraction-of-a-prize in every state, which is both more defensible
# and simpler to reason about.
#
# THE TABLE IS MEASURED, NOT CHOSEN. scripts/probes/calibrate_win_value.py reads the prize
# trajectory of 141,733 real episodes and reports observed P(win | my prizes left, opp prizes left).
# Every one of the 36 non-terminal cells has n >= 17,017. A logistic in the prize DIFFERENCE fits it
# to a weighted RMSE of 0.185 in logit space (82% better than a constant), giving WIN_VALUE_SCALE
# below — but the residual is SYSTEMATIC, not noise: the same differential is worth materially more
# near the end of the race than at the start (worst same-diff spread 0.107 at n>=17k/cell, ~18 sigma).
# So the base logit is the measured table and WIN_VALUE_SCALE converts only the SUB-PRIZE terms.
# What this buys, and the reason it is not cosmetic: under VALUE_BASE a prize is worth exactly 1000
# everywhere, whereas measured it is worth +0.190 win probability at 2-2 and +0.009 at 6-2 — a 21x
# spread the flat leaf cannot express.
#
# KNOWN LIMIT, stated because it bounds the claim: the ETL derives prizes-taken as 6-len(prize) and
# skips the empty-prize frame (scripts/episodes_etl.py:244). That guard is CORRECT — an empty prize
# list also occurs in pre-deal frames, where 6-0 would permanently pin a running max at 6 — but it
# means "0 prizes left" is never recorded, so the table covers exactly the NON-TERMINAL domain
# (left in 1..6). That is the leaf's domain; terminals are returned directly above.
#
# CONSEQUENCE worth knowing before reading a trace: many sub-prize terms are documented as "clamped
# < one prize so it never flips a prize-decisive ordering". Under WIN_VALUE that invariant is
# deliberately relaxed where the measurement says a prize barely matters (at 6-2 a prize is worth
# 0.009 while the sub-prize terms span ~0.045). That is the intended behaviour of a probability
# scale, not a regression.
WIN_VALUE = False
WIN_VALUE_SCALE = 1841.0   # fitted: logit P(win) = prize_diff * 1000 / SCALE  (slope 0.543/prize)
# (my_prizes_left, opp_prizes_left) -> base logit of P(we win). Measured; see the probe above.
WIN_PRIZE_LOGIT = {
    (1, 1): +0.0000, (1, 2): +0.7995, (1, 3): +1.5182, (1, 4): +2.1522, (1, 5): +2.5172, (1, 6): +2.5367,
    (2, 1): -0.7995, (2, 2): +0.0000, (2, 3): +0.7117, (2, 4): +1.2859, (2, 5): +1.7511, (2, 6): +1.8213,
    (3, 1): -1.5182, (3, 2): -0.7117, (3, 3): +0.0000, (3, 4): +0.5678, (3, 5): +1.0759, (3, 6): +1.3176,
    (4, 1): -2.1522, (4, 2): -1.2859, (4, 3): -0.5678, (4, 4): +0.0000, (4, 5): +0.5421, (4, 6): +0.9492,
    (5, 1): -2.5172, (5, 2): -1.7511, (5, 3): -1.0759, (5, 4): -0.5421, (5, 5): +0.0000, (5, 6): +0.4377,
    (6, 1): -2.5367, (6, 2): -1.8213, (6, 3): -1.3176, (6, 4): -0.9492, (6, 5): -0.4377, (6, 6): +0.0000,
}
VALUE_RICH = {"attacker_progress": 60.0, "bench": 15.0, "hand": 5.0}
# RACE_LEAF (V2): win-condition-aware tie-breakers below the prize scale (1000). For STALL (Froslass),
# our Resentful Refrain threat scales with the OPPONENT's hand and we win by grind/deck-out, NOT the HP
# race — so reward a large opp hand + the opponent's deck-out clock. Flag-off => byte-identical.
RACE_LEAF = False
RACE_W = {"resentful": 5.0, "opp_deckout": 2.0}
STARTING_DECK = 60
# EARLY_TEMPO (L1): the diagnosis (adversarial, leaf-free) — in LOSSES our energy attach STALLS at t3-4
# while WINS / the best same-deck pilot keep attaching and take prizes far faster. The RACE_LEAF is near-
# BLIND early and attaching energy changes neither prize nor HP, so the depth-1 search has NO leaf reason to
# commit energy early. L1 = an EARLY (turns<=6) ATTACKER-READINESS term: reward energy progress toward an
# attack's cost on the ONE most-ready intended attacker (NOT bench width, NOT a hand/P(win) calibrator —
# both refuted as RICH_EVAL), plus a light KO-next-turn prize-pressure nudge. Sits BELOW the prize scale
# (max 120 << 1000) so it only breaks prize-tie develop-vs-stall lines; decays to 0 by round 9 so it never
# double-counts the late prize race (RACE_LEAF / the prize term own the late game). Flag-off => byte-identical.
EARLY_TEMPO = False
EARLY_TEMPO_W = 80.0          # readiness reward at full progress on the intended attacker (in the ~260 early band)
EARLY_TEMPO_TURN_FULL = 6     # full weight at/under this many rounds-per-player
EARLY_TEMPO_TURN_ZERO = 9     # linear decay to 0 by here (no late double-count)
EARLY_TEMPO_KO_BONUS = 40.0   # light prize-pressure: board that can take a KO next turn
# PRIZE_RACE_TEMPO (teacher-shaped, prize_race-deck-ONLY): the Archaludon diagnosis — piloted by our
# dominion the deck is field 0.564 (WORSE than froslass 0.644) and matched-state agreement vs the strong
# the reference entrant is 0.33 (ATTACH 0.04 / ATTACK 0.13 = we UNDER-tempo). The existing prize-race terms
# are INERT for this deck: _race_leaf_terms returns 0 for prize_race (fires only stall/mill), and
# RACE_LEAF/WINCON_LEAF encode option-value + matchup-urgency, NOT attach-toward-attacker / press-for-KO /
# hold-the-lead. So a NEW prize_race tempo/lead leaf: (a) ATTACKER-READINESS (energy progress toward our
# most-ready attacker's cost, reuse _attacker_progress — NOT turn-gated, unlike EARLY_TEMPO: the teacher
# keeps fueling T13-18), (b) KO-NOW (reuse _ko_potential, lethal-only), (c) PRIZE-LEAD-HOLD (reward a
# POSITIVE prize lead, capped). FROSLASS-SAFE: returns 0 unless _my_win_condition()=="prize_race", so a
# config with this flag ON is still leaf-value-IDENTICAL for stall/mill decks. Whole term clamped < one
# prize (PRACE_CLAMP < VALUE_BASE["prize"]=1000) so it shapes ties, never flips a prize-decisive ordering.
# Flag-off => byte-identical to the dominion (test_pracetempo_byte_identical.py).
PRIZE_RACE_TEMPO  = False
PRACE_READY_W = 70.0          # (a) energy progress toward our attacker's attack-cost (NOT turn-gated)
PRACE_KO_W    = 90.0          # (b) can take a prize THIS turn (dominant sub-term: agreement gap largest on ATTACK)
PRACE_LEAD_W  = 45.0          # (c) per-prize bonus for holding/extending a POSITIVE lead (capped at 3)
PRACE_CLAMP   = 700.0         # whole-term ceiling < one prize (1000) — never flips a prize ordering
# DENY_EVOLUTION (deck-agnostic opponent-side leaf term): the FIELD is ~83% evolution-dependent, yet the
# leaf is STAGE-BLIND and self-focused — the opponent enters value ONLY via prize count, aggregate op_hp
# (stage-blind sum), current-active burst, and hand size. There is ZERO term for KOing/softening an
# opponent PRE-EVOLUTION basic to DENY the evolved attacker (e.g. Riolu(80HP) active t1 -> Mega Lucario
# ex 3-prize t3 — the loser could 1-shot the Riolu). This term rewards damage ALREADY removed from a
# deniable opponent basic, scaled by the belief-inferred evolved form's prize-tier (1/2/3) and evolution
# imminence. Distinct from RICH_EVAL (self potentials), EARLY_TEMPO (our readiness), threat_envelope (our
# exposure): it is the ONLY term that reads opponent stage/evolvesFrom/future-form prize-tier. SAFETY: it
# is a sub-prize tie-breaker (whole term clamped < one prize=1000) that NEVER flips a prize-decisive
# decision, and credits only damage ALREADY on the opp basic (no self-KO/over-extension incentive of its
# own; the existing OWN_PRIZE_RISK/SATURATE_VALUE penalties net out our forward exposure). Belief-gated:
# contributes 0 unless the repaired posterior is non-fallback AND >= DENY_EVO_MIN_CONF. Flag-off =>
# byte-identical to the dominion.
DENY_EVOLUTION        = False
DENY_EVO_W            = 70.0    # per-prize-tier value of a board where the evolved attacker is DENIED (<< 1000)
DENY_EVO_ACTIVE_BONUS = 1.4     # multiplier when the denied basic is the opp ACTIVE and 1-shottable now
DENY_EVO_MIN_CONF     = 0.15    # belief-posterior confidence floor to trust the evolved-form identity
# POLICY_PRIOR (Layer A — deck-agnostic pro-CONSENSUS PRIOR over actions): a base layer of "simply
# good / best moves" that nearly all strong pilots agree on, FROM WHICH the maxmin SEARCH (B) and the
# belief-narrowed opp node (C) DEVIATE only when their value argument beats the prior bonus. The prior is
# a pure ADDITIVE BONUS, never a hard constraint — so its MAGNITUDE is the deviation threshold (a move
# whose minimax value beats the consensus move by more than the bonus still wins => "deviate when the
# search says so"; a belief-narrowed real opponent that punishes the consensus move drops its minimax
# value below the bonus => "deviate on Strong Belief"). TWO bounded, override-able mechanisms:
#   (1) PRUNE-SURVIVAL: in _candidates(), ADD POLICY_PRIOR_PRUNE_BONUS (per matching rule) to the H._score
#       sort key for consensus-matching options so they SURVIVE the TOP_M prune and reach the leaf (fixes
#       the ~39% prune side the diagnosis localized at _candidates(): the passive H._score drops the
#       aggressive/consensus move below TOP_M BEFORE search). Ordering-only: it changes which moves are
#       SEARCHED, never the value of a searched move — minimax still picks the true-value argmax among the
#       (now consensus-inclusive) candidate set. Bounded so it cannot force a genuinely bad move to win.
#   (2) PRIZE-EQUAL LEAF TIEBREAKER: in _state_value, ADD a bonus for consensus-compliant resulting board
#       FEATURES (a fueled attacker = attach-to-attacker; a developed bench = bench-develop), GATED to
#       fire ONLY when the move does NOT change either prize count (prize-equal), so it is decisive among
#       prize-tied moves but can NEVER flip a prize-decisive choice (the one rule consensus pros never
#       break: do not give up a prize for a soft prior). Whole term clamped < one prize (POLICY_PRIOR_CLAMP
#       << VALUE_BASE["prize"]=1000).
# DECK-AGNOSTIC: helps FROSLASS now (the ship deck / primary gate). Detectors mirror the offline mine
# (consensus_mine2.detect) on the SDK observation. Flag-off => byte-identical to the dominion
# (test_policyprior_byte_identical.py: 0 mismatches on a froslass+archaludon corpus when off).
# NOTE on the mine: the offline mine found only no_idle_pass is a >=0.85 near-unanimous rule AND dominion
# already complies (0.999) — the other four are multi-step frame-sampling artifacts. So tuning may find NO
# addressable froslass gap (the honest NULL). The infra is built per directive; per-rule weights default 0
# for the artifact rules so ON-with-defaults is a safe no-op on those, and the leaf is prize-equal-gated.
POLICY_PRIOR = False
# per-rule prune-survival bonuses (added to H._score so the option survives the TOP_M sort). The mine's
# qualifying rule is no_idle_pass (don't END with development available); the others are kept tunable but
# default 0 (frame-sampling artifacts — tuning decides if any nonzero helps froslass). Sized as a fraction
# of a typical H._score gap (~tens-hundreds) so consensus moves clear TOP_M without dominating real value.
POLICY_PRIOR_PRUNE = {
    "no_idle_pass":       0.0,   # an END option when development remains: PENALIZE end / boost non-end
    "bench_develop":      0.0,   # play a Basic to an open bench slot
    "attach_to_attacker": 0.0,   # attach energy to a not-yet-payable in-play attacker
    "take_ko":            0.0,   # an attack whose base damage >= defender current HP (prize-CHANGING — see leaf gate)
    "evolve_better":      0.0,   # evolve into a strictly-higher-HP form
    # T3 mechanism 2 (2026-07-16 diagnosis, memory project_ptcg_t3_attach_refuted; the OTHER real
    # mechanism besides ATTACH_FIRST — mechanism 1, shipped). Measured on the real engine: the
    # 1300-peak/1100-sustained Crustle reference entrant sends 57.8% [52.6,62.8] of energy
    # attaches to the BENCH; we send 9.5-34.2% depending on deck/context (CIs non-overlapping).
    # Cause: attach_active +50 (heuristic_agent.py POLICY_W) scores the active target ~50 points
    # above an equally-eligible bench target, so under DIVERSE_CANDS=False (the shipped default)
    # _candidates()'s plain idx[:TOP_M=3] sort virtually always fills the TOP_M slots with the
    # active-target attach before a same-turn bench-target attach gets a look — measured: no bench
    # attach reaches the search on 57.1% of attach-legal frames. A naive fix (zeroing attach_active
    # in heuristic_agent.py) FAILED: it did not raise bench share (34.2%->21.0%), because the
    # active/bench tie then breaks on stable-sort order, not value. This rule instead gives a
    # BENCH-targeted not-yet-payable attach (bench_attach, a subset of attach_to_attacker restricted
    # to inPlayArea==BENCH) its own prune-survival bonus, independent of attach_to_attacker's weight
    # (kept 0.0 above) — so it can compete for a TOP_M slot on ITS OWN sort-key merit without
    # depending on the active-target rule ever being turned on. Ordering-only (see the mechanism doc
    # above _candidates()): raises which moves are SEARCHED, never a searched move's value — minimax
    # still picks the true argmax among the (now bench-inclusive) candidate set.
    "bench_attach":       0.0,
    # SEQUENCING (2026-07-21, Kaggle discussion review of top-1%-competitive-player advice, memory
    # project_ptcg_plan_top_elo W0): draw-before-tutor when both are co-legal in the same MAIN
    # decision. W0 code audit found this is a REAL gap, not a moot point — MAIN decisions do offer
    # ABILITY/PLAY together single-select, and the flat base weights actually run BACKWARDS from the
    # pro rule (PLAY=350 > ABILITY=300, an unrelated weight-tuning artifact, not deliberate), so a
    # tutor would out-score a bare draw by default today. Same shape as bench_attach: a
    # prune-survival bonus on the draw option so it can compete for a TOP_M slot on its own
    # sort-key merit, ordering-only, never changes a searched move's value.
    "sequencing_draw_first": 0.0,
    # HIGH_VALUE_ABILITY (2026-07-21, memory project_ptcg_ability_divergence): corpus-scale finding
    # (OCEL W1, N=82k/24k Crustle-vs-Crustle, CI-separated) -- the SAME Crustle decklist teacher
    # that entrant uses ABILITY at 14.5% of MAIN decisions vs our 7.6%, a residual that survives
    # after controlling for deck composition (validated byte-exact against raw replay JSON). Mechanism
    # (code-verified, no engine needed): Mega Kangaskhan ex's "Run Errand" (draw 2 cards, the deck's
    # only activatable ability) is correctly scored +44 by H._skill_value() when COMPREHEND=True, but
    # POLICY_W["ability"]=300 vs POLICY_W["play"]=350 is a flat 50-point gap -- an unrelated
    # weight-tuning artifact, same class as sequencing_draw_first's PLAY>ABILITY finding above -- so
    # even WITH the +44 bonus (344 total) the ability still loses a same-turn bare PLAY option (350)
    # for a TOP_M slot. This likely explains why COMPREHEND's own prior field A/Bs came back flat: the
    # skill score can be right and still not move behavior because an unrelated weight sits on top.
    # SCOPED FIX (not a COMPREHEND flip -- that flag's own field-test history is genuinely mixed, see
    # the memory note): give any currently-legal ABILITY option whose H._skill_value() clears
    # HIGH_VALUE_ABILITY_MIN its own prune-survival bonus, INDEPENDENT of the COMPREHEND flag's
    # on/off state (this rule calls _skill_value() directly, not through COMPREHEND's gate) and
    # independent of sequencing_draw_first (which only fires when a draw AND a tutor are BOTH
    # co-legal) -- this is strictly broader: fires whenever a genuinely strong ability is competing
    # for a slot against anything, not only a tutor. Ordering-only: raises which moves are SEARCHED,
    # never a searched move's value -- minimax still picks the true argmax among the (now
    # ability-inclusive) candidate set.
    "high_value_ability": 0.0,
    # S2 — MATCHUP_PRUNE rules (2026-07-22, report/hariyama_loss_anatomy_20260722.md). Both fire
    # ONLY when the MATCHUP_PRUNE flag is ON and _matchup_prune_gate() passes (belief-MAP archetype
    # label contains MATCHUP_PRUNE_FAMILY_KW, non-fallback, conf >= MATCHUP_PRUNE_MIN_CONF — the
    # OPP_DECK_MODEL/DENY_EVO confidence convention). Deck-GENERAL detectors (card-DB weakness +
    # prize-tier lookups, NO card names hardcoded); the archetype gate only SCOPES where they act.
    # (a) matchup_avoid_feed — intended weight NEGATIVE (a penalty): an ATTACH/ENERGY option whose
    #     target is weakness-doubled by a FUELED opponent in-play attacker's type AND concedes >=2
    #     prizes on KO. The measured loss chain (86% of Hariyama-family losses): fueling our
    #     Fighting-weak 3-prize Mega Kangaskhan ex into a mono-{F} board of 320-540 one-shots —
    #     every energy committed there is deleted with the body, and the KO is 3/6 of their prizes.
    #     Ordering-only: the penalty drops the feed-attach out of the TOP_M set so the search
    #     spends its slots on non-feed lines; it never changes a searched move's value.
    # (b) matchup_snipe — weight POSITIVE: an ATTACK option that ONE-SHOTS (raw damage >= current
    #     HP, the take_ko convention) a LOW-HP defender (hp <= MATCHUP_SNIPE_HP_MAX). The win
    #     template in our own games: Crustle's 120 one-shots the 110-HP Solrock/Lunatone support
    #     package and finishes Hariyama — support-snipe rate 1.10/game in WINS vs 0.24 in LOSSES.
    #     Subset of take_ko (kept 0.0 above — this rule is matchup-scoped, that one is global).
    "matchup_avoid_feed": 0.0,
    "matchup_snipe":      0.0,
}
# floor for _skill_value(cd) to count as a "high value" ability for the rule above -- calibrated to
# Run Errand's 44 (draw 2 cards), excludes a bare single-card generic draw (30) and a non-Pokémon
# tutor (25) so the rule stays scoped to genuinely strong abilities, not any activatable ability.
HIGH_VALUE_ABILITY_MIN = 40.0
# S2 — MATCHUP_PRUNE (2026-07-22, report/hariyama_loss_anatomy_20260722.md surgery candidate S2):
# master switch for the two matchup-gated POLICY_PRIOR_PRUNE rules above. INDEPENDENT of the
# POLICY_PRIOR flag (same precedent as high_value_ability's COMPREHEND-independence): the prune-bonus
# path in _candidates() runs when EITHER flag is on, so this lever can be screened standalone without
# dragging in POLICY_PRIOR's prize-equal leaf tiebreaker + its WARM_START-cache exclusion (a compute
# confound, the 07-06 A/B class). Ordering-only by construction — both rules live exclusively in the
# _consensus_option_idx/_policy_prior_prune_bonus sort-key path, never in _state_value. Flag-off =>
# byte-identical (rule sets are never emitted; weights also default 0.0 — double-guarded).
MATCHUP_PRUNE = False
MATCHUP_PRUNE_MIN_CONF = 0.15    # posterior-confidence floor (OPP_DECK_MODEL_MIN_CONF convention)
MATCHUP_PRUNE_FAMILY_KW = "lucario"  # belief-MAP label keyword gate (ARCH_PRIZE_RATE_KW convention);
                                     # config-overridable so other families can reuse the machinery
MATCHUP_SNIPE_HP_MAX = 130.0     # defender current-HP ceiling for the snipe rule (covers the 110-HP
                                 # Solrock/Lunatone band + a chipped 150-HP Hariyama)
# --- SUMO_EXPOSURE + GUST_SNIPE_PRIORITY (2026-07-23, anti-Hariyama threat rules) ----------------
# Intel source: the 07-22 pro-scene sweep + report/hariyama_loss_anatomy_20260722.md (#1 confirmed
# ladder deficit, 0.367 [0.269,0.477] n=79). ⚠️ ENGINE-VERIFIED against EN_Card_Data / cg.api —
# the translated-JP intel PARTLY did not verify, encode ONLY what did:
#   • VERIFIED: Hariyama (674, non-ex, HP150, {F}, weak {P}) skill "Heave-Ho Catcher" — an
#     ON-EVOLVE gust: "when you play this Pokémon from your hand to evolve 1 of your Pokémon …
#     Switch in 1 of your opponent's Benched Pokémon to the Active Spot." NOT the claimed
#     every-turn "Sumo Catch" ability (no such card/skill in the engine DB). Exposure model:
#     our bench is pull-reachable while they hold an IN-PLAY, not-yet-evolved pre-evolve
#     (Makuhita 673) — the engine-true precondition for the next Heave-Ho (losses show
#     forced_swaps 1.4/game, >=1 in 82% of losses).
#   • VERIFIED: Wild Press 210 ({F}{F}{F}, 70 self-dmg): x2 into our Fighting-weak 3-prize Mega
#     Kangaskhan ex (756, HP300, weak {F}) = 420 = OHKO; Mega Lucario ex Mega Brave 270 -> 540.
#     MKang KO'd >=1 in 86% of losses (median first-OHKO turn 6) — the don't-expose-MKang rule
#     is the highest-value part of this lever.
#   • VERIFIED: Lunatone (675, HP110) "Lunar Cycle" draw-3 needs Solrock in play; Solrock (676,
#     HP110) Cosmic Beam needs Lunatone — sniping EITHER breaks both (their thin draw engine).
#   • VERIFIED: our Crustle (345) "Mysterious Rock Inn" prevents ALL damage from opponent
#     Pokémon-ex — ONLY their non-ex Hariyama line can damage the wall at all, so killing
#     Makuhita pre-evolve / Hariyama renders the wall INVINCIBLE to this deck (their_Hariyama
#     KO'd >=1 in only 42% of losses — under-prioritized today).
#   • DID NOT VERIFY: "Sumo Catch" (every-turn ability) — encoded as on-evolve instead;
#     "Wave Strike recycles energy" — no such attack in the DB; the family's energy recycler is
#     Mega Lucario ex "Aura Jab" (attach up to 3 {F} from discard), so energy-denial on the
#     FAMILY is still low value, but the mechanism is Aura Jab, not a Hariyama attack.
# SUMO_EXPOSURE (leaf term + CE gust-discount override): while pull-exposed (family gate + an
# in-play pre-evolve of an on-evolve-gust evolution, card-DB-derived — no hardcoded names),
# treat our MULTI-prize benched pieces their board (incl. the PROJECTED evolved form of the pull
# piece itself: Makuhita's energies + Hariyama's Wild Press) can OHKO as active-exposed.
# 🔴 BOUNDARY (user directive): may change WHICH piece we bench / when — must NEVER suppress
# benching itself (never_benched = the #1 loss predictor; benchless-KO = 62%/32% of losses).
# Enforced by construction: the penalty scales with (prizes_for_ko − 1), so a 1-prize body — and
# an EMPTY bench — always contribute exactly 0; only multi-prize placement choices re-rank.
# Flag-off => byte-identical (term never called; CE discount expression unchanged).
SUMO_EXPOSURE = False
SUMO_EXPOSURE_W = 300.0    # per prize ABOVE 1 of the most-exposed benched piece (MKang worst case
                           # 2*300=600 < one prize 1000 — shapes placement, never flips the race)
_SUMO_CE_GUST = 0.6        # STRUCTURAL (not a swept tunable, the RISK_SLOW_CUTOFF convention): CE
                           # gust-discount override while pull-exposed — the pull needs no unseen
                           # gust CARD, only the evolve they hold (~4-copy line, 1.4 swaps/game)
_SUMO_ROOT = [False]       # per-decision armed cache (agent() sets; leaves read — the _CE_ROOT_SLOW
                           # convention; reset EVERY decision so a stale gate never leaks)
# GUST_SNIPE_PRIORITY (OUR gust-target valuation, additive archetype-conditional term — never a
# rewrite of H._gust_target_value's exact-arithmetic bands): vs the Hariyama family prioritize
#   pre-evolve snipe (Makuhita before it evolves)  >  draw-engine piece (Lunatone/Solrock)  >
#   Hariyama itself (no bonus).
# Deck-general detectors: (1) target's name is a pre-evolve of an on-evolve-gust evolution (the
# same card-DB map as SUMO_EXPOSURE); (2) target has a draw-text skill OR is NAMED in another
# in-play opponent piece's skill/attack text (the Lunatone<->Solrock pair coupling). Bonuses sit
# BELOW the +8000 KO-now band (a guaranteed-prize gust is never displaced) and above the 0.5*hp
# tiebreak. The flag lives on S (coevo genome is S-keys only); H reads only the per-decision
# armed cell H._GS_ARMED (seat-scoped: the MIN node's simulated opponent gusts are NOT skewed).
# Flag-off => byte-identical (cell stays None; H's gust expression unchanged).
GUST_SNIPE_PRIORITY = False
GUST_SNIPE_PREEVO_W = 2000.0   # pre-evolve snipe bonus (dominates the +3000-passenger sub-order
                               # and the ±1000 KO-back band by design; << the 8000 KO-now band)
GUST_SNIPE_ENGINE_W = 800.0    # draw-engine piece bonus (Lunatone/Solrock tier, below pre-evolve)
# per-rule leaf tiebreaker weights (added to _state_value, PRIZE-EQUAL gated). Only the prize-NEUTRAL,
# state-observable rules get a leaf term (attach-to-attacker => attacker_progress; bench_develop => bench
# width). take_ko is prize-CHANGING (pe_safe_frac=0 in the mine) so it gets NO leaf term — it can never be
# a prize-equal tiebreak. no_idle_pass / evolve_better are move-shaped (END vs not / which evolve), not a
# stable board feature, so they live in the PRUNE mechanism only.
POLICY_PRIOR_LEAF = {
    "attach_to_attacker": 0.0,   # * best _attacker_progress over in-play (reward a fueled-toward-cost attacker)
    "bench_develop":      0.0,   # * benched-count (reward a developed bench), capped
}
POLICY_PRIOR_LEAF_BENCH_CAP = 3  # cap the bench_develop leaf credit (don't reward hoarding > this many)
POLICY_PRIOR_CLAMP = 700.0       # whole leaf term ceiling < one prize (1000) — never flips a prize ordering
# SATURATE_VALUE (bounded ~P(win) leaf): the rating is BINARY (win/loss; margin does NOT affect Elo), so
# lead beyond a win is worthless and value should SATURATE — and as we NEAR a win we should protect the
# worst case (maximin) more deliberately ("before we OVERWIN, account for worst-case states"). Implemented
# BELOW the prize scale (the dominant prize term is UNTOUCHED — saturating it would make prizes stop
# dominating and is a bad search guide, the documented leaf-confound): when we are AHEAD and our active is
# exposed to the opponent's full next-turn burst (_envelope_can_ko), subtract a penalty that GROWS as our
# remaining prizes -> 0. "Don't throw away a near-win" thus EMERGES from the value, not a hand-coded stance.
# Flag-off => byte-identical.
SATURATE_VALUE = False
LEAD_PROTECT_W = 300.0     # < one prize (1000): worst-case-when-ahead penalty unit (per prize at risk)
STARTING_PRIZES = 6
# PIVOT_VALUE (2026-07-29, the ATTACK/RETREAT gap): against Elo>=1050 players our MAIN mix is
# ATTACK 19.8% vs their 12.5% and RETREAT 0.8% vs their 3.7%, while PLAY/ATTACH/EVOLVE match within
# ~2pp — the one deck-independent behavioral gap the strong-player mining found. MECHANISM, not
# quota: the leaf is position-blind — a retreat changes neither prizes nor HP, so minimax sees zero
# value in repositioning even though RETREAT reaches the candidate set (DIVERSE_CANDS). This term
# penalizes a board whose ACTIVE is strictly out-positioned by a benched body against the current
# opponent active (damage via _ce_attack_damage: payable-within-one-attach, weakness- and
# ex-immunity-aware). Full weight only when the bench body crosses the KO BREAKPOINT the active
# cannot (the Dunsparce lesson: chip credit pushes premature attacks), a small PIVOT_CHIP fraction
# for large non-KO gaps. After an actual pivot the penalty vanishes on the post-state, so the
# retreat line gains exactly this much leaf value — and an attack from the out-positioned active
# keeps paying it, which moves BOTH shares in the observed direction. Sub-prize (250 < 1000) so it
# shapes ties, never flips a prize-decisive ordering; under WIN_VALUE it scales by the local prize
# gradient like every other sub-prize term. NO validated claim that closing this gap raises Elo
# (the compliance lesson) — flag ships OFF, the ladder decides. Flag-off => byte-identical.
PIVOT_VALUE = False
PIVOT_W = 250.0            # penalty when a benched body crosses a KO line the active cannot
PIVOT_CHIP = 0.3           # fraction of PIVOT_W for large non-KO damage gaps (>= 40 dmg)
# OWN_PRIZE_RISK (rules-audit rank-1, the LIVE loss cause): the PLAIN leaf scores our active as a big HP
# ASSET with NO offsetting prize-LIABILITY — so a 310/330-HP Mega ex (megaEx = 3 prizes) is over-exposed,
# and at depth-1 + self-copy the search never sees a cheap 1-prize attacker converting our 3 prizes. This
# is the "out-prized while out-damaging" shape (verified in 3 ladder losses). ALWAYS-ON (independent of the
# other OFF flags): if our active can be KO'd next turn by the opponent's best fueled attack, discount by
# the prizes we'd concede × prizes_for_ko (3 for Mega ex). Sub-prize weight => SHAPES ties, never flips the
# prize race. Flagged for ablation; field-gated (mirror can't expose it). Risk: over-passivity vs aggro
# (the refuted OPP_THREAT mode) — that's exactly what the field A/B must rule out before shipping.
OWN_PRIZE_RISK = False
OWN_PRIZE_RISK_W = 300.0
# RICH_STATE (audit #6): read DISCARD recoverable-energy (a deck-general re-fuel resource the leaf ignores).
# NOTE tools + maxHp are ENGINE-BAKED into pk.hp/maxHp (api.py:343 "Current Max HP" already includes attached
# tool HP), which the agent reads — so no separate tool/maxHp term is needed; only discard was truly unread.
RICH_STATE = False
RICH_STATE_W = 2.0
# CONCENTRATION (VALUE_REJECTION_ANATOMY, n=725 teacher-disagreement frames, z up to 9.8): the ONE measured
# value-gap where our leaf sign-CONFLICTS with strong pilots — pre-registered there and, until now, NEVER
# BUILT. On disagreement frames the EXPERT post-state (vs our search pick) had MORE energy on the active
# attacker (my_active_energy +0.119, z 9.6), MORE energy pre-loaded on a bench successor (my_bench_tot_energy
# +0.105, z 8.5), and FEWER benched Pokemon (my_bench_count -0.146, z -9.4; benched-ex -0.145): experts
# CONCENTRATE, we DISPERSE — strongest in the MID-GAME we lose. Every existing bench term
# (RICH_EVAL/STAGE_EVAL/POLICY_PRIOR bench_develop) rewards bench WIDTH with a + sign — the exact opposite
# sign. This term RE-SIGNS bench value: reward a fueled active + ONE fueled successor, penalize UNFUELED
# width beyond a safety floor. Distinct from BENCH_FORCE (a benching RULE) — this is a VALUE re-sign, and
# distinct from EARLY_TEMPO (turn-gated, single-attacker, no width sign). DECK CAVEAT: the regression was
# mined on the CRUSTLE teacher; froslass is a spread deck where width may be correct — so this is flag-OFF,
# screened offline for decision-quality (>=5% argmax-flip-toward-expert, the pre-registered gate) then
# floor-protected FIELD A/B before any ship. Whole term clamped < one prize (never flips a prize decision).
# Flag-off => byte-identical (test_concentration_byte_identical.py).
CONCENTRATION      = False
CONC_ACTIVE_W      = 90.0    # (a) energy progress on our ACTIVE attacker toward its cost (the +0.119 signal)
CONC_SUCCESSOR_W   = 70.0    # (b) energy progress on the single most-ready BENCH successor (the +0.105 signal)
CONC_WIDTH_W       = 60.0    # (c) per-Pokemon penalty for UNFUELED benched width above the floor (the -0.146 sign-fix)
CONC_BENCH_FLOOR   = 2       # keep this many unfueled backups FREE (never fights BENCHLESS_LETHAL); penalize the 3rd+
CONC_CLAMP         = 600.0   # whole-term ceiling < one prize (1000) — shapes prize-tied lines, never flips a prize
# NEXT_EXEC (DRAW-LUCK leaf, ported from the _msx/sub5_dominion lineage — built + unit-tested there,
# flag-off, stranded off the frontier). THE missing "anticipated draw luck" capability: the leaf is
# evaluated at END-OF-OUR-TURN and our own deck is a FIXED deck.csv order in _predict (vary=True shuffles
# only the OPPONENT deck), so a setup line is valued as if we ALWAYS draw the top of deck — a partial
# combo with 4 copies of the missing piece still in deck scores IDENTICALLY to one with 0 copies left.
# NEXT_EXEC values OUR NEXT TURN's executability at the leaf: a bounded PENALTY for option-starved states
# (no payable attack / no fuelable backup / no attach line), each weighted by the OWN-DECK HYPERGEOMETRIC
# P(the next draw fixes it) over our KNOWN remaining pool (deck+prizes = MY_DECK multiset minus hand,
# discard, in-play). So "2 of 3 combo pieces + 4 of the 3rd still in deck" earns credit a fixed-order leaf
# denies. Total capped at NEXT_EXEC_CAP < one prize => shapes ties, never flips the prize race. Flag-off =>
# byte-identical (test_nextexec_byte_identical.py). Leaf-only port (the _msx widening + root prior are a
# later add if the leaf term earns a field slot).
NEXT_EXEC          = False
NEXT_EXEC_ATTACK_W = 600.0   # no payable attack next turn (weighted by 1 - P(draw an energy that fixes it))
NEXT_EXEC_BACKUP_W = 250.0   # no fueled-or-fuelable bench backup (halved when the active isn't threatened)
NEXT_EXEC_ATTACH_W = 150.0   # no energy-attach line in hand (weighted by 1 - P(draw energy))
NEXT_EXEC_CAP      = 900.0   # hard total ceiling (< one prize=1000 => prize-decision-safe)
_WIN_CONDITION = [None]   # our win-condition, classified ONCE from MY_DECK (cached)
LATE_CONVERSION = {
    "enabled": False,            # OFF => byte-identical baseline; arena flips on + tunes
    "stage_trigger": 1,          # apply when _game_stage >= this (1=mid/prizes<=4, 2=late/prizes<=2)
    "even_or_ahead_only": True,  # only press when (opp.prize_left - my.prize_left) >= lead_margin
    "lead_margin": 0,            # >=0 = even-or-ahead; >0 requires a real lead before pressing
    "press_ko": 120.0,           # bonus * _ko_potential when the press condition holds (close it out)
    "lethal_setup": 60.0,        # bonus when one attach from lethal (_ko_potential == 0.6)
    "stall_penalty": -40.0,      # penalty when even/ahead late with NO ko threat (discourage hoarding)
    "per_stage_scale": [0.0, 0.6, 1.0],   # scale whole term by stage [early,mid,late]
    "hp_hoard_cap": 0.0,         # 0=off; >0 caps board-HP lead's contribution once we should be closing
}


def _resolve_value_db_path(db_path):
    import os
    cands = [db_path] if db_path else []
    cands += ["arena/value_db.json", "value_db.json",
              "/kaggle_simulations/agent/value_db.json",
              "/kaggle_simulations/agent/arena/value_db.json"]
    if "__file__" in dir():
        here = os.path.dirname(os.path.abspath(__file__))
        cands += [os.path.join(here, "value_db.json"), os.path.join(here, "arena", "value_db.json")]
    for p in cands:
        if p and os.path.exists(p):
            return p
    return None


def load_value_db(db_path=None) -> bool:
    """Load leaf-value params from value_db.json into VALUE_BASE / VALUE_RICH / STAGE_W /
    LATE_CONVERSION (partial: only keys present are overridden; the rest keep shipped values).

    Returns True if a db was applied, False on any failure — on failure the shipped constants stay,
    so the agent degrades to baseline. Called ONLY from main.py when cfg.value_db is set; never at
    import => the leaf eval is byte-identical unless explicitly enabled.
    """
    try:
        import json
        p = _resolve_value_db_path(db_path)
        if not p:
            return False
        with open(p) as f:
            db = json.load(f)
        b = db.get("base")
        if isinstance(b, dict):
            for k in VALUE_BASE:
                if k in b:
                    VALUE_BASE[k] = float(b[k])
        r = db.get("rich")
        if isinstance(r, dict):
            for k in VALUE_RICH:
                if k in r:
                    VALUE_RICH[k] = float(r[k])
        sw = db.get("stage_w")
        if isinstance(sw, dict):
            for k in ("tempo", "board", "ko"):
                if isinstance(sw.get(k), list) and len(sw[k]) == 3:
                    STAGE_W[k] = [float(x) for x in sw[k]]
            if "hand" in sw:
                STAGE_W["hand"] = float(sw["hand"])
        lc = db.get("late_conversion")
        if isinstance(lc, dict):
            for k, v in lc.items():
                if k in LATE_CONVERSION:
                    LATE_CONVERSION[k] = v
        return True
    except Exception:
        return False


MY_DECK = H.MY_DECK
DIAG = H.DIAG  # shared diagnostics dict (search_path / heur_path / rollouts counters)
_BASIC_ID = next((c.cardId for c in H.CARDS.values() if getattr(c, "basic", False)),
                 MY_DECK[0] if MY_DECK else 1)

# GLOBAL meta forward-evolution map (DENY_EVOLUTION): predecessor NAME -> [evolved CardData], over the
# WHOLE card DB (engine `evolvesFrom` holds the predecessor's NAME, api note _frontier_ha L484). Distinct
# from H's MY_DECK-only evolution chain (used for OUR anti-brick benching). Built at import; never READ
# unless DENY_EVOLUTION is on, so building it has NO value effect (parallel to the constants above).
_META_EVOLVES_TO: dict = {}
for _cd in H.CARDS.values():
    _ef = getattr(_cd, "evolvesFrom", None)
    if _ef:
        _META_EVOLVES_TO.setdefault(_ef, []).append(_cd)


def _load_learned():
    import json
    import os
    for p in ("learned_eval.json",
              os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".",
                           "learned_eval.json"),
              "/kaggle_simulations/agent/learned_eval.json"):
        try:
            with open(p) as f:
                m = json.load(f)
            return (m["mean"], m["std"], m["weights"])
        except Exception:
            continue
    return None


_LMODEL = _load_learned()


def _learned_value(state, me: int) -> float:
    """Standardized logistic-regression score (logit of P(win)) — monotonic in win prob, so a
    valid value for ranking search leaves. Terminal overrides handled by the caller."""
    if features is None or _LMODEL is None:
        return 0.0
    f = features.extract(state, me)
    if f is None:
        return 0.0
    mu, sd, w = _LMODEL
    z = 0.0
    last = len(w) - 1
    for i in range(min(len(w), len(f))):
        # NaN/inf guard (review #10): a feature with zero training variance gives sd==0 -> a
        # divide-by-zero that NaN-poisons z and silently mis-ranks every search leaf. Floor sd and
        # drop any non-finite standardized term so a degenerate model degrades, never corrupts.
        if i == last:
            z += w[i]
            continue
        s = sd[i] if (sd[i] and sd[i] > 1e-9) else 1e-9
        term = w[i] * (f[i] - mu[i]) / s
        if term == term and term not in (float("inf"), float("-inf")):   # term==term is False for NaN
            z += term
    return z


def _attacker_progress(pk) -> float:
    """0..1: how close `pk` is to paying its evolution line's top attack (energy tempo)."""
    cd = H.CARDS.get(pk.id)
    if not cd:
        return 0.0
    top = H._line_top_cost(cd)
    if top <= 0:
        return 1.0
    return min(1.0, len(pk.energies) / top)


def _early_tempo_terms(mine, opp, turn) -> float:
    """L1 EARLY attacker-readiness: reward energy progress toward an attack's cost on the ONE
    most-ready intended attacker, scaled to early turns and decaying after. Monotone in readiness;
    sits BELOW the prize scale so it only breaks prize-tie develop-vs-stall lines. NOT a bench-width
    or hand term (refuted as RICH_EVAL), NOT a P(win) calibrator."""
    # turn gate: full weight through EARLY_TEMPO_TURN_FULL rounds-per-player, linear -> 0 by _ZERO.
    # `turn` is the raw engine count (2 plies/round, api.py:368), so compare on rounds = turn/2.
    rounds = max(0.0, (turn or 0) / 2.0)
    if rounds >= EARLY_TEMPO_TURN_ZERO:
        return 0.0
    if rounds <= EARLY_TEMPO_TURN_FULL:
        gate = 1.0
    else:
        gate = (EARLY_TEMPO_TURN_ZERO - rounds) / float(EARLY_TEMPO_TURN_ZERO - EARLY_TEMPO_TURN_FULL)
    # most-ready attacker: max progress over our in-play Pokemon (single intended attacker, NOT a sum
    # over the bench -> does NOT reward wide-but-unfueled boards, the refuted differentiator).
    best = 0.0
    for pk in (([mine.active[0]] if mine.active else []) + [b for b in mine.bench if b]):
        p = _attacker_progress(pk)          # min(1, attached / line_top_cost) in [0,1]
        if p > best:
            best = p
    add = EARLY_TEMPO_W * best * gate
    add += EARLY_TEMPO_KO_BONUS * _ko_potential(mine, opp) * gate    # light KO-next-turn prize-pressure
    return add


def _prize_race_tempo_terms(mine, opp) -> float:
    """PRIZE_RACE_TEMPO: prize_race-deck-ONLY tempo/lead leaf, learned from a top-ranked entrant's Archaludon
    beatdown (attach-to-3-cost-then-swing, hold the lead). OUR perspective only.
    (a) ATTACKER-READINESS: energy progress toward our most-ready attacker's cost (reuse
        _attacker_progress; NOT turn-gated, unlike EARLY_TEMPO — the teacher keeps fueling T13-18).
    (b) KO-NOW: ability to take a prize this turn (reuse _ko_potential; 1.0 payable / 0.6 one-attach).
    (c) PRIZE-LEAD-HOLD: bonus scaled by our prize lead (reward extending/holding +diff, capped at 3).
    Whole term clamped < one prize so it shapes ties, never flips a prize-decisive ordering.
    FROSLASS-SAFE: returns exactly 0.0 for non-prize_race decks (stall/mill) even with the flag ON."""
    if _my_win_condition() != "prize_race":          # FROSLASS-SAFE GATE: 0 for stall/mill decks
        return 0.0
    # (a) attacker-readiness: max progress over our in-play Pokemon (single intended attacker, NOT a
    #     bench sum -> doesn't reward wide-but-unfueled boards, the refuted RICH_EVAL signal).
    ready = 0.0
    for pk in (([mine.active[0]] if mine.active else []) + [b for b in mine.bench if b]):
        p = _attacker_progress(pk)                   # min(1, attached / line_top_cost) in [0,1]
        if p > ready:
            ready = p
    add  = PRACE_READY_W * ready
    # (b) KO-now: prize-pressure (reuse _ko_potential — LETHAL-only, no chip-credit; 1.0 / 0.6 / 0).
    add += PRACE_KO_W * _ko_potential(mine, opp)
    # (c) prize-lead-hold: reward a POSITIVE lead only (don't reward being behind); fewer of OUR
    #     prizes left = ahead. Caps the rewarded lead at 3 so it can't dominate the term.
    lead = len(opp.prize) - len(mine.prize)          # >0 = we are ahead
    if lead > 0:
        add += PRACE_LEAD_W * min(3, lead)
    return max(-PRACE_CLAMP, min(PRACE_CLAMP, add))


def _concentration_terms(mine, opp) -> float:
    """CONCENTRATION: re-sign bench VALUE toward the measured expert posture (see the flag comment).
      (a) ACTIVE FUEL (+):  energy progress on our active attacker toward its cost (reuse _attacker_progress).
      (b) SUCCESSOR (+):    energy progress on the single most-ready BENCH attacker — a MAX over the bench,
                            NOT a sum, so it rewards ONE fueled second wave, never raw width.
      (c) OVER-BENCH (-):   penalize benched Pokemon carrying NO energy beyond CONC_BENCH_FLOOR — the pure
                            board WIDTH the expert avoids. Fires only ABOVE the floor, so a benchless-cliff
                            backup (BENCHLESS_LETHAL needs >=1 bench) is never penalized.
    All sub-prize; whole term clamped to CONC_CLAMP < one prize. Reads only our own board (energies + bench
    occupancy), so it is excluded from the WARM_START (seat,prizes,hp) cache."""
    my_act = mine.active[0] if mine.active else None
    bench = [b for b in mine.bench if b]
    # (a) fuel the active attacker toward its cost
    add = CONC_ACTIVE_W * (_attacker_progress(my_act) if my_act else 0.0)
    # (b) pre-fuel ONE bench successor (max progress over the bench, NOT a sum over width)
    succ = 0.0
    for b in bench:
        p = _attacker_progress(b)
        if p > succ:
            succ = p
    add += CONC_SUCCESSOR_W * succ
    # (c) penalize UNFUELED bench width beyond the safety floor (keep backups, avoid spreading to sitters)
    n_empty = sum(1 for b in bench if not getattr(b, "energies", None))
    if n_empty > CONC_BENCH_FLOOR:
        add -= CONC_WIDTH_W * (n_empty - CONC_BENCH_FLOOR)
    return max(-CONC_CLAMP, min(CONC_CLAMP, add))


# --- NEXT_EXEC (draw-luck) helpers: static state reads, no engine calls (ported from _msx/sub5) ------
def _card_type_of(cid) -> int:
    cd = H.CARDS.get(cid)
    try:
        return int(cd.cardType) if cd is not None else -1
    except Exception:
        return -1


def _is_energy_id(cid) -> bool:
    return _card_type_of(cid) in (5, 6)          # CardType BASIC_ENERGY / SPECIAL_ENERGY


def _is_basic_pkmn_id(cid) -> bool:
    cd = H.CARDS.get(cid)
    return bool(cd is not None and _card_type_of(cid) == 0 and getattr(cd, "basic", False))


def _units_short(have, cost) -> int:
    """How many energy UNITS short `have` is of paying `cost` (0 = payable now). Mirrors H._payable's
    greedy type matching (colorless flexible, RAINBOW wild) but returns the deficit."""
    from collections import Counter
    havec = Counter(int(e) for e in have)
    short = 0
    need_colorless = 0
    for c in cost:
        ci = int(c)
        if ci == H._COLORLESS:
            need_colorless += 1
        elif havec.get(ci, 0) > 0:
            havec[ci] -= 1
        elif any(havec.get(w, 0) > 0 for w in H._WILD):
            for w in H._WILD:
                if havec.get(w, 0) > 0:
                    havec[w] -= 1
                    break
        else:
            short += 1
    rem = sum(havec.values())
    if rem < need_colorless:
        short += need_colorless - rem
    return short


def _best_attack_short(pk):
    """Min units-short over pk's DAMAGING attacks (0 = an attack is payable now); None = no attack."""
    if pk is None:
        return None
    cd = H.CARDS.get(pk.id)
    if not cd:
        return None
    best = None
    for aid in cd.attacks:
        atk = H.ATTACKS.get(aid)
        if atk is None or (getattr(atk, "damage", 0) or 0) <= 0:
            continue
        s = _units_short(pk.energies, atk.energies)
        if best is None or s < best:
            best = s
    return best


# --- LEAF_READY (2026-07-25): the leaf's #1 measured blindness -------------------------------
# The leaf is prize*1000 + hp*1, so a free energy attach is worth EXACTLY 0.0 to the search. On our
# own 2,162 ladder games the ~1305-Elo teacher completes an attack cost when it can 96.9% of the
# time; we do it 57.2% = 1.489 fixable violations/game, the largest single class in the ledger.
#
# WHY AN EPSILON IS ENOUGH. At recurse_depth=1 the leaf is evaluated at END OF OUR TURN (_rollout
# returns _state_value once control leaves us) and the heuristic finishes the turn either way, so
# "attach then attack" and "attack now" usually reach leaves with IDENTICAL prizes and IDENTICAL
# hp-sums — an EXACT tie, resolved today by candidate order. The only difference on the board is
# one more energy, priced at zero. This term prices it, and nothing else.
#
# WHY NOT _attacker_progress (:1017): it is colour-BLIND (divides len(energies) by a COUNT, so 3
# Psychic on a Fighting cost scores 1.0 while being unpayable), it measures progress to the LINE's
# top cost (a Snorunt whose own 1-cost attack IS payable scores 0.33), and it is linear, so the
# completing 2->3 attach and the wasteful 4->5 attach are credited identically. _units_short mirrors
# H._payable's greedy colour matching and returns the true deficit, so it can express "completes".
#
# WHY THIS IS NOT RACE_CLOCK (field-refuted -0.055 [-0.095,-0.015] at N=600): RACE_CLOCK added a
# uniform PRESS bonus that acted on ~11% of decisions and made us more aggressive. This adds no
# aggression, reads only our own board, saturates at payable (no overfill incentive) and is small
# enough that it can never outbid the HP swing of an actual attack.
LEAF_READY = False          # flag-off => byte-identical
LEAF_READY_ACT_W = 120.0    # our ACTIVE can pay a damaging attack on the leaf board
LEAF_READY_NEAR_W = 40.0    # ...or is exactly one unit short (so 2-short -> 1-short is not flat)
LEAF_READY_BENCH_W = 30.0   # at least ONE benched body is payable — a successor, not a width bonus
LEAF_READY_CLAMP = 300.0    # hard ceiling at 0.3 prizes: this must never overrule the prize race


def _leaf_ready_terms(mine) -> float:
    """Sub-prize credit for a board that can actually attack. OWN board only, clamped."""
    v = 0.0
    act = mine.active[0] if mine.active else None
    s = _best_attack_short(act)
    if s == 0:
        v += LEAF_READY_ACT_W
    elif s == 1:
        v += LEAF_READY_NEAR_W
    for b in (mine.bench or []):
        if b is not None and _best_attack_short(b) == 0:
            v += LEAF_READY_BENCH_W       # ONE ready successor; short-circuit, never a width SUM
            break
    if v:
        DIAG["leaf_ready"] = DIAG.get("leaf_ready", 0) + 1     # ship_gate FIRES evidence
    return max(-LEAF_READY_CLAMP, min(LEAF_READY_CLAMP, v))


def _known_pool(mine):
    """(n_energy, n_basic, N) of our UNSEEN cards (deck + face-down prizes) via decklist accounting:
    MY_DECK multiset minus hand, discard, and in-play (mons + pre-evolutions + attached energy/tools).
    Prizes are a uniform random subset of the unseen pool, so the next-draw hypergeometric is exactly
    P(X) = count_X / N. Fallback (deck mismatch, e.g. replaying another sub's game): estimate from the
    SEEN fraction instead, so the term degrades gracefully rather than lying."""
    from collections import Counter
    pool = Counter(MY_DECK)
    seen_ids = []

    def _take(cid):
        if cid is None:
            return
        seen_ids.append(cid)
        if pool.get(cid, 0) > 0:
            pool[cid] -= 1

    for c in (getattr(mine, "hand", None) or []):
        _take(getattr(c, "id", None))
    for c in (getattr(mine, "discard", None) or []):
        _take(getattr(c, "id", None))
    for pk in (list(mine.active or []) + list(mine.bench or [])):
        if pk is None:
            continue
        _take(pk.id)
        for pe in (getattr(pk, "preEvolution", None) or []):
            _take(getattr(pe, "id", None))
        for ec in (getattr(pk, "energyCards", None) or []):
            _take(getattr(ec, "id", None))
        for tl in (getattr(pk, "tools", None) or []):
            _take(getattr(tl, "id", None))
    N = (getattr(mine, "deckCount", 0) or 0) + len(mine.prize or [])
    N = max(N, 1)
    unaccounted = sum(1 for cid in seen_ids if cid is not None and Counter(MY_DECK).get(cid, 0) == 0)
    if MY_DECK and unaccounted <= 4:
        n_e = sum(n for cid, n in pool.items() if n > 0 and _is_energy_id(cid))
        n_b = sum(n for cid, n in pool.items() if n > 0 and _is_basic_pkmn_id(cid))
        return n_e, n_b, N
    # fallback: SEEN-fraction estimate (deck mismatch / no decklist) — INFERRED, coarse but sane
    n_seen = max(len(seen_ids), 1)
    fe = sum(1 for cid in seen_ids if _is_energy_id(cid)) / n_seen
    fb = sum(1 for cid in seen_ids if _is_basic_pkmn_id(cid)) / n_seen
    return int(round(fe * N)), int(round(fb * N)), N


def _next_exec_penalty(mine, opp) -> float:
    """Executability of OUR NEXT TURN at this leaf: bounded penalty for option-starved states, each
    component weighted by the OWN-DECK hypergeometric P(the next draw fixes it). Components: (1) no
    payable attack, (2) no fueled-or-fuelable backup behind the active (second wave), (3) no energy-
    attach line in hand. Total capped at NEXT_EXEC_CAP < one prize. Reads only our own board + decklist."""
    act = mine.active[0] if mine.active else None
    bench = [pk for pk in (mine.bench or []) if pk]
    hand_ids = [getattr(c, "id", None) for c in (getattr(mine, "hand", None) or [])]
    n_e_hand = sum(1 for cid in hand_ids if _is_energy_id(cid))
    n_b_hand = sum(1 for cid in hand_ids if _is_basic_pkmn_id(cid))
    n_e, n_b, N = _known_pool(mine)
    p_e = min(1.0, n_e / N)                     # P(next single draw is an energy)
    p_b = min(1.0, n_b / N)                     # P(next single draw is a basic)
    pen = 0.0
    # (1) CAN-ATTACK next turn
    s_act = _best_attack_short(act)
    s_act = 99 if s_act is None else s_act
    if s_act > 0:
        if s_act == 1:                          # one attach away
            if n_e_hand > 0:
                pen += 0.2 * NEXT_EXEC_ATTACK_W     # line exists in hand — tempo risk only
            else:
                pen += NEXT_EXEC_ATTACK_W * (1.0 - p_e)   # depends on the top-deck draw
        else:
            pen += NEXT_EXEC_ATTACK_W               # >=2 short: cannot attack next turn
    # (2) SECOND WAVE: fueled-or-fuelable backup behind the active
    def _fuelable(b):
        s = _best_attack_short(b)
        return s is not None and s <= 1
    backup = any(_fuelable(b) for b in bench)
    if not backup:
        op_act = opp.active[0] if opp.active else None
        threatened = bool(act is not None and op_act is not None and H._opp_can_ko(op_act, act))
        w = NEXT_EXEC_BACKUP_W * (1.0 if threatened else 0.5)
        if not bench:
            # a basic IN HAND (or a likely draw) covers the cliff only partially (not on board yet)
            cover = 0.6 * (1.0 if n_b_hand > 0 else p_b)
            pen += w * (2.0 - cover)
        else:
            pen += w
    # (3) ATTACH LINE: nothing to attach — the next fuel depends on the draw
    if n_e_hand == 0:
        pen += NEXT_EXEC_ATTACH_W * (1.0 - p_e)
    return min(pen, NEXT_EXEC_CAP)


def _can_1shot(mine, opp_pk) -> bool:
    """True if OUR fueled active can KO `opp_pk` (an opponent in-play Pokemon) THIS turn — same damage
    calc as _ko_potential (weakness x2, resistance -RESISTANCE_VALUE), but targeted at an arbitrary opp
    Pokemon (active OR bench) and requiring the attack to be PAYABLE now (a real 1-shot, not one-attach-
    away). Used by DENY_EVOLUTION to give full denial credit when a deniable basic is 1-shottable now."""
    my_act = mine.active[0] if mine.active else None
    if not (my_act and opp_pk):
        return False
    if getattr(mine, "asleep", False) or getattr(mine, "paralyzed", False):
        return False
    cd = H.CARDS.get(my_act.id)
    if not cd:
        return False
    atk = H._strongest_attack(cd)
    if not atk or atk.damage <= 0:
        return False
    dmg = atk.damage
    cd_t = H.CARDS.get(opp_pk.id)
    if cd_t and cd_t.weakness is not None and cd.energyType == cd_t.weakness:
        dmg *= 2
    if cd_t and cd_t.resistance is not None and cd.energyType == cd_t.resistance:
        dmg = max(0, dmg - (getattr(H, "RESISTANCE_VALUE", 0) or 0))   # SV-era -30 (guard None)
    if dmg < (getattr(opp_pk, "hp", 0) or 0):
        return False
    return H._payable(my_act.energies, atk.energies)


def _deny_evolution_terms(mine, opp, turn) -> float:
    """DENY_EVOLUTION: opponent-side, stage-aware leaf bonus for having DAMAGED a deniable opponent
    pre-evolution basic, scaled by the belief-inferred evolved form's prize-tier (1/2/3) and evolution
    imminence. Belief-gated (non-fallback + conf>=DENY_EVO_MIN_CONF); contributes 0 otherwise. Whole
    term is clamped < one prize so it tie-breaks but never flips a prize-decisive decision. Credits only
    damage ALREADY on the opp basic (no self-KO incentive of its own)."""
    arch = _OPP_ARCH[0]
    # arch = (archetype, conf, fallback) from belief.map_archetype; require a real inference.
    if not (isinstance(arch, tuple) and len(arch) >= 3):
        return 0.0
    conf = arch[1] or 0.0
    fallback = arch[2]
    if fallback or conf < DENY_EVO_MIN_CONF:
        return 0.0
    consist = _OPP_CONSIST_IDS[0]
    if not consist:                       # no belief-consistent particle id-set => can't trust identity
        return 0.0
    rounds = max(0.0, (turn or 0) / 2.0)
    # evolution imminence: a basic that has been in play >=1 turn is eligible to evolve next turn
    # (engine evolve-not-first-turn rule). Proxy with rounds>=1 (turn>=2); scale by posterior confidence
    # that the line completes. Public-state only (no per-turn flags).
    imm = (1.0 if rounds >= 1.0 else 0.0) * conf
    if imm <= 0.0:
        return 0.0
    total = 0.0
    op_active = opp.active[0] if opp.active else None
    for pk in ((opp.active or []) + (opp.bench or [])):
        if pk is None:
            continue
        cd_basic = H.CARDS.get(pk.id)
        if cd_basic is None:
            continue
        # (1) pk must be a deniable PRE-evolution form (a Basic / not the top of its line) ...
        is_pre = bool(getattr(cd_basic, "basic", False)) or (
            getattr(cd_basic, "evolvesFrom", None) is None and cd_basic.name in _META_EVOLVES_TO)
        if not is_pre:
            continue
        # (2) ... whose name has candidate evolved forms in the global meta map ...
        forms = _META_EVOLVES_TO.get(cd_basic.name)
        if not forms:
            continue
        # (3) ... belief gate: keep only forms whose card-id is in the belief-consistent decklists,
        # then take the HIGHEST prize-tier consistent form (most valuable to deny).
        best_tier = 0
        for f in forms:
            if getattr(f, "cardId", None) not in consist:
                continue
            tier = 3 if getattr(f, "megaEx", False) else (2 if getattr(f, "ex", False) else 1)
            if tier > best_tier:
                best_tier = tier
        if best_tier <= 0:
            continue
        # denial realized = fraction of the basic's max HP we've already removed (a KO'd basic is absent
        # from opp.active/bench, so its absence is the full-denial case; a 1-shottable basic gets full
        # credit because the KO is reachable THIS turn).
        max_hp = (getattr(cd_basic, "hp", 0) or 0) or (getattr(pk, "hp", 0) or 0)
        denial = 0.0
        if max_hp > 0:
            denial = max(0.0, (max_hp - (getattr(pk, "hp", 0) or 0)) / float(max_hp))
        if _can_1shot(mine, pk):
            denial = 1.0
        if denial <= 0.0:
            continue
        base = DENY_EVO_W * best_tier * imm * denial
        if op_active is not None and pk is op_active and _can_1shot(mine, pk):
            base *= DENY_EVO_ACTIVE_BONUS   # ACTIVE + 1-shottable now = the textbook denial
        total += base
    # hard ceiling below one prize: a tie-breaker among prize-comparable lines, never a prize-flip.
    return min(total, 0.6 * VALUE_BASE["prize"])


def _game_stage(mine, opp) -> int:
    """Markov game phase from prizes remaining (fewer left = later): 0 early, 1 mid, 2 late."""
    left = min(len(mine.prize), len(opp.prize))
    if left >= 5:
        return 0
    if left >= 3:
        return 1
    return 2


def _ko_potential(mine, opp) -> float:
    """0..1 lethal-range signal: can our active threaten/KO the opponent's active soon?
    Encodes the win condition (KO breakpoints) the flat HP-sum is blind to."""
    my_act = mine.active[0] if mine.active else None
    op_act = opp.active[0] if opp.active else None
    if not (my_act and op_act):
        return 0.0
    if getattr(mine, "asleep", False) or getattr(mine, "paralyzed", False):
        return 0.0   # audit #4: our active is Asleep/Paralyzed -> can't attack this turn -> no KO
    cd = H.CARDS.get(my_act.id)
    if not cd:
        return 0.0
    atk = H._strongest_attack(cd)
    if not atk or atk.damage <= 0:
        return 0.0
    dmg = atk.damage
    cd_op = H.CARDS.get(op_act.id)
    if cd_op and cd_op.weakness is not None and cd.energyType == cd_op.weakness:
        dmg *= 2
    if cd_op and cd_op.resistance is not None and cd.energyType == cd_op.resistance:   # SV-era -30
        dmg = max(0, dmg - H.RESISTANCE_VALUE)
    # LETHAL-only: reward being able to KO the active (now or one attach away). Do NOT reward
    # chip damage — partial credit pushes premature attacks instead of developing (Dunsparce bug).
    if dmg < op_act.hp:
        return 0.0
    return 1.0 if H._payable(my_act.energies, atk.energies) else 0.6


def _opp_blocked(opp) -> bool:
    """The opponent's active can't attack next turn (Asleep/Paralyzed) — nulls our threat predicates
    (audit #4: special conditions). Reads the PlayerState bools the Pokemon-only _opp_can_ko can't see."""
    return bool(getattr(opp, "asleep", False) or getattr(opp, "paralyzed", False))


def _opp_burst_dmg(op_act, my_act, opp_hand_n, my_hand_n=0):
    """Opponent's KO damage on our active next turn, split (static, static+dynamic) — the dynamic part
    is hand/energy scaling (alakazam Powerful Hand = 20×their hand) the value fn is blind to. The HP
    term already prices STATIC threats, so the OPP_THREAT penalty fires only when the DYNAMIC scaling
    is what makes it lethal (static alone wouldn't KO) — isolating the burst-comeback blindness and NOT
    punishing normal attackers (lucario), which over-defends and loses the race. Weakness-aware.

    The two hand counts are NOT interchangeable and the names invert across the call, which is why
    this was long half-wired. Here the ATTACKER is the opponent, so `opp_hand_n` (their hand) is what
    _attack_dynamic calls `my_hand_n` — "for each card in YOUR hand", i.e. Powerful Hand. `my_hand_n`
    (OUR hand) is what it calls `opp_hand_n` — "for each card in your OPPONENT'S hand", i.e.
    Resentful Refrain, 50 x our hand. That fourth argument was never passed, so it defaulted to 0 and
    every opponent Refrain-class attack was priced at zero dynamic damage against us — the one attack
    whose whole threat IS our hand size."""
    cd = H.CARDS.get(getattr(op_act, "id", None)) if op_act else None
    cdm = H.CARDS.get(getattr(my_act, "id", None)) if my_act else None
    if not cd or not cdm:
        return (0.0, 0.0)
    if H._wall_blocks_all(my_act, op_act):  # Rock Inn wall: opp ex attacks do 0 to our active
        return (0.0, 0.0)                   # (transitively nulls _envelope_can_ko). Flag-off ==
                                            # H._ex_immune_vs; ON, a piercing attack survives it.
    walled = H.ROCK_INN_SCOPED and H._ex_immune_vs(my_act, op_act)
    wk = 2.0 if (cdm.weakness is not None and cd.energyType == cdm.weakness) else 1.0
    rz = H.RESISTANCE_VALUE if (cdm.resistance is not None and cd.energyType == cdm.resistance) else 0.0  # SV-era -30
    best_static = best_total = 0.0
    for aid in getattr(cd, "attacks", []) or []:
        atk = H.ATTACKS.get(aid)
        if not atk:
            continue
        if walled and not H._attack_pierces_effects(atk):
            continue                       # this one IS prevented; the piercing sibling is not
        stat = max(0.0, (atk.damage or 0) * wk - rz)
        tot = max(0.0, ((atk.damage or 0) + H._attack_dynamic(atk, opp_hand_n, my_act,
                                                             opp_hand_n=my_hand_n)) * wk - rz)
        if tot > best_total:
            best_static, best_total = stat, tot
    return (best_static, best_total)


def _envelope_can_ko(mine, opp, arch_label) -> bool:
    """Can the opponent KO our active NEXT turn using their FULL realistic potential — best fueled
    attack (incl. dynamic + weakness) + their archetype's probable booster — vs our active's HP?
    This is the belief-aware upgrade of H._opp_can_ko (which only checks their current-energy attack)."""
    my_act = mine.active[0] if mine.active else None
    op_act = opp.active[0] if opp.active else None
    if not (my_act and op_act):
        return False
    # CONDITIONS (audit #4): a Paralyzed opponent active can't attack next turn; Asleep can't attack until
    # it wakes (coin flip) — neither is a reliable lethal threat, so don't over-defend against it.
    if getattr(opp, "paralyzed", False) or getattr(opp, "asleep", False):
        return False
    # EX-IMMUNITY (Rock Inn): our active takes 0 from an opponent ex attacker — the boosters below (arch +
    # ability) amplify a PREVENTED attack, so they must not resurrect the threat past _opp_burst_dmg's 0.
    if H._wall_blocks_all(my_act, op_act):
        return False
    _, tot = _opp_burst_dmg(op_act, my_act, getattr(opp, "handCount", 0) or 0,
                            my_hand_n=getattr(mine, "handCount", 0) or 0)
    if _OPP_AMP_LIVE[0]:                                   # drop the booster threat once it's depleted
        tot += _arch_dmg_bonus(arch_label)
    tot += H._ability_boost(opp)                           # audit #1: opp in-play damage-booster abilities
    eff_hp = (getattr(my_act, "hp", 0) or 0)               # audit #4: our pending poison/burn lowers eff. HP
    if getattr(mine, "poisoned", False):
        eff_hp -= 10
    if getattr(mine, "burned", False):
        eff_hp -= 20
    return tot >= eff_hp


def _prize_plan(mine, opp):
    """Do we KNOW what prizes we can take THIS turn? Enumerate the active-KO available to us now.

    Returns (prizes_here, reaches_lethal, our_rate): prizes_here = prizes from KOing the opp active now
    (2 if it's an ex, 1 else, 0 if we can't KO); reaches_lethal = that KO wins the game (>= our prizes
    left); our_rate = our forward prizes-per-turn estimate (KO now -> that many; one attach from lethal
    -> ~1; else 0.5 developing). This is the explicit 'which prizes can we take' the flat value lacked."""
    ma = mine.active[0] if mine.active else None
    oa = opp.active[0] if opp.active else None
    if not ma or not oa:
        return (0, False, 0.5)
    cd = H.CARDS.get(getattr(ma, "id", None))
    # ROCK_INN_SCOPED: plan the prize off our strongest attack THEIR wall does not stop (flag-off
    # this is exactly H._strongest_attack, and the guard below is exactly H._ex_immune_vs).
    atk = H._strongest_unwalled_attack(cd, oa, ma) if cd else None
    if not atk or (atk.damage or 0) <= 0:
        return (0, False, 0.5)
    dmg = atk.damage
    cdo = H.CARDS.get(getattr(oa, "id", None))
    if cdo and cdo.weakness is not None and cd.energyType == cdo.weakness:
        dmg *= 2
    if cdo and cdo.resistance is not None and cd.energyType == cdo.resistance:   # SV-era -30
        dmg = max(0, dmg - H.RESISTANCE_VALUE)
    payable = H._payable(ma.energies, atk.energies)
    # EX-IMMUNITY (Rock Inn): if THEIR active walls OUR attacker (their Rock Inn vs our ex), our attack does
    # 0 — we can't KO it, so don't plan a prize on it (mirror + vs Crustle/Sylveon; attack with a NON-ex instead).
    can_ko = dmg >= (getattr(oa, "hp", 0) or 0) and payable and not H._wall_blocks_attack(oa, ma, atk)
    prizes_here = (H._prizes_for_ko(oa) if can_ko else 0)   # Mega ex=3/ex=2/else 1 (api.py:476-477)
    reaches_lethal = can_ko and prizes_here >= len(mine.prize)
    if can_ko:
        our_rate = float(prizes_here)              # take 1 (or 2 for ex) this turn
    else:
        our_rate = 1.0                             # developing attacker — neutral baseline (symmetric w/ opp default)
    return (prizes_here, reaches_lethal, our_rate)


def _prize_race_value(mine, opp) -> float:
    """MULTI-TURN, ARCHETYPE-CONDITIONED race assessment: who reaches 0 prizes first?

    our_clock = our_prizes_left / our_rate (from _prize_plan); opp_clock = opp_prizes_left / their
    archetype prize-RATE (aggro/burst race, stall crawl) — the OPPONENT-CATEGORY + OUR-SITUATION
    lookahead. Returns PRIZE_RACE_W·(opp_clock − our_clock): positive when OUR clock is shorter (we win
    the race). A lethal-reaching KO short-circuits to a large (sub-terminal) bonus so search closes."""
    prizes_here, reaches_lethal, our_rate = _prize_plan(mine, opp)
    if reaches_lethal:
        return 3000.0                              # we can win NOW (below the 1e6 terminal, above a prize)
    my_left = max(1, len(mine.prize))
    opp_left = max(1, len(opp.prize))
    opp_rate = _opp_prize_rate((_OPP_ARCH[0] or (None, 0))[0])
    our_clock = my_left / max(0.5, our_rate)       # turns for US to take our last prize
    opp_clock = opp_left / max(0.3, opp_rate)      # turns for THEM, paced by their archetype
    v = PRIZE_RACE_W * (opp_clock - our_clock)
    if prizes_here >= 2:                           # value lining up a multi-prize ex/Mega-ex KO (tempo)
        v += 0.5 * PRIZE_RACE_W
    # CLAMP below a prize so the race SHAPES choices within the prize structure, never flips it.
    return max(-PRIZE_RACE_CLAMP, min(PRIZE_RACE_CLAMP, v))


def _survival_value(mine, opp) -> float:
    """CONTROL-role value: the OPPOSITE of racing — reward SURVIVING and GRINDING the game long so
    our disruption/walls out-resource the opponent. Generic (no deck names): (a) value the opponent
    being FAR from their last prize (drag the game out = our plan), (b) penalise our active being
    EXPOSED to a KO next turn (don't trade into the faster deck), (c) reward bench depth (resilience
    to their KOs). Clamped below a prize so it shapes, never flips, the prize race."""
    v = 0.0
    opp_left = len(opp.prize)
    v += ROLE_SURVIVAL_W * opp_left                      # the more prizes THEY still need, the better for us
    my_act = mine.active[0] if mine.active else None
    op_act = opp.active[0] if opp.active else None
    if my_act and op_act and H._opp_can_ko(op_act, my_act, _opp_blocked(opp)):
        prize_at_risk = H._prizes_for_ko(my_act)   # Mega ex=3/ex=2/else 1 (api.py:476-477)
        v -= ROLE_SURVIVAL_W * 2.0 * prize_at_risk      # exposed active = a trade we want to AVOID as control
    v += 0.5 * ROLE_SURVIVAL_W * min(3, sum(1 for pk in mine.bench if pk))   # bench resilience (cap 3)
    return max(-ROLE_SURVIVAL_CLAMP, min(ROLE_SURVIVAL_CLAMP, v))


def _compute_our_speed() -> float:
    """Our deck's prizes/turn proxy from composition (matchup_role.deck_speed). Computed ONCE; an
    explicit config override (OUR_SPEED set before this runs) wins, since we KNOW our own deck."""
    if OUR_SPEED is not None:
        return OUR_SPEED
    if matchup_role is None:
        return matchup_role.SPEED_NEUTRAL if matchup_role else 1.0
    try:
        # deck_speed's is_ex receives a static card-DB Card (fields .ex/.megaEx, NOT the in-play
        # .id that H._is_ex resolves via CARDS.get(pk.id)) — passing H._is_ex here AttributeErrors
        # on cd.id, is swallowed below, and OUR speed silently collapses to NEUTRAL (role inert or
        # mis-signed: a control deck read as speed 1.0). Use a Card-native multi-prize check.
        return matchup_role.deck_speed(
            MY_DECK, H.CARDS, H._strongest_attack,
            lambda cd: bool(getattr(cd, "ex", False) or getattr(cd, "megaEx", False)))
    except Exception:
        return matchup_role.SPEED_NEUTRAL


def _my_win_condition() -> str:
    """Classify OUR win condition ONCE from MY_DECK's attack text (SelfModel; cached). Deck-general:
    Froslass (Resentful Refrain) -> stall; mill -> mill_deckout; else prize_race."""
    if _WIN_CONDITION[0] is not None:
        return _WIN_CONDITION[0]
    wc = "prize_race"
    if _SM is not None:
        texts = []
        for cid in set(MY_DECK):
            cd = H.CARDS.get(cid)
            for aid in (getattr(cd, "attacks", None) or []) if cd else []:
                atk = H.ATTACKS.get(aid)
                if atk:
                    texts.append(getattr(atk, "text", "") or "")
        try:
            wc = _SM.classify_win_condition(texts)
        except Exception:
            wc = "prize_race"
    _WIN_CONDITION[0] = wc
    return wc


def _race_leaf_terms(mine, opp) -> float:
    """Win-condition-aware tie-breakers (kept BELOW the prize scale so they only break prize ties).
    STALL: reward a large opp hand (our Resentful threat) + the opponent's deck-out clock."""
    wc = _my_win_condition()
    add = 0.0
    odc = getattr(opp, "deckCount", None)
    # cycle-5 REVERTED: gating the deckout term on the race-clock broke the froslass grind gain (it removed
    # a reward that HELPS grind matchups). The deckout harm vs aggro is a LEAF problem (the agent grinds into
    # a being-lethaled loss it can't see), fixed by the sharp worst-case-lethal leaf term (THREAT_ENVELOPE),
    # NOT by suppressing the deckout reward. So: keep the deckout grind reward; aggro safety comes from the leaf.
    if wc == "stall":
        add += RACE_W["resentful"] * (getattr(opp, "handCount", 0) or 0)   # real KO threat, matchup-independent
        if odc is not None:
            add += RACE_W["opp_deckout"] * (STARTING_DECK - odc)           # grind/deckout progress (helps froslass)
    elif wc == "mill_deckout" and odc is not None:
        add += RACE_W["opp_deckout"] * 2.0 * (STARTING_DECK - odc)
    return add


# WIN-CONDITION-AWARE / OPTION-VALUE / MATCHUP-CONDITIONAL LEAF — the highest-EV lever (the leaf is the
# bottleneck, midgame AUC~0.61). Three ideas, all clamped BELOW a prize so they shape ties, never flip the
# prize race: (1) OPTION VALUE / robustness — multiple viable attackers + a ready promotion = hard to punish,
# so if they counter one line you're still on a winning path; (2) MATCHUP-CONDITIONAL win condition (the
# PATIENCE GATE — patience only pays vs slow decks): belief-driven, URGENT vs a hand-scaler (Alakazam
# "×your hand" => finish/disrupt before their hand is lethal), DEFEND vs fast aggro (don't expose a 3-prize
# ex), else PATIENT (grind/stall = RACE_LEAF); (3) tempo toward the win con is LATE_CONVERSION.
WINCON_LEAF = False
WINCON_W = {"option": 55.0, "promotion": 120.0, "opp_hand_urgency": 28.0, "close_now": 250.0, "expose": 200.0}


def _viable_attackers(player) -> int:
    """How many of our in-play Pokemon can attack NOW (a payable damaging attack) — option value."""
    n = 0
    for pk in (list(getattr(player, "active", None) or []) + list(getattr(player, "bench", None) or [])):
        cd = H.CARDS.get(getattr(pk, "id", None)) if pk else None
        if not cd:
            continue
        for aid in (getattr(cd, "attacks", None) or []):
            atk = H.ATTACKS.get(aid)
            if atk and (atk.damage or 0) > 0 and H._payable(getattr(pk, "energies", []) or [], atk.energies):
                n += 1
                break
    return n


def _ready_promotion(player) -> bool:
    """A BENCHED Pokemon that can attack if our active dies — keeps us on a winning path after a KO."""
    for pk in (list(getattr(player, "bench", None) or [])):
        cd = H.CARDS.get(getattr(pk, "id", None)) if pk else None
        if not cd:
            continue
        for aid in (getattr(cd, "attacks", None) or []):
            atk = H.ATTACKS.get(aid)
            if atk and (atk.damage or 0) > 0 and H._payable(getattr(pk, "energies", []) or [], atk.energies):
                return True
    return False


def _opp_is_hand_scaler(opp) -> bool:
    """Opponent has an in-play attacker scaling with THEIR OWN hand (Alakazam "×your hand") — their win
    condition is a growing hand, so finish/disrupt BEFORE it's lethal (user: 'before unavoidable gg')."""
    for pk in (list(getattr(opp, "active", None) or []) + list(getattr(opp, "bench", None) or [])):
        cd = H.CARDS.get(getattr(pk, "id", None)) if pk else None
        for aid in (getattr(cd, "attacks", None) or []) if cd else []:
            atk = H.ATTACKS.get(aid)
            t = (getattr(atk, "text", "") or "").lower() if atk else ""
            if "for each card in your hand" in t and "opponent" not in t:
                return True
    return False


def _matchup_mode(mine, opp, arch) -> str:
    """Belief-driven leaf objective: 'urgent' vs hand-scalers, 'defend' vs fast aggro, else 'patient'."""
    if _opp_is_hand_scaler(opp):
        return "urgent"
    if _opp_prize_rate(arch) >= 1.3:
        return "defend"
    return "patient"


def _wincon_leaf_terms(mine, opp) -> float:
    add = WINCON_W["option"] * min(_viable_attackers(mine), 3)       # (1) robustness: viable attackers
    if _ready_promotion(mine):
        add += WINCON_W["promotion"]                                 #     + a ready backup
    arch = (_OPP_ARCH[0] or (None, 0))[0]                            # (2) matchup-conditional win condition
    mode = _matchup_mode(mine, opp, arch)
    if mode == "urgent":            # hand-scaler: their hand -> lethal; penalise the wait + value closing
        add -= WINCON_W["opp_hand_urgency"] * (getattr(opp, "handCount", 0) or 0)
        add += WINCON_W["close_now"] * _ko_potential(mine, opp)
    elif mode == "defend":          # fast aggro: don't expose our 3-prize ex to a near-lethal
        my_act = mine.active[0] if mine.active else None
        if my_act is not None and _envelope_can_ko(mine, opp, arch):
            add -= WINCON_W["expose"] * H._prizes_for_ko(my_act)
    else:                           # slow/control: grind/stall is our win condition
        add += _race_leaf_terms(mine, opp)
    return max(-PRIZE_RACE_CLAMP, min(PRIZE_RACE_CLAMP, add))        # clamp below a prize


def _policy_prior_leaf_terms(mine, opp) -> float:
    """PRIZE-EQUAL LEAF TIEBREAKER (Layer A, mechanism 2): reward a resulting board that reflects a
    consensus-good action having been taken — a fueled-toward-cost attacker (attach-to-attacker rule)
    and a developed bench (bench-develop rule). PRIZE-EQUAL GATED by the caller: this only ADDS to
    _state_value when the leaf's prize counts equal the root's (so it is decisive among prize-tied
    moves and can NEVER flip a prize-decisive ordering). Whole term clamped < one prize (the directive's
    bound). Deck-agnostic; take_ko gets NO leaf term (prize-CHANGING, pe_safe_frac=0 in the mine).
    Returns 0 when POLICY_PRIOR is off (caller-gated) or both weights are 0 => byte-identical."""
    v = 0.0
    wa = POLICY_PRIOR_LEAF.get("attach_to_attacker", 0.0)
    if wa:
        best = 0.0
        for pk in (([mine.active[0]] if mine.active else []) + [b for b in mine.bench if b]):
            if pk is not None:
                p = _attacker_progress(pk)        # min(1, attached/line_top_cost) in [0,1]
                if p > best:
                    best = p
        v += wa * best
    wb = POLICY_PRIOR_LEAF.get("bench_develop", 0.0)
    if wb:
        nb = min(POLICY_PRIOR_LEAF_BENCH_CAP, sum(1 for b in mine.bench if b))
        v += wb * nb
    if v > POLICY_PRIOR_CLAMP:
        v = POLICY_PRIOR_CLAMP
    elif v < -POLICY_PRIOR_CLAMP:
        v = -POLICY_PRIOR_CLAMP
    return v


# --- CLOSING_EXCHANGE helpers (all dead code unless the flag is on) ------------------------------
_CE_RE_OPP_HAND = re.compile(r"(\d+)\s+damage for each card in your opponent[’']?s hand")


def _ce_priors_for_arch(arch):
    """Resolve a belief archetype label to its matchup_priors entries (LIST). Exact key first;
    else a conservative COMPONENT join — labels are evolution-line canonical strings built from
    different decklists of the same family ('Mega Lucario ex / Solrock' vs priors' 'Hariyama /
    Mega Lucario ex'), so we match ' / '-components case-insensitively with substring tolerance
    (Dunsparce ⊂ Dudunsparce), components >=5 chars. Caller decides how to combine multi-hits."""
    pri = _load_matchup_priors()
    if not arch:
        return []
    key = str(arch)
    # 'dragapult' priors entries are excluded from the lookup UNIVERSE (exact hits included):
    # the mined "Budew / Dragapult ex" entry (mill, rate 0.30) conflates with the aggro Dreepy/
    # Drakloak/Dragapult deck — belief exact-labels the aggro deck with the mill entry's key
    # (A/B smoke: fast_dreepy slow_fire 100%), and dreepy was a 07-02 CE HARM cell.
    if key in pri and isinstance(pri[key], dict) and "dragapult" not in key.lower():
        return [pri[key]]
    mine = _ce_join_components(key)
    hits = []
    for k, v in pri.items():
        if not isinstance(v, dict) or "dragapult" in k.lower():
            continue
        theirs = _ce_join_components(k)
        if any(_ce_comp_match(a, b) for a in mine for b in theirs):
            hits.append(v)
    return hits


# Components that CANNOT identify an archetype for the join: dunsparce/dudunsparce is a splash
# draw engine in half the meta, and 'dragapult ex' sits in BOTH the Budew mill list and the aggro
# Dreepy list (the known collision pair; 07-02 sign differs between them) — joining on these
# conflates opposite-pace decks, so they are dropped from BOTH sides (fail-safe: no join => fast).
_CE_JOIN_STOP = ("dunsparce", "dudunsparce", "dragapult")
_CE_BRANDS = ("hop's ", "marnie's ", "iono's ", "cynthia's ", "team rocket's ", "n's ")


def _ce_join_components(label):
    out = []
    # normalize the two apostrophes: card names mix ASCII ' and U+2019 ("Iono's" vs "Iono’s") —
    # unnormalized, brand/family joins silently miss (dormant until a curly-quoted family matters)
    for c in str(label).replace("’", "'").split("/"):
        c = c.strip().lower()
        if len(c) >= 5 and not any(s in c for s in _CE_JOIN_STOP):
            out.append(c)
    return out


def _ce_comp_match(a, b) -> bool:
    """Substring containment, plus possessive-brand family match: 'hop's phantump' and 'hop's
    trevenant' are the SAME themed line (Phantump evolves into Trevenant) under different
    evolution-line canonical names."""
    if a in b or b in a:
        return True
    for br in _CE_BRANDS:
        if a.startswith(br) and b.startswith(br):
            return True
    return False


def _ce_pri_slow(pri) -> bool:
    try:
        if str(pri.get("wincon", "")) in ("stall", "mill_deckout"):
            return True
        return float(pri.get("rate")) <= CE_SLOW_RATE
    except (TypeError, ValueError):
        return False


def _ce_opp_is_slow() -> bool:
    """CE_SLOW_ONLY gate: identified opponent archetype is a SLOW deck (stall/mill wincon or mined
    prize pace <= CE_SLOW_RATE) per matchup_priors.json. A multi-key component join counts as slow
    only if ALL joined entries agree ('Dunsparce' joins three priors keys — all stall/mill — and
    must not be discarded; the FTC gains sit exactly there). Unknown archetype / missing prior /
    conflicting join / any error => False (fail-safe: the CE term stays OFF exactly where it was
    field-refuted)."""
    hits = _ce_priors_for_arch((_OPP_ARCH[0] or (None, 0))[0])
    if not hits:
        return False
    return all(_ce_pri_slow(p) for p in hits)


def _ce_prizes_for_ko(pk) -> int:
    """EXACT prizes the opponent takes for KOing `pk` (api.py:476-477): Mega ex=3, ex=2, else 1.
    The bool `_is_ex` undercounts a Mega ex as 2 — the closing exchange needs the magnitude
    (the verified T2 loss WAS a single 3-prize Mega KO)."""
    cd = H.CARDS.get(pk.id) if pk else None
    if not cd:
        return 1
    if getattr(cd, "megaEx", False):
        return 3
    if getattr(cd, "ex", False):
        return 2
    return 1


def _ce_payable_plus1(pk, atk) -> bool:
    """Payable now, or one attach away (the return-KO lands NEXT turn, so one more attach is real)."""
    cost = list(atk.energies)
    if H._payable(pk.energies, cost):
        return True
    if len(pk.energies) + 1 < len(cost):
        return False
    return any(H._payable(pk.energies, cost[:i] + cost[i + 1:]) for i in range(len(cost)))


def _ce_attack_damage(att_pk, tgt_pk, att_hand_n, tgt_hand_n) -> float:
    """Best payable-within-one-attach attack damage of `att_pk` into `tgt_pk` (weakness-aware).
    Dynamic sizing: hand-scaling engines via H._attack_dynamic (attacker's own hand) PLUS the
    OPP-HAND-scaling family (Resentful Refrain: 'damage for each card in your opponent's hand')
    at OPP_HAND_W per card in the DEFENDER owner's hand — previously unwired (sees-but-misvalues)."""
    cd = H.CARDS.get(att_pk.id)
    if not cd:
        return 0.0
    # EX-IMMUNITY (Rock Inn): the TARGET walls the attacker if it holds the ability and the attacker is an
    # ex — 0 damage. Symmetric: protects our Crustle from their ex AND their Crustle from ours (mirror).
    if H._wall_blocks_all(tgt_pk, att_pk):
        return 0.0
    walled = H.ROCK_INN_SCOPED and H._ex_immune_vs(tgt_pk, att_pk)
    cdt = H.CARDS.get(tgt_pk.id)
    best = 0.0
    for aid in cd.attacks:
        atk = H.ATTACKS.get(aid)
        if atk is None or not _ce_payable_plus1(att_pk, atk):
            continue
        if walled and not H._attack_pierces_effects(atk):
            continue
        dmg = float(atk.damage)
        dmg += H._attack_dynamic(atk, att_hand_n, tgt_pk)
        m = _CE_RE_OPP_HAND.search((getattr(atk, "text", "") or "").lower())
        if m:
            dmg += float(int(m.group(1))) / 50.0 * OPP_HAND_W * tgt_hand_n
        if dmg <= 0:
            continue
        if cdt is not None and cdt.weakness is not None and cd.energyType == cdt.weakness:
            dmg *= 2
        if dmg > best:
            best = dmg
    return best


def _ce_best_return_prizes(att_side, def_side) -> tuple:
    """(vs_active, vs_bench_gust): max prizes `att_side` can take NEXT turn with ONE attack on
    `def_side`'s board. Attackers: their active plus every benched pokemon (promote after a KO /
    switch lines), each gated by payable-within-one-attach. Targets: the defender's ACTIVE for the
    plain return, and the defender's BENCH for gust/board-pull lines (caller discounts those)."""
    att_hand = getattr(att_side, "handCount", 0) or 0
    def_hand = getattr(def_side, "handCount", 0) or 0
    attackers = [pk for pk in (att_side.active + att_side.bench) if pk]
    d_act = def_side.active[0] if def_side.active else None
    vs_active = 0
    vs_bench = 0
    for a in attackers:
        if d_act is not None:
            if _ce_attack_damage(a, d_act, att_hand, def_hand) >= (getattr(d_act, "hp", 0) or 0):
                vs_active = max(vs_active, _ce_prizes_for_ko(d_act))
        for b in def_side.bench:
            if b is None:
                continue
            if _ce_attack_damage(a, b, att_hand, def_hand) >= (getattr(b, "hp", 0) or 0):
                vs_bench = max(vs_bench, _ce_prizes_for_ko(b))
    return vs_active, vs_bench


def _evolve_tempo_term(mine, opp) -> float:
    """Option value (>= 0) of playable evolutions HELD in hand, phase-weighted by the measured
    strong-player hold curve. 'Playable' = evolvesFrom names a Pokemon on OUR board (a dead
    evolution in hand earns nothing). See the EVOLVE_TEMPO leaf-half flag block."""
    sw = EVOLVE_TEMPO_STAGE_W[min(_game_stage(mine, opp), len(EVOLVE_TEMPO_STAGE_W) - 1)]
    if sw <= 0.0:
        return 0.0
    board = set()
    for pk in (mine.active + mine.bench):
        if pk is not None:
            cd = H.CARDS.get(getattr(pk, "id", None))
            if cd is not None:
                board.add(getattr(cd, "name", None))
    if not board:
        return 0.0
    n = 0
    for c in (getattr(mine, "hand", None) or []):
        cd = H.CARDS.get(getattr(c, "id", c))
        if cd is not None and getattr(cd, "evolvesFrom", None) in board:
            n += 1
    if n:
        DIAG["evolve_tempo_leaf"] = DIAG.get("evolve_tempo_leaf", 0) + 1   # FIRES counter (5d)
    return min(EVOLVE_TEMPO_CLAMP, EVOLVE_TEMPO_HOLD_W * sw * n)


def _pivot_value_term(mine, opp) -> float:
    """Penalty (<= 0) when a benched body strictly out-positions our ACTIVE against the current
    opponent active. 0.0 when the active already crosses the KO line (keep attacking), when no
    bench body beats it, or when either active is missing. See the PIVOT_VALUE flag block."""
    my_act = mine.active[0] if mine.active else None
    op_act = opp.active[0] if opp.active else None
    if my_act is None or op_act is None or not mine.bench:
        return 0.0
    hp = getattr(op_act, "hp", 0) or 0
    if hp <= 0:
        return 0.0
    mh = getattr(mine, "handCount", 0) or 0
    oh = getattr(opp, "handCount", 0) or 0
    act_d = _ce_attack_damage(my_act, op_act, mh, oh)
    if act_d >= hp:
        return 0.0                                   # active KOs already — never discourage that
    best_b = 0.0
    for b in mine.bench:                             # early exit: one KO-crosser fixes the result,
        if b is None:                                # and _ce_attack_damage is ~1.5-3us a call in
            continue                                 # a leaf evaluated thousands of times/decision
        d = _ce_attack_damage(b, op_act, mh, oh)
        if d >= hp:
            return -PIVOT_W                          # bench crosses the KO line the active cannot
        if d > best_b:
            best_b = d
    if best_b - act_d >= 40.0:
        return -PIVOT_W * PIVOT_CHIP                 # large raw-damage gap, no breakpoint crossed
    return 0.0


def _closing_exchange_term(mine, opp) -> float:
    """Endgame exchange value (0.0 outside closing range). Penalise leaves where the opponent's
    best return CLOSES the game (CE_LOSS_W, above a 3-prize swing) or takes prizes (CE_KO_W per
    prize, sub-prize); credit leaves from which OUR next attack can close (CE_ANSWER_W). Gust
    exposure of our bench is discounted by CE_GUST_DISCOUNT (needs an unseen card)."""
    taken = 6 - len(mine.prize)
    if taken < CE_TRIGGER_TAKEN and len(opp.prize) > CE_TRIGGER_OPP_LEFT:
        return 0.0
    v = 0.0
    opp_left = len(opp.prize)
    ret_act, ret_gust = _ce_best_return_prizes(opp, mine)
    gd = CE_GUST_DISCOUNT
    if SUMO_EXPOSURE and _SUMO_ROOT[0] and ret_gust > 0 and _sumo_pull_live(opp):
        # SUMO_EXPOSURE: while pull-exposed (verified ON-EVOLVE Heave-Ho Catcher — see the flag
        # block) their gust line needs NO unseen gust card, only the evolve they hold; raise the
        # discount to the structural _SUMO_CE_GUST. Flag-off: gd == CE_GUST_DISCOUNT exactly.
        gd = max(gd, _SUMO_CE_GUST)
        DIAG["sumo_ce_gd_armed"] = DIAG.get("sumo_ce_gd_armed", 0) + 1
        if not (ret_act >= opp_left > 0):                    # gd-sensitive branches only
            if (CE_LOSS_W * gd if ret_gust >= opp_left > 0 else
                    CE_KO_W * max(ret_act, gd * ret_gust)) != \
               (CE_LOSS_W * CE_GUST_DISCOUNT if ret_gust >= opp_left > 0 else
                    CE_KO_W * max(ret_act, CE_GUST_DISCOUNT * ret_gust)):
                DIAG["sumo_ce_gd_changed"] = DIAG.get("sumo_ce_gd_changed", 0) + 1
    if ret_act >= opp_left > 0:
        v -= CE_LOSS_W
    elif ret_gust >= opp_left > 0:
        v -= CE_LOSS_W * gd
    else:
        v -= CE_KO_W * max(ret_act, gd * ret_gust)
    ans_act, _ = _ce_best_return_prizes(mine, opp)
    if ans_act >= len(mine.prize) > 0:
        v += CE_ANSWER_W
    return v


def _sigmoid(x: float) -> float:
    """Numerically stable logistic (no overflow at large |x|)."""
    if x >= 0.0:
        return 1.0 / (1.0 + _mth.exp(-x))
    e = _mth.exp(x)
    return e / (1.0 + e)


def _prize_gradient(ml: int, ol: int) -> float:
    """Measured logit value of TAKING one prize from (ml, ol) — i.e. how much this state's win
    probability actually moves per prize. Ranges ~0.02 (hopeless) to ~0.80 (match point), which is
    the phase-dependence the flat 1000-per-prize leaf cannot express.

    At ml == 1 the next prize ENDS the game, so there is no ml-1 row to difference against; the last
    measured step (1 vs 2 prizes left) is used, which is the steepest part of the race and the right
    order of magnitude. Never returns <= 0: a non-positive gradient would inverting-scale every
    sub-prize term, and the table's monotonicity (pinned in tests) guarantees it cannot happen."""
    a, b = (2, 1) if ml <= 1 else (ml, ml - 1)
    g = WIN_PRIZE_LOGIT.get((b, ol), 0.0) - WIN_PRIZE_LOGIT.get((a, ol), 0.0)
    return g if g > 1e-6 else 1e-6


# (base logit, prize gradient) per cell, derived ONCE at import from the two pure functions above —
# the leaf is called thousands of times per decision and was doing three keyed lookups plus a call
# frame per eval for what is a compile-time constant. Bit-identical by construction.
_WIN_CELL = {cell: (WIN_PRIZE_LOGIT[cell], _prize_gradient(*cell)) for cell in WIN_PRIZE_LOGIT}


def _sub_prize_units(sub: float) -> float:
    """Convert the material remainder (everything the prize table does not already account for)
    into PRIZE UNITS for the win-probability leaf.

    EXACTLY LINEAR inside |sub| <= one prize. That is where every documented sub-prize term lives —
    they are all clamped under 1000 by design — so a 700-point term stays exactly 0.7 prizes and the
    "never flips a prize-decisive ordering" invariant is preserved verbatim.

    LOG-COMPRESSED beyond one prize, because two leaf terms are deliberately NOT sub-prize: the
    deckout and benchless penalties are written as fractions of a TERMINAL (0.4 and 0.5 of
    VALUE_BASE["loss"] = 400,000 and 500,000) to mean "near-terminal". Passed through linearly they
    would be -400 and -500 prize units, which saturate the logistic to a flat 0.0 — identical to a
    CERTAIN loss and, worse, identical to EACH OTHER, so the search could no longer order a
    deck-out next turn against having no Pokemon in play. Compression keeps them strongly negative
    (~7 prize units, more than the entire 6-prize race) while keeping them strictly ordered."""
    p = sub / VALUE_BASE["prize"]
    a = abs(p)
    if a <= 1.0:
        return p
    return _mth.copysign(1.0 + _mth.log(a), p)


def _scaled_margin(m: float) -> float:
    """Express a margin CONFIGURED on the material scale (one prize = 1000) on the ACTIVE leaf scale.

    Every tuned margin in this module — EARLY_STOP_MARGIN, ROBUST_BAND, the trace bands — is a
    number of material points, and every shipped config that sets one means "about this fraction of
    a prize". Under WIN_VALUE the leaf is a probability in [0,1], so those numbers are ~3 orders of
    magnitude too large: ROBUST_BAND=500 would put EVERY candidate in the tie band and
    EARLY_STOP_MARGIN would never fire again. Converting here keeps each threshold's INTENT and
    means no config has to be rewritten (and none can be silently mis-read).

    LEAF_V3 AUDIT (site 1 of 5): NO CHANGE REQUIRED, and that is a claim about scale, not a shrug.
    LEAF_V3 adds inside _state_value_material on the MATERIAL scale, below the prize term, so the
    unit of every margin is unchanged and each of these thresholds keeps its intent. What LEAF_V3
    does change is the DISTRIBUTION of margins — the shipped leaf is degenerate (all three searched
    candidates tie in 53.0% of MAIN decisions), so EARLY_STOP_MARGIN and ROBUST_BAND will fire
    LESS often once the leaf can separate candidates. That is the point of the lever, not a
    regression, but it means neither threshold may be re-tuned in the same change as this flag."""
    if not WIN_VALUE:
        return m
    return _sigmoid(m / WIN_VALUE_SCALE) - 0.5


def _state_value(state, me: int) -> float:
    """THE search leaf. Returns a MATERIAL score by default, or P(win) in [0,1] under WIN_VALUE.

    Every caller is a search leaf (_rollout, _minimax, _tree_best), so this is the one place the
    scale is decided. See the WIN_VALUE block for exactly what the transform does and does NOT
    preserve — in particular it is NOT a global monotone transform of the material score, and
    states in different prize configurations deliberately reorder."""
    if not WIN_VALUE:
        return _state_value_material(state, me)
    if state is None:
        return 0.5
    if state.result == me:
        return 1.0
    if state.result == (1 - me):
        return 0.0
    if state.result == 2:
        return 0.5
    v = _state_value_material(state, me)
    if (LEARNED_VALUE and _VL is not None) or (LEARNED_EVAL and _LMODEL is not None):
        return _sigmoid(v)         # both learned leaves already emit a LOGIT, not a material score
    p = state.players
    mine, opp = p[me], p[1 - me]
    ml = min(6, max(1, len(mine.prize)))
    ol = min(6, max(1, len(opp.prize)))
    # Strip the prize term the measured table already accounts for; everything BELOW the prize scale
    # (hp diff + every flag term, each clamped under one prize by design) adjusts the logit — scaled
    # by what one prize is ACTUALLY worth here, so "0.7 of a prize" means that in every cell.
    # _WIN_CELL is total over the clamped (1..6)x(1..6) domain (pinned by tests), so no fallback
    # formula exists here — an earlier unreachable branch carried the discredited global-scale one.
    # LEAF_V3 AUDIT (site 2 of 5): NO CHANGE REQUIRED — by construction, and the construction is the
    # reason the term was built this shape. LEAF_V3's contribution is added inside
    # _state_value_material, so it is already part of `v`, therefore part of `sub` below, therefore
    # converted by the LOCAL prize gradient exactly like hp and every other sub-prize flag term. Had
    # it been added HERE (on the probability scale, where leaf_fe.score()'s units natively live) it
    # would have bypassed `sub` and been worth a fixed slab of win-probability in every prize cell —
    # the discredited global-scale mistake this block's comment above already records once.
    # Its clamp (<= 0.45 prize) also keeps |sub| inside _sub_prize_units' LINEAR region on its own,
    # so the axis is never log-compressed away; only a pile-up with other sub-prize terms can do
    # that, which is pre-existing and unchanged.
    base, grad = _WIN_CELL[(ml, ol)]
    sub = v - (ol - ml) * VALUE_BASE["prize"]
    return _sigmoid(base + _sub_prize_units(sub) * grad)


def _state_value_material(state, me: int) -> float:
    """Board value from our seat: prize race dominates (×1000), then board HP, then (if
    RICH_EVAL) small potentials that break ties: energy tempo, board development, hand size."""
    if state is None:
        return 0.0
    if state.result == me:
        return VALUE_BASE["win"]
    if state.result == (1 - me):
        return VALUE_BASE["loss"]
    if state.result == 2:            # terminal DRAW (review #3): neutral, not board-diff
        return VALUE_BASE["draw"]
    if LEARNED_VALUE and _VL is not None:      # strong learned P(win) leaf (AUC 0.729) — the deep-search unlock
        vlog = _VL.value_logit(state, me)
        if vlog is not None:
            return vlog
    if LEARNED_EVAL and _LMODEL is not None:   # the fit logistic-regression value (data, not hand)
        return _learned_value(state, me)
    p = state.players
    mine, opp = p[me], p[1 - me]
    my_hp = sum(pk.hp for pk in (mine.active + mine.bench) if pk)
    op_hp = sum(pk.hp for pk in (opp.active + opp.bench) if pk)
    # WARM_START (enh 9, flag-off): the PLAIN prize+hp value is fully determined by (seat, prizes,
    # hp-sums) so memoize it; skipped when STAGE/RICH/LATE add seat-extra terms the signature omits.
    plain = not (STAGE_EVAL or RICH_EVAL or LATE_CONVERSION["enabled"] or OPP_THREAT or PRIZE_RACE
                 or DECKOUT_AWARE or BENCHLESS_LETHAL or ROLE_AWARE or ROLE_MATCHUP
                 or RACE_LEAF or SATURATE_VALUE
                 or OWN_PRIZE_RISK or RICH_STATE or WINCON_LEAF or EARLY_TEMPO
                 or PRIZE_RACE_TEMPO or DENY_EVOLUTION or CONCENTRATION or NEXT_EXEC
                 # POLICY_PRIOR breaks the (seat,prizes,hp) signature ONLY through its LEAF half.
                 # _policy_prior_leaf_terms returns 0.0 when both weights are 0 -- which is every
                 # shipped config -- so gating on the weights restores the warm cache instead of
                 # disabling it whenever the (ordering-only) PRUNE half is armed. Measured: the
                 # live challenger ships policy_prior AND warm_start, and the memo never hit.
                 or _PP_LEAF_ARMED[0]
                 or (CLOSING_EXCHANGE and (not CE_SLOW_ONLY or _CE_ROOT_SLOW[0]))
                 or SUMO_EXPOSURE  # reads opp board + bench energies beyond the sig → no warm cache
                 or PIVOT_VALUE   # reads bench bodies/energies beyond the (seat,prizes,hp) sig
                 or DECK_RACE     # reads BOTH deckCounts beyond the (seat,prizes,hp) sig
                 or H.EVOLVE_TEMPO   # reads our HAND contents beyond the (seat,prizes,hp) sig
                 or LEAF_READY    # reads per-body ENERGIES beyond the (seat,prizes,hp) sig
                 or LEAF_V3       # reads BOTH deckCounts/handCounts/energyCards beyond the sig
                 or LEAF_DYN      # ... and BOTH body counts, via the lane detector
                 or RACE_CLOCK)  # read opp/board beyond the (seat,prizes,hp) sig
                 #  (RACE_CLOCK reads opp hand/deckCount/archetype beyond the sig → exclude from warm cache)
                 #  (RACE_LEAF reads opp hand/deck beyond the (seat,prizes,hp) sig → exclude from warm cache)
                 #  (DENY_EVOLUTION reads opp stage/HP + belief beyond the (seat,prizes,hp) sig → exclude too)
                 #  (CLOSING_EXCHANGE reads hands/energies beyond the sig -> no cache; under SLOW_ONLY
                 #   a NOT-slow root contributes 0 => leaf is plain again, cache restored)
    if WARM_START and plain:
        sig = (me, len(mine.prize), len(opp.prize), my_hp, op_hp)
        c = _WARM_CACHE.get(sig)
        if c is not None:
            return c
    v = (len(opp.prize) - len(mine.prize)) * VALUE_BASE["prize"]   # fewer of OUR prizes left = winning
    v += (my_hp - op_hp) * VALUE_BASE["hp"]
    if LEAF_READY:                       # can this board actually attack? (prize*1000 + hp*1 cannot say)
        v += _leaf_ready_terms(mine)
    if RACE_CLOCK:
        # VALUE LEAF v2 (report/VALUE_LEAF_V2.md): two-sided win-condition race clock as the PRIMARY
        # dynamic signal. adv>0 = we win the race at current velocities. Saturated rational curve
        # (x/(1+|x|), no math import) worth up to ±RACE_CLOCK_W prizes — CAN outrank one prize (refuse
        # a bait prize that loses the race), never the terminal. Posture: when BEHIND, press (bonus on
        # KO-capable boards) + accept trades (forgive part of OWN_PRIZE_RISK — hiding loses anyway).
        mc, oc = _race_clocks(mine, opp)
        a = (oc - mc) / max(0.5, RACE_CLOCK_SCALE)
        v += RACE_CLOCK_W * VALUE_BASE["prize"] * (a / (1.0 + abs(a)))
        if a < 0.0:
            pst = min(1.0, -a)                                   # behindness ∈ (0,1]
            v += POSTURE_PRESS_W * pst * _ko_potential(mine, opp)
            if OWN_PRIZE_RISK:
                my_act = mine.active[0] if mine.active else None
                op_act = opp.active[0] if opp.active else None
                if my_act is not None and op_act is not None and H._opp_can_ko(op_act, my_act, _opp_blocked(opp)):
                    v += POSTURE_RISK_RELIEF * pst * OWN_PRIZE_RISK_W * H._prizes_for_ko(my_act)
    if POLICY_PRIOR:                                              # Layer A prize-equal leaf tiebreaker
        rp = _POLICY_PRIOR_ROOT_PRIZES[0]
        # PRIZE-EQUAL GATE: fire only when this leaf changed NEITHER prize count vs the search root.
        if rp is not None and len(mine.prize) == rp[0] and len(opp.prize) == rp[1]:
            v += _policy_prior_leaf_terms(mine, opp)
    if OWN_PRIZE_RISK:
        # rank-1 live fix: our active is a prize LIABILITY, not just an HP asset. If the opp's fueled
        # active can KO it next turn, discount by the prizes we'd concede (3 for a Mega ex). Sub-prize.
        my_act = mine.active[0] if mine.active else None
        op_act = opp.active[0] if opp.active else None
        if my_act is not None and op_act is not None and H._opp_can_ko(op_act, my_act, _opp_blocked(opp)):
            v -= OWN_PRIZE_RISK_W * H._prizes_for_ko(my_act)
    if RICH_STATE:
        # audit #6: recoverable BASIC/SPECIAL ENERGY in our discard = a re-fuel resource the flat leaf
        # ignores (deck-general; marginal for the low-energy STALL deck). Sub-HP weight => shapes ties.
        disc = getattr(mine, "discard", None) or []
        n_e = 0
        for c in disc:
            cid = getattr(c, "id", c)
            cd = H.CARDS.get(cid)
            if cd and getattr(cd, "cardType", None) in (5, 6):   # BASIC_ENERGY / SPECIAL_ENERGY (api.py:45-46)
                n_e += 1
        v += RICH_STATE_W * n_e
    if RACE_LEAF:                                                  # V2 win-condition-aware tie-breakers
        v += _race_leaf_terms(mine, opp)
    if EARLY_TEMPO:                                                # L1: early attacker-readiness (energy commitment)
        v += _early_tempo_terms(mine, opp, getattr(state, "turn", 0))
    if PRIZE_RACE_TEMPO:                                          # prize_race-only tempo/lead leaf (teacher-shaped)
        v += _prize_race_tempo_terms(mine, opp)
    if CONCENTRATION:                                             # measured sign-fix: fuel active+successor, penalize unfueled width
        v += _concentration_terms(mine, opp)
    if NEXT_EXEC:                                                 # draw-luck: penalize option-starved next turns, weighted by P(draw fixes it)
        v -= _next_exec_penalty(mine, opp)
    if DENY_EVOLUTION:                                             # deny the opp's evolved attacker (opponent-side, stage-aware)
        v += _deny_evolution_terms(mine, opp, getattr(state, "turn", 0))
    if WINCON_LEAF:                                                # option-value + matchup-conditional win con
        v += _wincon_leaf_terms(mine, opp)
    if SATURATE_VALUE:
        # bounded-P(win): protect the lead HARDER as we near a win (maximin-when-ahead). Fires only when
        # AHEAD and our active can be KO'd by the opp's full next-turn burst — penalty scales with how
        # close WE are to 0 prizes. Sub-prize magnitude => shapes ties, never flips the prize race.
        lead = len(opp.prize) - len(mine.prize)                   # >0 = we are ahead (fewer of OUR prizes left)
        if lead > 0:
            my_act = mine.active[0] if mine.active else None
            if my_act is not None and _envelope_can_ko(mine, opp, (_OPP_ARCH[0] or (None, 0))[0]):
                prize_at_risk = H._prizes_for_ko(my_act)   # Mega ex=3/ex=2/else 1 (api.py:476-477)
                close = (STARTING_PRIZES - len(mine.prize)) / float(STARTING_PRIZES)   # 0..1, ->1 near a win
                v -= LEAD_PROTECT_W * prize_at_risk * (0.5 + close)                    # 0.5x (early)..~1.4x (match point)
    if PIVOT_VALUE:                                                # ATTACK/RETREAT gap: position-aware active
        v += _pivot_value_term(mine, opp)
    if H.EVOLVE_TEMPO:                                             # held playable evolutions carry option value
        v += _evolve_tempo_term(mine, opp)
    if CLOSING_EXCHANGE and (not CE_SLOW_ONLY or _CE_ROOT_SLOW[0]):
        v += _closing_exchange_term(mine, opp)
    if SUMO_EXPOSURE and _SUMO_ROOT[0]:                            # anti-Hariyama bench exposure
        v += _sumo_exposure_term(mine, opp)                        # (verified on-evolve gust; see flag block)
    if STAGE_EVAL:
        # Markov-phase-weighted potentials (early=develop, mid=trade, late=close) + KO encoding.
        s = _game_stage(mine, opp)
        my_act = mine.active[0] if mine.active else None
        if my_act:
            v += STAGE_W["tempo"][s] * _attacker_progress(my_act)
        v += STAGE_W["board"][s] * sum(1 for pk in mine.bench if pk)
        v += STAGE_W["ko"][s] * _ko_potential(mine, opp)
        v += STAGE_W["hand"] * (getattr(mine, "handCount", 0) - getattr(opp, "handCount", 0))
    elif RICH_EVAL:
        # flat potentials (SUB-4 baseline) kept BELOW the HP scale so they only break ties.
        my_act = mine.active[0] if mine.active else None
        if my_act:
            v += VALUE_RICH["attacker_progress"] * _attacker_progress(my_act)
        v += VALUE_RICH["bench"] * sum(1 for pk in mine.bench if pk)
        v += VALUE_RICH["hand"] * (getattr(mine, "handCount", 0) - getattr(opp, "handCount", 0))
    if LATE_CONVERSION["enabled"]:
        v += _late_conversion_bonus(mine, opp)
    if OPP_THREAT:
        # discount our value when our active is exposed to an opponent burst-KO next turn (the
        # alakazam blindness): subtract the prizes we'd concede, scaled by the threat weight.
        my_act = mine.active[0] if mine.active else None
        op_act = opp.active[0] if opp.active else None
        if my_act and op_act:
            hp = getattr(my_act, "hp", 0) or 0
            stat, tot = _opp_burst_dmg(op_act, my_act, getattr(opp, "handCount", 0) or 0,
                                       my_hand_n=getattr(mine, "handCount", 0) or 0)
            # fire ONLY when the DYNAMIC burst is what makes it lethal (static alone wouldn't KO) —
            # the hand-scaling comeback the value fn can't see; static threats are already in the HP term.
            if tot >= hp > stat:
                prize_at_risk = H._prizes_for_ko(my_act)   # Mega ex=3/ex=2/else 1 (api.py:476-477)
                v -= OPP_THREAT_W * prize_at_risk
    if PRIZE_RACE:
        # archetype-conditioned multi-turn race: who reaches 0 prizes first (their rate = archetype).
        v += _prize_race_value(mine, opp)
    if (ROLE_AWARE or ROLE_MATCHUP) and matchup_role is not None:
        # GENERIC role layer: BEATDOWN -> follow the race; CONTROL -> follow survival/grind. The role
        # + VEIL-confidence weight are cached once per decision (_OPP_ROLE); weight 0 (thin VEIL or
        # even matchup) => 0 contribution => plain robust play. This is the SIGN the prior terms missed.
        # ROLE_MATCHUP (S1) reuses this exact term — same _OPP_ROLE cache, same role_value direction —
        # only the ROOT assignment differs (matchup_priors wincon gate instead of speed comparison).
        role, weight = _OPP_ROLE[0]
        if role != 0 and weight > 0.0:
            v += matchup_role.role_value(role, weight,
                                         _prize_race_value(mine, opp),
                                         _survival_value(mine, opp), ROLE_W)
    if DECKOUT_AWARE:
        # SELF-MILL avoidance (engine loss reason 2: start a turn with 0 deck cards). The penalty ramps
        # as our deck thins toward 0, so search avoids draw/thin plays that would deck us out — and,
        # since the marginal value of drawing is negative once the deck is short, it naturally stops
        # over-drawing when our win-cons are already on board (the user's "dynamic value" ask).
        dc = getattr(mine, "deckCount", None)
        if dc is not None:
            if dc <= 0:
                v -= 0.4 * abs(VALUE_BASE["loss"])         # forced-draw-loss imminent: near-terminal
            elif dc <= DECKOUT_FLOOR:
                v -= DECKOUT_W * (DECKOUT_FLOOR - dc + 1)   # thinner deck = larger penalty
    if DECK_RACE:
        # the RELATIVE half of deck awareness (see the flag block): deck-count differential, live
        # only inside the grind horizon where deck-out is a reachable win condition for either side.
        my_dc = getattr(mine, "deckCount", None)
        op_dc = getattr(opp, "deckCount", None)
        if my_dc is not None and op_dc is not None \
                and min(my_dc, op_dc) <= DECK_RACE_TRIGGER:
            d = DECK_RACE_W * (my_dc - op_dc)
            v += max(-DECK_RACE_CLAMP, min(DECK_RACE_CLAMP, d))
            DIAG["deck_race"] = DIAG.get("deck_race", 0) + 1   # FIRES counter (ship_gate 5d)
    if LEAF_V3 and not LEAF_DYN:
        # `and not LEAF_DYN` is the PRECEDENCE rule, and it is a no-op for every config that exists:
        # LEAF_DYN defaults False and no shipped config sets it, so this predicate is bit-identical
        # to the bare `if LEAF_V3:` it replaces. The two flags are alternative fits of the SAME
        # axis (one global vector vs a lane mixture of three), so summing them would count the
        # resource axis twice and blow the pairwise prize bound both of them are clamped to honour.
        # THE WITHIN-POSITION RESOURCE AXIS (see the flag block). prize*1000 above is untouched;
        # this is a second axis clamped strictly below one prize, because the fixed-effects
        # coefficients were calibrated HOLDING PRIZES FIXED and using them to trade a prize is
        # outside their support. features() returns None rather than a zero vector when the state
        # cannot supply a field, so an abstention is COUNTED, never scored as an average board.
        if _LEAF_FE is None:
            DIAG["leaf_v3_dead"] = DIAG.get("leaf_v3_dead", 0) + 1   # armed but unbundled — LOUD in DIAG
        else:
            _fv3 = _LEAF_FE.features(state, me)
            if _fv3 is None:
                DIAG["leaf_v3_abstain"] = DIAG.get("leaf_v3_abstain", 0) + 1
            else:
                # STRUCTURAL ceiling: min() against the prize scale itself, so the axis cannot
                # flip a prize-decisive ordering even if LEAF_V3_CLAMP is misconfigured or
                # VALUE_BASE is rescaled underneath it. The bound is PAIRWISE (2*C < one prize) —
                # see the flag block for why 0.45, not the 0.6 the neighbouring term uses.
                _c3 = min(LEAF_V3_CLAMP, LEAF_V3_CEIL_FRAC * VALUE_BASE["prize"])
                v += max(-_c3, min(_c3, LEAF_V3_W * _LEAF_FE.score_from_features(_fv3)))
                DIAG["leaf_v3"] = DIAG.get("leaf_v3", 0) + 1      # FIRES counter (ship_gate 5d)
    if LEAF_DYN:
        # THE DYNAMIC OBJECTIVE (see the flag block): the same resource axis, but the coefficient
        # vector is a mixture over the three TERMINAL CONDITIONS weighted by a detector that reads
        # only leaf-visible proximity (prizes taken, deck counts, body counts — all public for both
        # seats, all already read elsewhere in this function). REFUTED at its pre-registered
        # per-condition gate (DECK_OUT -0.0930 [-0.1257,-0.0644]); this stays off.
        #
        # Everything the LEAF_V3 branch guards is guarded identically here, for the same reasons:
        # the module can be missing from the bundle (COUNTED, not crashed), extraction can abstain
        # (COUNTED, never scored as 0.0 — score_dyn returns None where score() returns 0.0
        # precisely so the caller can tell the two apart), and the ceiling is re-derived
        # STRUCTURALLY from VALUE_BASE at every evaluation so no config can widen the axis past
        # its share of a prize. The lane vectors are up to 4.3x the global vector's norm
        # (leaf_fe.W_LANE: DECK_OUT L2 0.682 vs global 0.158), so the clamp is not decorative here
        # — it binds on 3.1% of real corpus rows against 0.3% for LEAF_V3.
        if _LEAF_FE is None:
            DIAG["leaf_dyn_dead"] = DIAG.get("leaf_dyn_dead", 0) + 1   # armed but unbundled
        else:
            _sdy = _LEAF_FE.score_dyn(state, me)
            if _sdy is None:
                DIAG["leaf_dyn_abstain"] = DIAG.get("leaf_dyn_abstain", 0) + 1
            else:
                _cd = min(LEAF_DYN_CLAMP, LEAF_V3_CEIL_FRAC * VALUE_BASE["prize"])
                v += max(-_cd, min(_cd, LEAF_DYN_W * _sdy))
                DIAG["leaf_dyn"] = DIAG.get("leaf_dyn", 0) + 1     # FIRES counter (ship_gate 5d)
    if BENCHLESS_LETHAL:
        # EMPTY-BENCH cliff (engine loss reason 3: no Pokemon in Active Spot). A benchless board loses
        # the instant our active is KO'd — the single biggest documented field loss (5/16). Penalty is
        # large only when the opponent can actually KO our active next turn; otherwise a soft cliff.
        if sum(1 for pk in mine.bench if pk) == 0:
            my_act = mine.active[0] if mine.active else None
            op_act = opp.active[0] if opp.active else None
            if my_act is None:
                v -= 0.5 * abs(VALUE_BASE["loss"])         # no Pokemon in play at all
            else:
                # THREAT_ENVELOPE: judge against their full fueled burst + archetype booster (belief);
                # else the visible current-energy attack only.
                can_ko = (_envelope_can_ko(mine, opp, (_OPP_ARCH[0] or (None, 0))[0]) if THREAT_ENVELOPE
                          else (op_act is not None and H._opp_can_ko(op_act, my_act, _opp_blocked(opp))))
                if can_ko:
                    v -= BENCHLESS_W                       # benchless AND active dies next turn ⇒ ~loss
                else:
                    v -= 0.1 * BENCHLESS_W                 # benchless cliff: one KO from losing, avoid
    if WARM_START and plain and len(_WARM_CACHE) < _WARM_CAP:
        _WARM_CACHE[(me, len(mine.prize), len(opp.prize), my_hp, op_hp)] = v
    return v


def _late_conversion_bonus(mine, opp) -> float:
    """The convergent #1 fix: when even/ahead on prizes past mid-game, REWARD pressing for the KO /
    lethal setup and PENALISE passive stalling (board-HP hoarding with no KO threat) — so search
    stops choosing "stay even and stall" lines while the opponent's big attacker takes 3 prizes.

    Gated by LATE_CONVERSION['enabled'] (False by default => never called from baseline). All weights
    live in value_db.json so the arena tunes them. Magnitude is held BELOW one prize (1000) so it only
    breaks ties between equal-prize lines (press vs stall), never overrides the prize race itself.
    """
    lc = LATE_CONVERSION
    s = _game_stage(mine, opp)
    if s < int(lc.get("stage_trigger", 1)):
        return 0.0
    prize_lead = len(opp.prize) - len(mine.prize)        # >0 = we're ahead (fewer of our prizes left)
    if lc.get("even_or_ahead_only", True) and prize_lead < lc.get("lead_margin", 0):
        return 0.0
    scale_list = lc.get("per_stage_scale", [0.0, 0.6, 1.0])
    scale = scale_list[s] if s < len(scale_list) else 1.0
    if scale <= 0.0:
        return 0.0
    kp = _ko_potential(mine, opp)                        # 1.0 = can KO now, 0.6 = one attach away, 0 = no
    bonus = 0.0
    if kp >= 1.0:
        bonus += lc.get("press_ko", 0.0)                 # we can close -> press for the KO
    elif kp >= 0.6:
        bonus += lc.get("lethal_setup", 0.0)             # one attach from lethal -> value the setup
    else:
        bonus += lc.get("stall_penalty", 0.0)            # even/ahead late with no threat -> stalling, bad
    return bonus * scale


# --- TRUE_DECK (2026-07-25): feed search_begin the deck we ACTUALLY still have -----------------
# MEASURED BUG (scripts/probes/probe_your_deck_live.py): `_predict` returns MY_DECK[:] as
# `your_deck` -- the static 60-card decklist, never reduced by draws/discards/plays, and UNSHUFFLED
# (the vary-shuffle below touches only `odeck`). cg/api.py:530 says your_deck is ignored ONLY when
# select.deck != None; otherwise :555 length-checks it and :577 marshals it into libcg. So on every
# MAIN frame the search rolls our draws from deck.csv order, including cards already in the discard,
# the hand and on the board. Probe: reversing the SAME multiset changed the rollout value in
# 16/63 = 25.4% of MAIN frames, mean |delta| 70.7, MAX 1020.0 (one prize = 1000).
#
# The fix is arithmetic, not estimation: our own 60 is known a priori, and every own zone except the
# deck and the face-down prizes is fully visible, so remaining = MY_DECK - seen EXACTLY.
# Keyed on Card.serial (unique per match) rather than card id, because select.effect/contextCard
# routinely point at a card already counted in another zone and id-multiset counting
# double-subtracts it (the audit measured residual 4/200 games -> 0/700 on serial-keying).
#
# FAIL-SAFE: the accounting must reconcile (len(remaining) == deckCount + prizeCount) or we fall
# back to the old MY_DECK[:] and count it. A wrong deck is worse than a stale one -- search_begin
# raises ValueError if len(your_deck) < deckCount, which would crash the turn.
TRUE_DECK = False


def _own_seen_ids(st, me):
    """id-multiset of every OWN card currently visible, deduped by serial.

    Zones: hand, discard, and every body on active+bench INCLUDING its attachments. The attachment
    attrs are `energyCards` / `tools` / `preEvolution` (cg/api.py:345-348) -- note `energies` is a
    list of EnergyType, i.e. TYPES not card identities, and is useless for subtraction. Guessing
    `attachedCards` instead of `energyCards` made the reconciliation fail on 78% of calls (measured:
    true_deck 1,655 vs true_deck_fallback 5,837) because every attached energy went uncounted.
    Face-down prizes are NOT visible and must not be subtracted.
    """
    seen, out = set(), []
    mine = st.players[me]

    def take(c):
        if c is None:
            return
        sn = getattr(c, "serial", None)
        cid = getattr(c, "id", None)
        if cid is None or sn is None or sn in seen:
            return
        seen.add(sn)
        out.append(cid)

    for c in (getattr(mine, "hand", None) or []):
        take(c)
    for c in (getattr(mine, "discard", None) or []):
        take(c)
    for pk in list(getattr(mine, "active", None) or []) + list(getattr(mine, "bench", None) or []):
        if pk is None:
            continue
        take(pk)
        for attr in ("energyCards", "tools", "preEvolution"):
            for c in (getattr(pk, attr, None) or []):
                take(c)
    return out


def _board_zone_ids(st):
    """Own-deck cards that sit in BOARD-level zones, which live on State and not on PlayerState:

      stadium  (api.py:377)  list[Card], size 0 or 1 — a stadium in play came out of somebody's
                             deck, and if it came out of OURS it is not in ours any more.
      looking  (api.py:378)  list[Card|None] — the cards currently being looked at mid-effect
                             (a deck search in progress). None entries are face-down.

    MEASURED (probe_true_deck_gap, 172 frames): missing these two made the reconciliation fail on
    116 frames — delta +1 on 99 (the stadium) and +9 on 4 (a nine-card look). Returns the two sets
    separately because ownership is ambiguous: a stadium can be the OPPONENT's copy of a card we
    also run, so the caller must not subtract blind — it tries each combination and keeps whichever
    reconciles (see _remaining_own).
    """
    def ids(seq):
        out = []
        for c in (seq or []):
            cid = getattr(c, "id", None)
            if cid is not None:
                out.append(cid)
        return out
    return ids(getattr(st, "stadium", None)), ids(getattr(st, "looking", None))


def _remaining_own(st, me):
    """MY_DECK minus every visible own card -> (deck+prize) as a list of ids, or None if the
    accounting does not reconcile against deckCount + prizeCount.

    The length check IS the oracle, so where ownership of a board zone is ambiguous we try each
    subtraction and accept the first that reconciles exactly. That can only ever turn a fallback
    into a correct answer: a combination that does not reconcile is rejected exactly as before,
    and a NEGATIVE count (more copies seen than the list holds) still rejects outright.
    """
    import collections as _c
    mine = st.players[me]
    want = int(getattr(mine, "deckCount", 0) or 0) + len(getattr(mine, "prize", None) or [])
    base = _c.Counter(_own_seen_ids(st, me))
    stad, look = _board_zone_ids(st)
    for extra in ((), stad, look, stad + look):          # cheapest-first; () is the legacy behaviour
        pool = _c.Counter(MY_DECK)
        pool.subtract(base)
        if extra:
            pool.subtract(_c.Counter(extra))
        if any(v < 0 for v in pool.values()):
            continue                     # saw more copies than the list holds -> not this one
        rem = list(pool.elements())
        if len(rem) == want:
            return rem
    return None



def _own_pred(st, me, mine, seed):
    """(your_deck, your_prize) for search_begin. TRUE_DECK off => byte-identical legacy behaviour."""
    if not TRUE_DECK:
        return MY_DECK[:], MY_DECK[:len(mine.prize)]
    rem = _remaining_own(st, me)
    if rem is None:
        DIAG["true_deck_fallback"] = DIAG.get("true_deck_fallback", 0) + 1
        return MY_DECK[:], MY_DECK[:len(mine.prize)]
    DIAG["true_deck"] = DIAG.get("true_deck", 0) + 1
    # Shuffle OUR pool too. The legacy path passed deck.csv order on every rollout, so the search
    # saw the same fabricated draw sequence every time -- not even determinization noise.
    r = random.Random(((len(mine.prize) * 73856093) ^ (seed * 2654435761)) & 0xFFFFFFFF)
    r.shuffle(rem)
    return rem, rem[:len(mine.prize)]


def _predict(obs, vary: bool = False, seed: int = 0):
    """Build valid search_begin predictions (sizes from obs; our-turn eval is prediction-robust).

    vary=True re-shuffles the predicted OPPONENT deck so a determinization samples a
    different plausible opponent hand/prizes — averaging over several reduces the variance
    that a single crude prediction injects into the 2-ply value.

    The shuffle uses a LOCAL rng seeded DETERMINISTICALLY from the board state + sample index
    (not the global `random`, which would make traces irreproducible and pollute global RNG —
    flagged in review). Same state + same k → same determinization → reproducible decisions.
    """
    st = obs.current
    me, opp = st.yourIndex, 1 - st.yourIndex
    mine, op = st.players[me], st.players[opp]
    op_active = [_BASIC_ID] if (op.active and len(op.active) > 0 and op.active[0] is None) else []
    if OPP_DECK_MODEL and belief is not None:
        # W3 OPPONENT MODEL: a CONSISTENT belief world — the opponent's unseen deck/prize/hand drawn
        # WITHOUT replacement from the posterior particle minus their revealed cards (see the flag
        # comment). Seed mixed with board state exactly like the vary-shuffle below, so each world
        # index k gives a different-but-reproducible completion (same state + same k → same world).
        # getattr: a stale belief.py without sample_world degrades to the legacy path, never raises.
        _sw = getattr(belief, "sample_world", None)
        if _sw is not None:
            try:
                s = ((len(mine.prize) * 73856093) ^ (len(op.prize) * 19349663)
                     ^ (op.handCount * 83492791) ^ ((seed + 1) * 2654435761)) & 0xFFFFFFFF
                if BELIEF_VARIANTS:
                    # count-granular real-list posterior (see the BELIEF_VARIANTS flag block);
                    # a stale belief.py without the kwarg TypeErrors into this except → legacy.
                    w = _sw(obs, s, BELIEF_VARIANTS_MIN_CONF, use_variants=True)
                else:
                    w = _sw(obs, s, OPP_DECK_MODEL_MIN_CONF)
            except Exception:
                w = None
            if w is not None:
                odeck, oprize, ohand = w
                DIAG["opp_deck_model_worlds"] = DIAG.get("opp_deck_model_worlds", 0) + 1
                if BELIEF_VARIANTS and getattr(belief, "LAST_WORLD_SOURCE", None) == "variants":
                    # FIRES instrumentation (ship_gate): worlds actually drawn from the variants
                    # posterior (vs the legacy fall-through inside sample_world).
                    DIAG["belief_variants_worlds"] = DIAG.get("belief_variants_worlds", 0) + 1
                _yd, _yp = _own_pred(st, me, mine, seed)
                return (_yd, _yp, odeck, oprize, ohand, op_active)
    # opponent deck prediction: a BELIEF-sampled real meta deck (consistent with the opponent's
    # visible cards, so search_begin stays legal) — fixes the self-copy weakness — else MY_DECK.
    odeck = None
    if BELIEF and belief is not None:
        try:
            odeck = belief.sample_opponent_deck(obs, seed)
        except Exception:
            odeck = None
    odeck = list(odeck) if odeck else MY_DECK[:]
    if vary:
        # manual mix (no hash() — it's salted per-process via PYTHONHASHSEED → irreproducible)
        s = ((len(mine.prize) * 73856093) ^ (len(op.prize) * 19349663)
             ^ (op.handCount * 83492791) ^ (seed * 2654435761)) & 0xFFFFFFFF
        random.Random(s).shuffle(odeck)
    _yd, _yp = _own_pred(st, me, mine, seed)
    return (_yd, _yp, odeck, odeck[:len(op.prize)], odeck[:op.handCount], op_active)


def _roll_end_value(st, me: int) -> float:
    """ROLL_TO_END's return on the ACTIVE leaf scale — the ONE seam for deep-rollout returns.

    Under WIN_VALUE everything (terminals included) is _state_value's P(win); flag-off keeps the
    legacy shape byte-identically: ±1.0/0.0 terminals, else the material leaf scaled to ~[-1,1].
    Consolidated 2026-07-31 from three inline forks — the scale decision was stated four times
    (here 3x + _state_value) and the terminal copy could drift from the leaf's own mapping.

    LEAF_V3 AUDIT (site 3 of 5): NO CHANGE REQUIRED. Both branches route through _state_value, so
    LEAF_V3 rides the same path the hp term does — under WIN_VALUE it is already a probability, and
    flag-off it is divided by 1e6 with everything else. Worth stating because the /1e6 is the one
    place a NON-terminal leaf is squeezed next to hard ±1.0 terminals: LEAF_V3's ceiling is 450
    material = 4.5e-4 after the divide, i.e. it can shade a non-terminal rollout return but can never
    approach a terminal. That ordering is the invariant this seam exists to protect."""
    if WIN_VALUE:
        return _state_value(st, me)          # already a probability; terminals map to 1.0/0.0/0.5
    if st is not None and st.result != -1:
        return 1.0 if st.result == me else (-1.0 if st.result == (1 - me) else 0.0)
    return _state_value(st, me) / 1e6


def _rollout(ss, me: int, deadline: float | None = None) -> float:
    """Roll forward with the heuristic; value from our seat.

    1-ply (ROLL_2PLY=False): stop at the end of OUR turn (when control leaves `me`).
    2-ply (ROLL_2PLY=True):  keep rolling through ONE opponent turn (the heuristic plays
      whoever is to move, so the opponent answers with its best heuristic line), then value
      when control returns to us — so a move that exposes us to a lethal counter is punished.
    Any terminal (a win/loss inside either turn) returns immediately.

    `deadline` (wall-clock): when a long ROLL_TO_END / ROLL_CAP rollout overruns the per-decision
    budget we BREAK and return the current leaf eval (scaled like a cap hit), so a single deep
    rollout can never blow the pool — DeadlineExceeded is structurally impossible. d=0 1-ply
    rollouts are ~free so the check is cheap; it only ever bites the deep horizons.
    """
    obs, sid = ss.observation, ss.searchId
    if ROLL_TO_END:
        # Deep MC: play BOTH sides with the heuristic to terminal; return the ACTUAL result
        # (+1 win / -1 loss / 0 draw) — a real signal, not the coarse eval. Averaged over the
        # belief-sampled determinizations by the caller = a Monte-Carlo win-probability.
        for _ in range(ROLL_END_CAP):
            st = obs.current
            if st is None or obs.select is None:
                return 0.0
            if st.result != -1:
                return _roll_end_value(st, me)       # terminal (under WIN_VALUE a draw is 0.5)
            if deadline is not None and time.time() > deadline:
                return _roll_end_value(st, me)       # deadline → current leaf on the rollout scale
            ss = api.search_step(sid, H._choose(obs))
            obs, sid = ss.observation, ss.searchId
        return _roll_end_value(obs.current, me)      # cap hit → leaf on the rollout scale
    saw_opp = False
    for _ in range(ROLL_CAP):
        st = obs.current
        if st is None or st.result != -1 or obs.select is None:
            return _state_value(st, me)
        cur = st.yourIndex
        if cur != me:
            if not ROLL_2PLY:
                return _state_value(st, me)   # 1-ply: value at our turn's end
            saw_opp = True                    # 2-ply: roll the opponent's reply
        elif saw_opp:
            return _state_value(st, me)        # back to us after a full opp turn -> horizon
        if deadline is not None and time.time() > deadline:
            return _state_value(st, me)        # deadline mid-rollout → return current leaf eval
        ss = api.search_step(sid, H._choose(obs))
        obs, sid = ss.observation, ss.searchId
    return _state_value(obs.current, me)


# --- THE DOMINION: depth-limited minimax over a single belief world ----------------------------
# Per main decision the caller draws K_WORLDS belief determinizations and averages a minimaxed
# value per candidate over them; here we evaluate ONE world. The engine only forks from root
# (search_begin), so — like _tree_best — each minimax NODE re-forks from `obs` and replays the
# move `seq` that leads to it, then either branches (enumerating an opponent reply at a MIN node)
# or rolls a player's turn forward with the heuristic. `seq` is a flat list of search_step selects.
_DOM_ERR = [0]      # caught minimax/rollout errors this process (surfaced at TRACE_LEVEL≥1)


def _roll_to_branch(ss, deadline):
    """From `ss`, step the heuristic until we reach a BRANCHABLE MAIN decision (single-select,
    ≥2 options) or a terminal. Forced/multi sub-selects are consumed by the heuristic — they don't
    cost minimax depth. Returns (ss, st, branchable: bool). At a branchable node the caller decides
    whether it's a MAX (ours) or MIN (opp) node from st.yourIndex."""
    sid = ss.searchId
    obs = ss.observation
    for _ in range(ROLL_CAP):
        st = obs.current
        if st is None or st.result != -1 or obs.select is None:
            return ss, st, False
        sel = obs.select
        if sel.context == SelectContext.MAIN and sel.maxCount == 1 and len(sel.option) >= 2:
            return ss, st, True                        # a real branch point
        if deadline is not None and time.time() > deadline:
            return ss, st, False
        ss = api.search_step(sid, H._choose(obs))       # forced/multi → heuristic, no depth cost
        sid, obs = ss.searchId, ss.observation
    return ss, obs.current, False


# --- replay identity keys (2026-07-22 in-world divergence fix) ---------------------------------
# ROOT CAUSE (VPS forensics, _odm_dbg/odm_repro.py): the engine resolves hidden information
# freshly on EVERY search_begin (std::random_device-seeded mt19937 in libcg; no seeding API), so
# two forks of the SAME (obs, preds) can reach the same branch point with DIFFERENT option lists
# (measured: 3 re-forks of one MIN node saw 18/19/12 options; the pre-OPP_DECK_MODEL self-copy
# world diverges the same way — its near-identical hidden zones just kept the option COUNTS
# stable, so stale positional indices stayed in range and silently replayed a possibly-different
# move). A reply index recorded in one fork is therefore only a POSITION HINT in the next fork.
# Fix: each branch select in `seq` carries an identity key of the option it meant; `_replay`
# re-matches by identity when the stored position no longer holds that option. This repairs BOTH
# failure shapes for ALL world types: the OPP_DECK_MODEL-surfaced out-of-range throw (world was
# dropped) and the pre-existing in-range wrong-move replay (silent value corruption).
_OPT_KEY_FIELDS = ("type", "number", "area", "index", "playerIndex", "toolIndex", "energyIndex",
                   "count", "inPlayArea", "inPlayIndex", "attackId", "cardId", "serial",
                   "specialConditionType")
# position-free subset: drops within-zone indices/serials that legitimately shift between two
# hidden-zone completions of the same public state (two copies of a card are interchangeable).
_OPT_KEY_POSFREE = ("type", "number", "area", "playerIndex", "count", "inPlayArea",
                    "attackId", "cardId", "specialConditionType")


def _opt_key(o, fields=_OPT_KEY_FIELDS):
    """Hashable identity key of a select Option (enums coerced to int; missing fields None)."""
    out = []
    for f in fields:
        v = getattr(o, f, None)
        try:
            v = int(v) if v is not None else None
        except (TypeError, ValueError):
            v = str(v)
        out.append(v)
    return tuple(out)


def _rematch_select(sel, mv, sig):
    """Resolve branch select `mv` (a [option_index] hint) against THIS fork's option list using the
    identity key `sig` recorded when the index was chosen. Returns the select to step ([j]) or the
    original mv (positional fallback, exactly the legacy behavior) — raises ValueError only when
    the option is unmatchable AND the index is out of range (a genuinely diverged world; the caller
    drops this line for this world, exactly like the legacy engine throw it replaces)."""
    n = len(sel.option) if (sel is not None and sel.option is not None) else 0
    i = mv[0] if (len(mv) == 1 and isinstance(mv[0], int)) else None
    if i is None:                                       # non-single select (never a branch move):
        return mv                                       # legacy positional behavior
    full, posfree = sig
    if 0 <= i < n and _opt_key(sel.option[i]) == full:
        return mv                                       # fast path: same option, same position
    for j in range(n):                                  # exact option, moved position
        if _opt_key(sel.option[j]) == full:
            DIAG["replay_rematch_exact"] = DIAG.get("replay_rematch_exact", 0) + 1
            return [j]
    for j in range(n):                                  # same move, different hidden-zone slot
        if _opt_key(sel.option[j], _OPT_KEY_POSFREE) == posfree:
            DIAG["replay_rematch_posfree"] = DIAG.get("replay_rematch_posfree", 0) + 1
            return [j]
    if 0 <= i < n:                                      # unmatched but in range: keep the legacy
        DIAG["replay_pos_fallback"] = DIAG.get("replay_pos_fallback", 0) + 1
        return mv                                       # positional replay (pre-fix behavior)
    DIAG["replay_unmatched"] = DIAG.get("replay_unmatched", 0) + 1
    raise ValueError("replay divergence: option unmatched and index out of range "
                     f"(idx {i}, {n} options)")         # caught per-line by _minimax, as before


def _replay(obs, preds, seq, deadline, sigs=None):
    """Re-fork from root and replay the BRANCH selects in `seq`: apply seq[0] at the live root MAIN
    decision, then for each later branch move roll the heuristic to the next branch point and apply
    it there. Returns the SearchState positioned just after the last branch move (caller must
    search_end). The engine forks only from root, so every node reconstructs its prefix this way.

    `sigs` (parallel to `seq`, entries may be None) carries per-move identity keys
    ((full, position-free) from `_opt_key`): the engine re-randomizes hidden zones per fork (see
    the divergence-fix comment above), so a stored option INDEX is re-matched by identity against
    this fork's option list before stepping. sigs=None / entry None => exact legacy behavior."""
    ss = api.search_begin(obs, *preds, False)
    for k, mv in enumerate(seq):
        if k > 0:
            ss, st, branchable = _roll_to_branch(ss, deadline)
            if not branchable:                          # prefix no longer reachable (shouldn't happen)
                return ss
        sig = sigs[k] if (sigs is not None and k < len(sigs)) else None
        if sig is not None:
            mv = _rematch_select(ss.observation.select, mv, sig)
        ss = api.search_step(ss.searchId, mv)
    return ss


OPP_MODEL = None  # None => MIN (adversarial worst-case = today's byte-identical default). Else a dict
                  # {"mode": "min"|"greedy"|"softmax", "tau": float}: elo-conditional opponent at minimax
                  # opp nodes. Mirrors src/core/state.OpponentModel (the container can't import core).

# RISK_SLOW_TAU — archetype-SELECTIVE soft-min: the one untested slice of the risk-posture axis. A
# BLANKET global softmax was SUB-21 (stayed inside baseline CI; B4 refuted "risk posture" as THE
# ladder gap, our froslass pilot ≈ the best froslass pilot). This arms the softmax opponent (OPP_MODEL,
# above) ONLY vs a confidently-slow, NON-fallback archetype — opp prizes/turn <= RISK_SLOW_CUTOFF,
# the decks that genuinely won't punish a commit — and HOLDS the hard-MIN vs every aggro deck that
# broke RACE_CLOCK (dunsparce 0.9 / lucario 1.6 / dragapult 1.5 / alakazam 1.4 / archaludon 1.1).
# Decided per-decision at the root of agent(); 0.0 => OPP_MODEL never set => BYTE-IDENTICAL to shipped.
RISK_SLOW_TAU = 0.0        # softmax temperature for the slow-slice opponent; 0.0 = OFF (hard-MIN)
RISK_SLOW_CUTOFF = 0.7     # STRUCTURAL (not a swept tunable): opp prizes/turn <= this = "slow, won't punish"


# --- REPLY_TENDENCY (2026-07-22, report/opp_predictability_20260722.md): mined MIN-node reply ----
# prior — the evidence-backed anticipation lever. MEASURED on 77k+ real in-band (550-1100) ladder
# decisions: the self-copy MIN node (H._score ordering, TOP_Mo=2 prefix) misses real opponents'
# actual replies 37-57% (worst Hariyama/MegaLucario 0.570, Alakazam/Dudunsparce 0.548); the
# dominant confusions are PATIENCE-shaped (we predict ATTACK, they PLAY/develop; PLAY->END;
# PLAY->ABILITY), and real players HOLD offered gust-class cards (taken 5-13% of offered frames vs
# our model's assumed 20-33%) — so the searched TOP_Mo reply set is systematically the WRONG worst
# case. This lever re-orders the MIN node's candidate replies with a mined per-archetype × phase
# tendency bonus (scripts/mine_opp_tendencies.py -> opp_tendencies.json, shipped in the bundle):
# BOOST-ONLY (maximin correction 2026-07-23, user directive): boost reply TYPES the belief-MAP
# archetype actually favors in this phase — W·max(0, p_pick − 0.5) — and NEVER penalize types it
# rarely picks; a gust-class PLAY reply uses the gust-specific held-rate cell (the strongest
# measured gap).
# 🔴 MAXIMIN PRINCIPLE (why boost-only): the MIN node must assume opponent PERFECTION for safety.
# "They rarely play it" must never remove a dangerous (our-score-top) reply from the searched
# set — that bets our safety on their blunder. Tendency information may only ADD their-favored
# replies into TOP_Mo survival: under a true MIN, adding candidates can only make the value more
# pessimistic-accurate; removing candidates is the unsafe direction. The one-sided guarantee is
# exact on the sort KEY: no reply's key ever drops below its legacy H._score (at a fixed TOP_Mo
# width a BOOSTED reply can still outrank a legacy candidate — that is the sanctioned direction).
# Opponent imperfection is exploitable only on OUR (MAX) side — see the design note below.
# ORDERING-ONLY by construction: the bonus enters exclusively the _reply_order sort
# key, so it changes WHICH opponent replies survive the TOP_Mo prune (get searched), never a
# searched reply's value — MIN still takes the min over the searched set, and a lethal reply
# (H._score >= 10000) can never be displaced by the [0, +REPLY_TENDENCY_W/2] boost band. Gated like
# OPP_DECK_MODEL/DENY_EVO: fires only on a non-fallback belief MAP at conf >=
# REPLY_TENDENCY_MIN_CONF that resolves (exact label, else the _ce_comp_match family join — the
# 07-22 label-robustness directive) to a mined entry with n_decisions >= REPLY_TENDENCY_MIN_N and
# not low_confidence; the pooled "_fallback" entry NEVER steers (no read => no reply prior).
# Flag-off => byte-identical (loader never runs; the ordering expression is unchanged).
#
# DESIGN NOTE — MAX-node counterpart: BUILT 2026-07-23 as TRAP_TIEBREAK (flag block below the
# _RT tables): among OUR candidate moves whose MIN values sit within an epsilon band
# (safety-equal under the maximin), prefer the move whose tendency-weighted EXPECTED opponent
# reply is best for us — trap-laying as a TIE-BREAK among already-safe moves, exploitation
# strictly subordinate to maximin (it may never override a MIN-value difference outside the
# band). Relation to the killed felt/secure leaf (VALUE_LEAF_DYNAMIC,
# project_ptcg_value_leaf_dynamic): that was a LEAF RE-SIGN (re-scoring terminal values) killed
# by its own addressability scan; this counterpart is a MOVE TIE-BREAK at our MAX node — a
# different family. Future sessions: do not conflate the two or treat that kill as prior art
# against this note.
REPLY_TENDENCY = False
REPLY_TENDENCY_W = 200.0        # ordering-bonus scale: bonus = W·max(0, p_pick − 0.5) ∈ [0, +W/2]
                                # — BOOST-ONLY (maximin, see block above); tens-to-hundreds vs the
                                # H._score bands (POLICY_PRIOR_PRUNE sizing), << the rule bands
                                # 3000/5000 and lethal 10000
REPLY_TENDENCY_MIN_CONF = 0.15  # posterior-confidence floor (OPP_DECK_MODEL_MIN_CONF convention)
REPLY_TENDENCY_MIN_N = 100      # ignore artifact entries AND per-type cells below this many
                                # decisions (the artifact marks entry-level low_confidence too)
_REPLY_TENDENCIES: dict = {}    # mined artifact: {library-label: entry, "_fallback": pooled}
_REPLY_TENDENCIES_TRIED = [False]
_RT_ROOT = [None]               # tendencies entry resolved ONCE per MAIN decision in agent()
                                # (the _CE_ROOT_SLOW convention — never a per-node label join)
# raw option-type int -> artifact type-name. The miner (scripts/mine_opp_tendencies.py) imports
# THIS table so producer and consumer can never drift on the grouping.
_RT_TYPE_NAME = {5: "ATTACH", 6: "ATTACH", 8: "ATTACH", 7: "PLAY", 9: "EVOLVE",
                 10: "ABILITY", 12: "RETREAT", 13: "ATTACK", 14: "END"}
# gust-class card names (the report's setup list): board-pull trainers real players HOLD.
# Compared apostrophe-normalized + lowercase (the _ce_join_components curly-quote lesson).
_RT_GUST_NAMES = ("Boss's Orders", "Lisia's Appeal", "Pokemon Catcher", "Pokémon Catcher",
                  "Prime Catcher", "Team Rocket's Giovanni")


def _rt_norm_name(s) -> str:
    return str(s).replace("’", "'").strip().lower()


_RT_GUST_SET = frozenset(_rt_norm_name(n) for n in _RT_GUST_NAMES)

# --- TRAP_TIEBREAK (2026-07-23): the MAX-node counterpart of REPLY_TENDENCY — the DESIGN NOTE
# above, now built. At the ROOT move selection, among OUR candidates whose belief-averaged
# minimax values lie within TRAP_TIEBREAK_EPS of the best backed value, break the tie toward the
# move that best punishes the opponent's LIKELY (mined-empirical) reply: each near-tied root
# move's FIRST MIN node exposes its searched replies' backed values; weight them by the
# archetype's p_pick reply-class rates (the SAME opp_tendencies.json cells, loader and
# belief/conf/n gates as REPLY_TENDENCY — _RT_ROOT resolves once per decision for either flag;
# the pooled _fallback never steers) and prefer the best tendency-weighted expectation,
# per-world expectations averaged over the K belief worlds.
# 🔴 THE MAXIMIN INVARIANT (user directive, non-negotiable): the PRIMARY key remains the
# worst-case minimax value — a move OUTSIDE the epsilon band must NEVER win the tie-break.
# Enforced by construction in _trap_apply: only the within-eps band is ever re-ranked, so the
# chosen move's worst-case guarantee is within eps of optimal by definition; an incumbent with
# no expectation signal is never displaced, and only by a STRICTLY better expectation. This is a
# MOVE tie-break, NOT a leaf re-sign — leaf evaluation is untouched (the felt/secure
# VALUE_LEAF_DYNAMIC family is KILLED by its own addressability scan; do not resurrect it here).
# EPS sizing: root values live on the _state_value scale (prize = 1000, hp = 1; H._score's
# option bands — attacks ~2000 / plays ~350 — are the ORDERING scale, not this one). The root
# spread spans several prizes and the existing tie machinery uses ROBUST_BAND = 500 ("< half a
# prize"); 150 ≈ 2-5% of a multi-prize spread = clearly-safety-equal moves only.
# Flag-off => byte-identical: the trap sink is never threaded (None => _minimax/_world_value
# behave exactly as before), info never carries trap_exp, and the tie-break block in agent() is
# never entered. DIAG: trap_armed / trap_band / trap_changed (armed vs ACTUALLY-changed — the
# 222-armed-vs-0-actual instrumentation lesson).
TRAP_TIEBREAK = False
TRAP_TIEBREAK_EPS = 150.0   # backed-value band counted as "safety-equal" (see EPS sizing above)


def _load_reply_tendencies() -> dict:
    """opp_tendencies.json loader (the _load_matchup_priors multi-candidate path pattern).
    Graceful everywhere: absent/broken file => {} => the lever stays inert (bonus 0)."""
    if not _REPLY_TENDENCIES_TRIED[0]:
        _REPLY_TENDENCIES_TRIED[0] = True
        try:
            here = os.path.dirname(os.path.abspath(__file__))
        except NameError:                  # kaggle_environments execs the source without __file__
            here = os.getcwd()
        for p in ("opp_tendencies.json", os.path.join(here, "opp_tendencies.json"),
                  "src/agent/opp_tendencies.json", "/kaggle_simulations/agent/opp_tendencies.json"):
            try:
                with open(p) as f:
                    _REPLY_TENDENCIES.update(json.load(f) or {})
                break
            except Exception:
                continue
    return _REPLY_TENDENCIES


def _rt_usable(e) -> bool:
    return (isinstance(e, dict) and not e.get("low_confidence")
            and int(e.get("n_decisions") or 0) >= REPLY_TENDENCY_MIN_N)


def _rt_resolve_root():
    """Resolve the belief-MAP archetype to its mined tendencies entry, ONCE per MAIN decision.
    None unless the MAP is a real inference (non-fallback tuple, conf >= REPLY_TENDENCY_MIN_CONF)
    AND a usable mined entry exists: exact label first, else the conservative _ce_comp_match
    component join (label-robust across library rotations); multi-hit joins take the largest-n
    entry. "_"-prefixed keys (incl. the pooled _fallback) never steer."""
    a3 = _OPP_ARCH[0]
    if not (isinstance(a3, tuple) and len(a3) >= 3) or a3[2] or not a3[0]:
        return None
    try:
        if float(a3[1]) < REPLY_TENDENCY_MIN_CONF:
            return None
    except (TypeError, ValueError):
        return None
    tab = _load_reply_tendencies()
    if not tab:
        return None
    key = str(a3[0])
    e = tab.get(key)
    if _rt_usable(e):
        return e
    mine = _ce_join_components(key)
    best = None
    for k, v in tab.items():
        if k.startswith("_") or not _rt_usable(v):
            continue
        theirs = _ce_join_components(k)
        if any(_ce_comp_match(a, b) for a in mine for b in theirs):
            if best is None or int(v.get("n_decisions") or 0) > int(best.get("n_decisions") or 0):
                best = v
    return best


def _rt_phase(turn) -> str:
    """early/mid/late — the w2/opp_predictability phase convention (miner emits the same)."""
    return "early" if turn <= 6 else ("mid" if turn <= 14 else "late")


def _reply_tendency_bonus(opt, cur) -> float:
    """Mined ordering-only bonus for ONE opponent reply candidate at a MIN node. BOOST-ONLY
    (maximin — see the flag block): positive for reply types the archetype favors in this phase
    (p_pick > 0.5), 0.0 for everything else — a rarely-picked type keeps its legacy H._score
    key, never penalized out of the searched set. A gust-class PLAY (resolved via H._option_card
    on the determinized world) uses the gust held-rate cell. 0.0 on any miss (unknown type /
    thin cell / any error — fail-inert)."""
    ent = _RT_ROOT[0]
    if ent is None:
        return 0.0
    try:
        name = _RT_TYPE_NAME.get(int(getattr(opt, "type", -1) or -1))
        if name is None:
            return 0.0
        ph = _rt_phase(_turn_count(cur.current))
        cell = None
        if name == "PLAY":
            cd = H._option_card(cur, opt)
            if cd is not None and _rt_norm_name(getattr(cd, "name", "")) in _RT_GUST_SET:
                g = ent.get("gust") or {}
                cell = (g.get("by_phase") or {}).get(ph) or g.get("overall")
        if cell is None:
            cell = (((ent.get("phases") or {}).get(ph) or {}).get("p_pick") or {}).get(name)
        if not isinstance(cell, dict) or int(cell.get("n") or 0) < REPLY_TENDENCY_MIN_N:
            return 0.0
        return REPLY_TENDENCY_W * max(0.0, float(cell.get("rate")) - 0.5)
    except Exception:
        return 0.0


def _trap_reply_rate(opt, cur):
    """TRAP_TIEBREAK: RAW mined p_pick rate for ONE opponent reply candidate's class (phase
    cell; a gust-class PLAY uses the gust held-rate cell — identical cell resolution to
    _reply_tendency_bonus, but the raw rate UNCLAMPED: the tie-break needs the actual
    likelihood weight, not the boost-only ordering bonus, and it feeds an EXPECTATION over the
    already-searched MIN set, never the MIN ordering or value). None on any miss (no root
    entry / unknown type / thin cell below REPLY_TENDENCY_MIN_N / any error — fail-inert)."""
    ent = _RT_ROOT[0]
    if ent is None:
        return None
    try:
        name = _RT_TYPE_NAME.get(int(getattr(opt, "type", -1) or -1))
        if name is None:
            return None
        ph = _rt_phase(_turn_count(cur.current))
        cell = None
        if name == "PLAY":
            cd = H._option_card(cur, opt)
            if cd is not None and _rt_norm_name(getattr(cd, "name", "")) in _RT_GUST_SET:
                g = ent.get("gust") or {}
                cell = (g.get("by_phase") or {}).get(ph) or g.get("overall")
        if cell is None:
            cell = (((ent.get("phases") or {}).get(ph) or {}).get("p_pick") or {}).get(name)
        if not isinstance(cell, dict) or int(cell.get("n") or 0) < REPLY_TENDENCY_MIN_N:
            return None
        return float(cell.get("rate"))
    except Exception:
        return None


def _trap_apply(scored, cands, trap_exp, eps, best_v, best):
    """TRAP_TIEBREAK band re-rank (pure function — unit-testable without the engine). `scored`
    is the DESC-sorted (backed mean minimax value, cand) list; `trap_exp` maps candidate INDEX
    in `cands` -> tendency-weighted expectation (None = no signal); (best_v, best) = the
    incumbent selection (possibly ROBUST_TIEBREAK-adjusted). Returns (best_v, best, changed).

    🔴 MAXIMIN INVARIANT (user directive): only candidates within `eps` of scored[0][0] — the
    BEST backed value — are ever considered, so a move outside the band CANNOT be returned no
    matter what the tendencies say (the invariant test constructs exactly that case and must
    fail loud if this filter is ever weakened). Further safety: an incumbent with no signal is
    never displaced (nothing to compare), and a challenger must be STRICTLY better on the
    expectation (ties keep the incumbent's better worst-case)."""
    top_v = scored[0][0]
    band = [sv for sv in scored if top_v - sv[0] <= eps]
    if len(band) < 2:
        return best_v, best, False

    def _te(cand):
        try:
            return trap_exp.get(cands.index(cand))
        except ValueError:
            return None

    cur = _te(best)
    if cur is None:
        return best_v, best, False
    win_key, win_sv = None, None
    for sv in band:
        te = _te(sv[1])
        if te is None:
            continue
        key = (te, sv[0])
        if win_key is None or key > win_key:
            win_key, win_sv = key, sv
    if win_sv is None or win_sv[1] is best or win_key[0] <= cur:
        return best_v, best, False
    return win_sv[0], win_sv[1], True


# ================ SYM_DEDUPE (2026-08-09): free width by collapsing OPTION SYMMETRY ==============
# THE DEFECT. The engine emits one option per (card copy x legal target), so a menu is mostly the
# SAME move written down several times. Measured on real logged MAIN menus (ctx MAIN, maxCount==1,
# >=2 options; scripts/probes/probe_sym_dedupe.py --census):
#     episodes_full  25,048 menus / 229,209 options   14.92% are exact duplicates of another option
#     ladder_replays 20,645 menus / 180,612 options   15.57%
#     duplicate options per (episode, turn, seat): mean 8.5 / 5.4, p95 37 / 25, max 218 / 147
# TOP_M=3 then spends up to three of its three slots on ONE move. Collapsing them does not add a
# maximization, does not enumerate a plan, and does not touch any value: it only makes the SAME
# TOP_M prune buy three DISTINCT moves instead of three spellings of one. (Whole-turn global argmax
# over enumerated plans is TREE_SEARCH, measured 0.310/0.335 self-play N=200 -- that is a different
# thing and it stays refuted; this widens the same prune, it does not add a new argmax.)
#
# 🔴 THE IDENTITY KEY IS NOT THE OBVIOUS ONE, AND THE OBVIOUS ONE IS UNSOUND ON THIS ENGINE.
# The natural key -- (type, cardId, area, inPlayArea, inPlayIndex, attackId, playerIndex), i.e. the
# option fields minus the within-zone slot -- is WRONG HERE, because cabt populates NONE of the
# fields that would identify the card. Measured over 229,209 real MAIN options: `cardId` is non-None
# ZERO times, as are serial / playerIndex / count / energyIndex / toolIndex / number /
# specialConditionType. The only populated fields are type (229,209), index (174,254), area
# (105,065), inPlayArea + inPlayIndex (89,653) and attackId (16,369). PLAY options carry `index`
# ALONE, so that key collapses every playable hand card into a single "play something" option; and
# an ATTACH is (area=HAND, index=hand slot, inPlayArea/inPlayIndex=target), so dropping `index`
# without resolving it merges DIFFERENT CARDS onto the same target. It fires constantly, not in
# some corner: 28,167 of the collapse groups in 300 episodes merge different card identities, and
# the first one in the corpus is Basic {R} Energy + Basic {D} Energy onto the same active
# (91121116.json step 6) -- a different energy TYPE, which is the exact failure this key has to
# avoid. So the key resolves the card THROUGH THE OBSERVATION and only then drops the slot:
#     * the slot is dropped ONLY for an interchangeable ZONE (the hand), where a card carries no
#       per-copy state at all -- a hand entry is {id, playerIndex, serial} and nothing else, so two
#       entries with the same id are substitutable by construction;
#     * for an IN-PLAY area (ACTIVE/BENCH/STADIUM) `index` is KEPT, because there it names a BODY,
#       not a copy: two benched copies of one Pokemon differ in HP, damage and attached energy, and
#       collapsing "use bench[0]'s ability" with "use bench[1]'s ability" would be the same bug in
#       a different zone;
#     * anything that cannot be resolved to a card id gets a UNIQUE key and is never collapsed.
#       Fail-safe direction: an unresolved option costs us a duplicate, a wrongly-merged one costs
#       us a move. That also means dedupe is automatically INERT wherever the mover's hand is not
#       materialised in the observation, rather than silently wrong there.
# With the resolved key: 14.9%/15.6% of options collapse and card-identity collisions are 0 of 0
# (probe V4a proves this on the ENGINE, not just on the key: every member of every collapse group
# must step to the same terminal state signature).
#
# ORDERING-ONLY, LIKE REPLY_TENDENCY. This changes WHICH options survive the TOP_M / TOP_Mo prune;
# it never changes a searched option's value, never reorders the survivors among themselves (the
# score-descending order is preserved -- the FIRST, i.e. best-scored, member of each group is the
# representative), and never introduces an option the prune would not otherwise have reached.
# Flag-off => byte-identical (src/agent/test_symdedupe_byte_identical.py).
SYM_DEDUPE = False
# Zones where two entries with the same card id are SUBSTITUTABLE (no per-copy state). The hand is
# the only one we have verified field-by-field on this engine, and it is where ~100% of the measured
# duplicates live (ATTACH-from-hand 78,771 + PLAY 69,189 + EVOLVE-from-hand 10,882 of 174,254
# indexed options). DECK/DISCARD/LOOKING are plausibly substitutable too and are deliberately NOT
# included: they do not appear at a MAIN branch select, so adding them would widen the unsound
# surface for zero measured gain.
_SYM_SWAPPABLE_AREAS = (int(H.AreaType.HAND),)
# every Option field EXCEPT `type`, `area`, `index`, `cardId` (handled explicitly) and `serial`
# (which names a physical copy -- the very thing being collapsed; for an in-play body the kept
# (area, index) already pins it).
_SYM_TAIL_FIELDS = ("number", "playerIndex", "toolIndex", "energyIndex", "count",
                    "inPlayArea", "inPlayIndex", "attackId", "specialConditionType")


def _sym_key(obs, o, pos):
    """Identity key of option `o` at menu position `pos`: same key <=> same MOVE.

    Returns a UNIQUE key ("!", pos) whenever the option names a hand card we cannot resolve to a
    card id, so an unidentifiable option is never merged with anything."""
    t = getattr(o, "type", None)
    area = getattr(o, "area", None)
    idx = getattr(o, "index", None)
    cid = getattr(o, "cardId", None)
    # PLAY carries `index` as a HAND index with `area` unset (SDK: "index (int): Index within the
    # hand"); every other hand-sourced option sets area=HAND explicitly.
    from_hand = idx is not None and ((t == OptionType.PLAY and area is None)
                                     or (area is not None and int(area) in _SYM_SWAPPABLE_AREAS))
    if from_hand:
        if cid is None:
            c = H._resolve(obs, H.AreaType.HAND, idx, getattr(o, "playerIndex", None))
            cid = getattr(c, "id", None)
            if cid is None:
                return ("!", pos)               # unresolvable hand card => never collapse
        idx = None                              # the hand SLOT is interchangeable given the id
    return ((int(t) if t is not None else None), cid,
            (int(area) if area is not None else None), idx) + tuple(
        (int(v) if isinstance(v, int) else v)
        for v in (getattr(o, f, None) for f in _SYM_TAIL_FIELDS))


def _sym_dedupe(obs, sel, order):
    """`order` (a score-DESCENDING list of option indices) with symmetric duplicates removed,
    keeping the FIRST (best-scored) member of each identity class. Order-preserving.

    Any failure returns `order` untouched -- that is exactly the incumbent behaviour, and it is
    COUNTED in DIAG rather than swallowed, because a silently-inert lever is this repo's signature
    failure (opponent_adapt shipped unable to fire and nobody noticed)."""
    try:
        opts = sel.option
        seen, out = set(), []
        for i in order:
            k = _sym_key(obs, opts[i], i)
            if k in seen:
                continue
            seen.add(k)
            out.append(i)
        DIAG["sym_dedupe_fires"] = DIAG.get("sym_dedupe_fires", 0) + 1
        DIAG["sym_dedupe_removed"] = DIAG.get("sym_dedupe_removed", 0) + (len(order) - len(out))
        return out
    except Exception:
        DIAG["sym_dedupe_err"] = DIAG.get("sym_dedupe_err", 0) + 1
        return order


def _reply_order(sel, cur, width, opp_node):
    """TOP-`width` branch options, H._score-ordered (the shipped TOP_M/TOP_Mo prune). At an
    OPPONENT (MIN) node with REPLY_TENDENCY armed (flag ON + a root-resolved mined entry), each
    candidate's SORT KEY adds the mined BOOST-ONLY reply-tendency bonus (maximin: their-favored
    replies may be boosted into TOP_Mo survival; no reply's key ever drops below its legacy
    H._score) — ordering-only: this changes which opponent replies get SEARCHED (TOP_Mo
    survival), never a searched reply's value; the MIN over the searched set is untouched.
    Flag-off takes the identical legacy expression.

    SYM_DEDUPE applies to the SAME sorted order, before the `[:width]` cut: the survivors are the
    top-`width` DISTINCT moves rather than the top-`width` spellings. It composes with
    REPLY_TENDENCY without interacting -- that flag decides the ORDER, this one decides which of
    those are the same move -- and with the flag off the expression is the legacy one."""
    if REPLY_TENDENCY and opp_node and _RT_ROOT[0] is not None:
        DIAG["reply_tendency_fires"] = DIAG.get("reply_tendency_fires", 0) + 1
        order = sorted(range(len(sel.option)),
                       key=lambda i: H._score(sel.option[i], cur)
                       + _reply_tendency_bonus(sel.option[i], cur),
                       reverse=True)
    else:
        order = sorted(range(len(sel.option)),
                       key=lambda i: H._score(sel.option[i], cur), reverse=True)
    if SYM_DEDUPE:
        order = _sym_dedupe(cur, sel, order)
    return order[:width]


def _aggregate_opp(children, scores):
    """Aggregate opponent-reply child values per OPP_MODEL. OPP_MODEL None never calls this (the loop
    keeps its exact MIN). 'min' = adversarial worst-case (value-identical to the default); 'greedy' =
    opponent plays its OWN H-score argmax (exploitable, the low-elo assumption); 'softmax' = expectation
    over softmax(opp_score / tau) (mid/high-elo). children=[(option, our_value)], scores={option: H._score}."""
    m = OPP_MODEL or {}
    mode = m.get("mode", "min")
    if mode == "min" or not scores:
        return min(v for _, v in children)
    if mode == "greedy":
        o = max(children, key=lambda ov: scores.get(ov[0], float("-inf")))[0]
        return dict(children)[o]
    tau = float(m.get("tau", 1.0)) or 1e-6
    sc = [scores.get(o, 0.0) for o, _ in children]
    mx = max(sc)
    ws = [_mth.exp((s - mx) / tau) for s in sc]
    z = sum(ws) or 1.0
    return sum(w * v for w, (_, v) in zip(ws, children)) / z


def _minimax(obs, preds, me: int, seq, d: int, deadline, reply_sink=None,
             alpha: float = float("-inf"), beta: float = float("inf"), sigs=None,
             trap_sink=None) -> float:
    """Minimaxed value (from OUR seat) of the BRANCH line `seq` to depth `d`, alternating
    MAX(us)/MIN(opp). `seq` is a list of branch selects: seq[0]=our first move, then alternating
    best-response moves at each subsequent branch point. `sigs` mirrors `seq` with per-move
    option-identity keys so `_replay` can re-match a stale index in a freshly-randomized fork
    (see the divergence-fix comment above `_opt_key`).

    d<=0 → the current line is finished by a heuristic rollout (the shipped 1-ply leaf when
           seq=[first] and d=0 — byte-identical to the legacy _rollout).
    MIN  → at an OPPONENT branch point, enumerate their TOP_Mo H-scored replies, recurse each,
           take the MIN (a competent opponent best-responds — the upgrade over greedy ROLL_2PLY).
    MAX  → at OUR branch point, enumerate our TOP_M options, recurse each, take the MAX (our counter)."""
    ss = _replay(obs, preds, seq, deadline, sigs=sigs)
    try:
        # after the last branch move, roll the heuristic forward to the NEXT branch point
        ss, st, branchable = _roll_to_branch(ss, deadline)
        if st is None or st.result != -1 or not branchable:
            return _rollout(ss, me, deadline) if (st is not None and st.result == -1) \
                else _state_value(st, me)
        if d <= 0:
            return _rollout(ss, me, deadline)            # depth out → heuristic finishes the line
        cur = ss.observation
        sel = cur.select
        opp_node = (cur.current.yourIndex != me)
        width = TOP_Mo if opp_node else TOP_M
        idx = _reply_order(sel, cur, width, opp_node)   # REPLY_TENDENCY: mined MIN-node ordering
        okeys = {i: (_opt_key(sel.option[i]), _opt_key(sel.option[i], _OPT_KEY_POSFREE))
                 for i in idx}                           # identity keys for child replays
        opp_scores = ({i: H._score(sel.option[i], cur) for i in idx}
                      if (OPP_MODEL is not None and opp_node) else None)
        # TRAP_TIEBREAK: mined reply-class rates for THIS (top-level) MIN node's searched
        # replies — threaded only from _world_value when armed; None => zero extra work.
        trap_rates = ({i: _trap_reply_rate(sel.option[i], cur) for i in idx}
                      if (trap_sink is not None and opp_node) else None)
    finally:
        api.search_end()
    best = None
    best_reply = None
    children = [] if opp_scores is not None else None    # OPP_MODEL: collect opp replies to aggregate
    for o in idx:
        if deadline is not None and time.time() > deadline:
            break
        try:
            v = _minimax(obs, preds, me, seq + [[o]], d - 1, deadline, alpha=alpha, beta=beta,
                         sigs=(list(sigs) if sigs is not None else [None] * len(seq)) + [okeys[o]])
        except Exception:
            _DOM_ERR[0] += 1
            try:
                api.search_end()
            except Exception:
                pass
            continue
        if children is not None:
            children.append((o, v))
        if trap_rates is not None and trap_rates.get(o) is not None:
            trap_sink.append((trap_rates[o], v))         # (their p_pick, our backed value)
        if best is None or (v < best if opp_node else v > best):
            best, best_reply = v, o
        if ALPHA_BETA and children is None:              # enh 6: value-identical pruning (off => full;
            if opp_node:                                 # disabled when aggregating a non-MIN opp model)
                beta = min(beta, best)
            else:
                alpha = max(alpha, best)
            if beta <= alpha:
                break
    if children:                                         # OPP_MODEL aggregation overrides the MIN
        best = _aggregate_opp(children, opp_scores)
    if best is None:                                     # all branches errored → leaf fallback
        try:
            ss = _replay(obs, preds, seq, deadline, sigs=sigs)
            ss, _, _ = _roll_to_branch(ss, deadline)
            return _rollout(ss, me, deadline)
        finally:
            try:
                api.search_end()
            except Exception:
                pass
    if reply_sink is not None and opp_node and best_reply is not None:
        reply_sink.append([best_reply])                  # the opp reply that drove the MIN (trace)
    return best


def _world_value(obs, cand, me: int, seed: int, deadline, reply_sink=None,
                 trap_sink=None) -> float:
    """Minimaxed value of playing first-move `cand` in ONE belief world (preds sampled at `seed`).

    `cand` is a select (e.g. [i]); `seq` is a LIST OF selects, so we seed it with [cand].
    `trap_sink` (TRAP_TIEBREAK, armed only): collects (p_pick, backed value) for the FIRST MIN
    node's searched replies — the root move's punish profile in this world."""
    preds_k = _predict(obs, vary=(seed > 0), seed=seed)
    return _minimax(obs, preds_k, me, [list(cand)], RECURSE_DEPTH, deadline,
                    reply_sink=reply_sink, trap_sink=trap_sink)


TREE_SEARCH = False    # STRUCTURAL lever: explore move SEQUENCES over our turn (a bounded tree
TREE_DEPTH = 2         # branching on TREE_DEPTH of our MAIN decisions, TREE_WIDTH options each),
TREE_WIDTH = 2         # picking the first move that leads to the best achievable line — i.e. search
TREE_NODE_CAP = 64     # BEYOND the greedy heuristic rollout policy (the untested LOCM-winner shape).
DIVERSE_CANDS = False  # if True, the TOP_M candidates are the best option PER OptionType (so
                       # distinct lines — attach vs evolve vs ability vs attack — compete) instead
                       # of TOP_M near-duplicates of the same type (review #5 / SOTA move-pruning).

# --- ENH 2/6/9 (all flag-OFF by default => byte-identical to the shipped dominion) -------------
LETHAL_SHORTCUT = False  # enh 2: if a winning/lethal KO move exists, PLAY it without spending search
                         # budget (also the guard against the ISMCTS "all sims equal -> random" pick
                         # on already-decided boards). The heuristic scores a lethal KO >= 10000.
ALPHA_BETA = False       # enh 6: prune the RECURSE_DEPTH minimax with alpha-beta — value-identical,
                         # fewer engine re-forks, so depth=2 fits budget. Off = full enumeration.
WARM_START = False       # enh 9: memoize the PLAIN prize+hp leaf eval by a cheap board signature
                         # (skipped when STAGE/RICH/LATE eval adds terms the signature can't capture).
_WARM_CACHE: dict = {}   # bounded transposition cache for _state_value (plain eval only)
_WARM_CAP = 50000


def _tree_best(obs, preds, me, first, deadline):
    """Best achievable leaf value over OUR-turn move SEQUENCES that start with `first`.

    Explores up to TREE_WIDTH options at each of TREE_DEPTH branched MAIN decisions; forced/
    multi-select sub-decisions are taken by the heuristic (don't cost depth). Each node re-forks
    from the LIVE state via search_begin and replays the prefix (the engine only forks from root),
    then a heuristic rollout finishes the line. Returns the max leaf value — i.e. the value of
    playing the BEST line whose first move is `first`, not best-first-then-greedy-heuristic."""
    best = [-1e18]
    nodes = [0]

    def rec(seq, d):
        if nodes[0] >= TREE_NODE_CAP or time.time() > deadline:
            return
        nodes[0] += 1
        ss = api.search_begin(obs, *preds, False)
        branch_opts, forced_mv = None, None
        try:
            for mv in seq:
                ss = api.search_step(ss.searchId, mv)
            cur = ss.observation
            st = cur.current
            if st is None or st.result != -1 or st.yourIndex != me or cur.select is None:
                best[0] = max(best[0], _state_value(st, me))      # turn over / terminal
                return
            if d <= 0:
                best[0] = max(best[0], _rollout(ss, me))          # depth out → heuristic to end
                return
            sel = cur.select
            if sel.context != SelectContext.MAIN or sel.maxCount != 1 or len(sel.option) < 2:
                forced_mv = H._choose(cur)                        # forced/multi → heuristic step
            else:
                branch_opts = sorted(range(len(sel.option)),
                                     key=lambda i: H._score(sel.option[i], cur),
                                     reverse=True)[:TREE_WIDTH]
        finally:
            api.search_end()
        if forced_mv is not None:
            rec(seq + [forced_mv], d)                             # forced step: depth unchanged
        elif branch_opts is not None:
            for o in branch_opts:
                rec(seq + [[o]], d - 1)

    rec(list(first), TREE_DEPTH)
    return best[0] if best[0] > -1e17 else None


def _consensus_attack_cost(cd):
    """Min energy-count over the card's attacks (mirrors consensus_mine2._attack_cost), else None."""
    try:
        best = None
        for aid in (cd.attacks or []):
            at = H.ATTACKS.get(aid)
            if at is not None and getattr(at, "energies", None) is not None:
                n = len(at.energies)
                best = n if best is None else min(best, n)
        return best
    except Exception:
        return None


def _role_matchup_assign():
    """S1 per-decision role assignment (ROLE_MATCHUP; see the flag comment at its declaration).
    Returns CONTROL (-1, 1.0) iff the cached belief-MAP read (_OPP_ARCH, set once per decision in
    agent()) is a REAL inference (non-fallback), clears ROLE_MATCHUP_MIN_CONF, AND the mined
    matchup_priors entry for that archetype label says wincon == "stall" — else neutral (0, 0.0)
    (the leaf's role term then contributes exactly 0, plain robust play). Weight is a binary 1.0 on
    gate-pass (the RISK_SLOW_TAU/_deny_evolution_terms binary-gate precedent — NOT matchup_role's
    0.30-0.70 _conf_weight ramp, which would zero this lever in the 0.15-0.30 band its own gate
    admits). Increments the H.DIAG['role_matchup'] FIRES counter on activation. Never raises."""
    try:
        a3 = _OPP_ARCH[0]
        if (isinstance(a3, tuple) and len(a3) >= 3 and not a3[2] and a3[0]
                and (a3[1] or 0.0) >= ROLE_MATCHUP_MIN_CONF):
            pri = _load_matchup_priors().get(str(a3[0]))
            if pri and str(pri.get("wincon", "")).lower() == "stall":
                H.DIAG["role_matchup"] = H.DIAG.get("role_matchup", 0) + 1   # FIRES counter
                return (-1, 1.0)                        # CONTROL: leaf follows _survival_value
    except Exception:
        pass
    return (0, 0.0)


def _matchup_prune_gate() -> bool:
    """S2 archetype gate (MATCHUP_PRUNE): True iff the cached belief-MAP read (_OPP_ARCH, set once
    per decision in agent()) is a REAL inference (non-fallback — mirrors RISK_SLOW_TAU /
    _deny_evolution_terms / ROLE_AWARE), clears MATCHUP_PRUNE_MIN_CONF (the OPP_DECK_MODEL
    convention), and its label contains MATCHUP_PRUNE_FAMILY_KW (the ARCH_PRIZE_RATE_KW keyword
    convention — gate on ARCHETYPE, not exact deck, per project_ptcg_antipilot_challenge). Pure
    read of the root cache; False whenever belief/flags are unavailable."""
    if not MATCHUP_PRUNE:
        return False
    a = _OPP_ARCH[0]
    if not isinstance(a, tuple) or len(a) < 3 or a[2]:       # fallback MAP = a guess, not a read
        return False
    if (a[1] or 0.0) < MATCHUP_PRUNE_MIN_CONF:
        return False
    return MATCHUP_PRUNE_FAMILY_KW in str(a[0] or "").lower()


def _hariyama_family_gate() -> bool:
    """Shared archetype gate for SUMO_EXPOSURE / GUST_SNIPE_PRIORITY: a REAL (non-fallback)
    belief-MAP read at conf >= MATCHUP_PRUNE_MIN_CONF whose label contains
    MATCHUP_PRUNE_FAMILY_KW — the same family definition _matchup_prune_gate() applies (kept as a
    separate helper so these levers arm WITHOUT requiring the MATCHUP_PRUNE flag). Pure read of
    the root _OPP_ARCH cache; False whenever belief/flags are unavailable."""
    a = _OPP_ARCH[0]
    if not isinstance(a, tuple) or len(a) < 3 or a[2]:       # fallback MAP = a guess, not a read
        return False
    if (a[1] or 0.0) < MATCHUP_PRUNE_MIN_CONF:
        return False
    return MATCHUP_PRUNE_FAMILY_KW in str(a[0] or "").lower()


def _sumo_pull_live(opp) -> bool:
    """Engine-true pull precondition (verified: Heave-Ho Catcher is ON-EVOLVE): the opponent has
    an IN-PLAY, not-yet-evolved pre-evolve of an on-evolve-gust evolution (Makuhita today; the
    map is card-DB-derived via H._onevolve_gust_pre, no hardcoded names). Never raises."""
    try:
        pre = H._onevolve_gust_pre()
        if not pre:
            return False
        for pk in (opp.active + opp.bench):
            if pk is None:
                continue
            cd = H.CARDS.get(getattr(pk, "id", None))
            if cd is not None and H._norm_card_name(getattr(cd, "name", "")) in pre:
                return True
    except Exception:
        pass
    return False


def _sumo_evolved_dmg(pk_pre, tgt) -> float:
    """Best payable-within-one-attach damage of `pk_pre`'s PROJECTED on-evolve-gust evolution(s)
    into `tgt` (weakness-aware). The engine-true threat shape: evolving keeps the pre-evolve's
    attached energies, so a Makuhita on 2 {F} is one evolve away from Wild Press 210 (x2 into a
    Fighting-weak target). Ex-immunity guard mirrors _ce_attack_damage (Rock Inn walls ex forms
    only; Hariyama is non-ex so the wall does NOT protect against the verified line)."""
    cd_pre = H.CARDS.get(getattr(pk_pre, "id", None))
    if cd_pre is None:
        return 0.0
    forms = H._onevolve_gust_pre().get(H._norm_card_name(getattr(cd_pre, "name", "")))
    if not forms:
        return 0.0
    cdt = H.CARDS.get(getattr(tgt, "id", None))
    best = 0.0
    for cd2 in forms:
        walled = False
        if (getattr(cd2, "ex", False) or getattr(cd2, "megaEx", False)):
            shim = type("_Pk", (), {"id": cd2.cardId})()     # projected attacker for the ex wall
            if H._wall_blocks_all(tgt, shim):
                continue
            walled = H.ROCK_INN_SCOPED and H._ex_immune_vs(tgt, shim)
        for aid in (getattr(cd2, "attacks", None) or []):
            atk = H.ATTACKS.get(aid)
            if atk is None or (atk.damage or 0) <= 0 or not _ce_payable_plus1(pk_pre, atk):
                continue
            if walled and not H._attack_pierces_effects(atk):
                continue
            dmg = float(atk.damage)
            if cdt is not None and cdt.weakness is not None and cd2.energyType == cdt.weakness:
                dmg *= 2
            if dmg > best:
                best = dmg
    return best


def _sumo_exposure_term(mine, opp) -> float:
    """SUMO_EXPOSURE leaf term (called only when the flag is ON and _SUMO_ROOT armed — see the
    flag block for the verified mechanics + intel discrepancies). 0.0 unless the pull is LIVE
    (in-play pre-evolve on their board in THIS leaf). Penalty = SUMO_EXPOSURE_W x
    (prizes_for_ko − 1) of our most-exposed MULTI-prize benched piece that their board — real
    attackers via _ce_attack_damage, plus the projected evolved form of the pull piece itself —
    can OHKO after the pull. 🔴 (prizes − 1) is the never-suppress-benching boundary: a 1-prize
    body and an empty bench are ALWAYS penalty-free, so the term re-ranks WHICH multi-prize
    piece sits on the bench, never WHETHER we bench. Never raises (any error => 0)."""
    try:
        if not _sumo_pull_live(opp):
            return 0.0
        att_hand = getattr(opp, "handCount", 0) or 0
        def_hand = getattr(mine, "handCount", 0) or 0
        attackers = [a for a in (opp.active + opp.bench) if a is not None]
        pre = H._onevolve_gust_pre()
        pulls = [pk for pk in attackers
                 if H._norm_card_name(getattr(H.CARDS.get(getattr(pk, "id", None)), "name", "")
                                      or "") in pre]
        worst = 0
        for b in mine.bench:
            if b is None:
                continue
            pr = H._prizes_for_ko(b)
            if pr < 2:
                continue                                     # boundary: 1-prize bodies exempt
            bhp = getattr(b, "hp", 0) or 0
            hit = any(_ce_attack_damage(a, b, att_hand, def_hand) >= bhp for a in attackers) \
                or any(_sumo_evolved_dmg(p, b) >= bhp for p in pulls)
            if hit:
                worst = max(worst, pr - 1)
        if worst:
            DIAG["sumo_leaf_fired"] = DIAG.get("sumo_leaf_fired", 0) + 1
        return -SUMO_EXPOSURE_W * worst
    except Exception:
        return 0.0


def _consensus_option_idx(obs):
    """Layer-A pro-CONSENSUS PRIOR detector on the LIVE SDK observation — the SDK mirror of
    consensus_mine2.detect(). Returns {rule_name: set(option_index)} for the rules whose consensus
    move(s) are AVAILABLE at this MAIN single-select decision. Used by the PRUNE-SURVIVAL mechanism
    (raise the H._score sort key of these indices so they clear TOP_M). Detectors are deck-agnostic
    and structural — identical logic to the offline mine so the in-agent prior matches what was mined.
    Pure read; any error => empty (defer to the unmodified search)."""
    out = {}
    try:
        sel = obs.select
        if sel is None or int(sel.context) != int(SelectContext.MAIN) or sel.maxCount != 1:
            return out
        opts = sel.option
        if not opts or len(opts) < 2:
            return out
        st = obs.current
        me = st.players[st.yourIndex]
        opp = st.players[1 - st.yourIndex]
        hand = me.hand or []
        bench = me.bench or []
        # benchMax (engine field; default 5 if absent)
        bmax = getattr(me, "benchMax", None)
        if bmax is None:
            bmax = 5
        bench_open = sum(1 for b in bench if b) < bmax

        bench_set, attach_set, ko_set, evolve_set = set(), set(), set(), set()
        bench_attach_set = set()  # subset of attach_set targeting the BENCH (T3 mechanism 2, see below)
        end_set = set()
        supporter_play = set()
        # SEQUENCING (2026-07-21, Kaggle discussion review): a top-1% real-world PTCG player's stated
        # rule -- when a generic no-target draw AND a named-target search/tutor are BOTH legal in the
        # same decision, draw first (preserves the option value of what the draw might reveal before
        # committing the tutor to a specific target). Text classification mirrors _skill_value's own
        # patterns exactly (heuristic_agent.py) so this detector agrees with what the flat scorer would
        # call "draw" vs "search" -- not a new heuristic, a structural read of the same signal.
        draw_set, tutor_set = set(), set()
        # HIGH_VALUE_ABILITY (2026-07-21, memory project_ptcg_ability_divergence): see the
        # POLICY_PRIOR_PRUNE dict comment above for the full mechanism. Reuses the `cd` already
        # fetched in the ABILITY/PLAY branch below -- no extra H._option_card() call.
        high_value_ability_set = set()
        # S2 MATCHUP_PRUNE (2026-07-22): archetype gate + the FUELED opponent attacker TYPES,
        # computed ONCE per decision (option-invariant, like the bench/benchMax reads above).
        # "Fueled attacker" = any opp in-play Pokemon with >=1 attached energy whose card has a
        # damaging attack; its card energyType is what the weakness lookup in the ATTACH branch
        # below compares against (weakness applies off the ATTACKER's Pokemon type, the
        # H._opp_can_ko convention: cd.energyType == cdm.weakness). Flag-guarded CALL (the
        # OPP_DECK_MODEL sabotage-test convention): with MATCHUP_PRUNE off the helper is NEVER
        # invoked, so even a broken gate cannot disturb the other rules' detection.
        mp_gate = MATCHUP_PRUNE and _matchup_prune_gate()
        mp_threat_types = set()
        if mp_gate:
            for pk in (opp.active or []) + (opp.bench or []):
                if pk is None or len(getattr(pk, "energies", []) or []) < 1:
                    continue
                cdo = H.CARDS.get(getattr(pk, "id", None))
                if cdo is None:
                    continue
                atk_o = H._strongest_attack(cdo)
                if atk_o is not None and (getattr(atk_o, "damage", 0) or 0) > 0:
                    et = getattr(cdo, "energyType", None)
                    if et is not None:
                        mp_threat_types.add(et)
        mp_feed_set, mp_snipe_set = set(), set()
        for j, o in enumerate(opts):
            t = getattr(o, "type", None)
            if t == OptionType.END:
                end_set.add(j)
            elif t == OptionType.PLAY:
                hi = getattr(o, "index", None)
                if hi is not None and 0 <= hi < len(hand):
                    cd = H.CARDS.get(getattr(hand[hi], "id", None))
                    if bench_open and _is_basic_pokemon(cd):
                        bench_set.add(j)
                    if cd is not None and getattr(cd, "cardType", None) == H.CardType.SUPPORTER:
                        supporter_play.add(j)
            if t in (OptionType.ABILITY, OptionType.PLAY):
                cd = H._option_card(obs, o)
                sk = getattr(cd, "skills", None) if cd else None
                if sk:
                    txt = " ".join((getattr(s, "text", "") or "") for s in sk).lower()
                    is_tutor = "search your deck" in txt
                    is_draw = (bool(re.search(r"draw (\d+) card", txt))
                               or ("draw" in txt and "card" in txt))
                    if is_tutor:
                        tutor_set.add(j)
                    elif is_draw:
                        draw_set.add(j)
                if t == OptionType.ABILITY and cd is not None and H._skill_value(cd) >= HIGH_VALUE_ABILITY_MIN:
                    high_value_ability_set.add(j)
            elif t in (OptionType.ATTACH, OptionType.ENERGY, OptionType.ENERGY_CARD):
                tgt = H._resolve(obs, getattr(o, "inPlayArea", None),
                                 getattr(o, "inPlayIndex", None), st.yourIndex)
                if tgt is not None:
                    cd = H.CARDS.get(getattr(tgt, "id", None))
                    need = _consensus_attack_cost(cd) if cd else None
                    have = len(getattr(tgt, "energies", []) or [])
                    if need is not None and have < need:   # not-yet-payable attacker -> fuel it
                        attach_set.add(j)
                        if getattr(o, "inPlayArea", None) == H.AreaType.BENCH:
                            bench_attach_set.add(j)  # T3 mechanism 2 subset — see rule comment below
                if mp_gate and tgt is not None and mp_threat_types:
                    # S2 rule (a) matchup_avoid_feed: this attach targets one of OUR Pokemon that a
                    # fueled opponent attacker weakness-DOUBLES and whose KO concedes >=2 prizes —
                    # the feed line (deck-general: card-DB weakness + prizes_for_ko, no names).
                    cdt = H.CARDS.get(getattr(tgt, "id", None))
                    wk = getattr(cdt, "weakness", None) if cdt is not None else None
                    if wk is not None and wk in mp_threat_types and H._prizes_for_ko(tgt) >= 2:
                        mp_feed_set.add(j)
            elif t == OptionType.ATTACK:
                act = opp.active[0] if opp.active else None
                dhp = getattr(act, "hp", None) if act is not None else None
                at = H.ATTACKS.get(getattr(o, "attackId", None))
                dmg = getattr(at, "damage", None) if at is not None else None
                if dhp is not None and dmg is not None and dmg > 0 and dmg >= dhp:
                    ko_set.add(j)
                    # S2 rule (b) matchup_snipe: the one-shot lands on a LOW-HP piece (the
                    # breaker/support band) — same raw-damage convention as take_ko above.
                    if mp_gate and dhp <= MATCHUP_SNIPE_HP_MAX:
                        mp_snipe_set.add(j)
            elif t == OptionType.EVOLVE:
                hi = getattr(o, "index", None)
                evo = H.CARDS.get(getattr(hand[hi], "id", None)) if (hi is not None and 0 <= hi < len(hand)) else None
                tgt = H._resolve(obs, getattr(o, "inPlayArea", None),
                                 getattr(o, "inPlayIndex", None), st.yourIndex)
                if evo is not None and tgt is not None:
                    if (getattr(evo, "hp", 0) or 0) > (getattr(tgt, "maxHp", 0) or 0):
                        evolve_set.add(j)
        if bench_set:
            out["bench_develop"] = bench_set
        if attach_set:
            out["attach_to_attacker"] = attach_set
        if bench_attach_set:
            out["bench_attach"] = bench_attach_set
        if ko_set:
            out["take_ko"] = ko_set
        if evolve_set:
            out["evolve_better"] = evolve_set
        if draw_set and tutor_set:
            out["sequencing_draw_first"] = draw_set
        if high_value_ability_set:
            out["high_value_ability"] = high_value_ability_set
        if mp_feed_set:
            out["matchup_avoid_feed"] = mp_feed_set
        if mp_snipe_set:
            out["matchup_snipe"] = mp_snipe_set
        if mp_feed_set or mp_snipe_set:   # FIRES counter (ship_gate/inert-SUB-28 check): a matchup
            DIAG["matchup_prune_fires"] = DIAG.get("matchup_prune_fires", 0) + 1   # rule EMITTED
        # no_idle_pass: an END exists AND development (bench/attach/evolve/supporter) is available =>
        # consensus = any NON-END option (don't idle-pass with live development).
        if end_set:
            dev = bench_set | attach_set | evolve_set | supporter_play
            if dev:
                out["no_idle_pass"] = {j for j in range(len(opts)) if j not in end_set}
    except Exception:
        return {}
    return out


def _policy_prior_prune_bonus(obs, i, _rules=None):
    """PRUNE-SURVIVAL: additive bonus to option `i`'s H._score sort key in _candidates() so a
    consensus-matching option SURVIVES the TOP_M prune and reaches the leaf. Sum of per-rule
    POLICY_PRIOR_PRUNE weights over the rules that option i satisfies. Ordering-only (never changes
    a searched move's value). 0 when POLICY_PRIOR is off OR no rule fires => byte-identical.
    MATCHUP_PRUNE (S2, 2026-07-22) also opens this path so its archetype-gated rules can act
    standalone: with MATCHUP_PRUNE on and POLICY_PRIOR off the OTHER rules are still detected but
    their weights stay 0.0 (module defaults) => +0.0 on every sort key => ordering unchanged.
    PERF: `_rules` lets the caller pass the once-computed _consensus_option_idx(obs) (invariant across
    i) so _candidates() does not recompute it per option (O(n^2)->O(n), value-identical)."""
    if not (POLICY_PRIOR or MATCHUP_PRUNE):
        return 0.0
    rules = _consensus_option_idx(obs) if _rules is None else _rules
    if not rules:
        return 0.0
    b = 0.0
    for name, idxs in rules.items():
        if i in idxs:
            b += POLICY_PRIOR_PRUNE.get(name, 0.0)
    return b


# =====================================================================================
# SIB_RANKER (2026-08-10) — the TEACHER-PREFERENCE SIBLING RANKER as a PRIOR OVER THE
# INCUMBENT'S TOP-3, and nothing more.
#
# WHY. Held-out on 400,317 symmetric-deduped teacher preference pairs (1,790 episodes,
# 11 teachers Elo 910-1066 piloting our exact 60 cards, episode-disjoint 5-fold,
# episode-clustered CIs): the linear ranker orders pairs at 0.7710 [0.7662, 0.7752]
# vs the incumbent H._score's 0.6525 [0.6463, 0.6580]; on SAME-TYPE pairs 0.7514 vs
# 0.5606 — WITHIN an option class the incumbent is at chance, which is exactly the
# contrast the plain prize+HP leaf also cannot see (sibling attach/play targets tie).
#
# WHY ONLY A RE-RANK. Per-teacher accuracy does not price teacher SKILL (Spearman rho
# vs leaderboard +0.155, p≈0.66), so the score must never outrank a backed search
# value. Wiring, all in agent() at the ROOT decision only:
#   * search path — _sib_rerank permutes the order of the top-3 candidate list BEFORE
#     the world loop. The backed minimax value still decides; because agent()'s final
#     sort is STABLE, the ranker's order becomes the tie-break among value-TIED
#     candidates (the energy-blind leaf ties sibling attach/play targets constantly —
#     precisely the pairs the ranker wins). It never adds, drops, or revalues anything.
#   * heuristic path (budget < MIN_BUDGET_S / no search_begin_input) — when H._choose
#     picked the incumbent's own #1, the ranker's preferred member of the SAME top-3
#     is returned instead. If _choose picked anything else (its s>0/minCount guards),
#     its pick stands untouched.
# CONTAINMENT BY CONSTRUCTION: only the first min(K, 3, len) entries of the incumbent
# list are permuted — a candidate outside the incumbent's top-3 can never be promoted;
# non-MAIN contexts never reach this code (_candidates returns None); rollout/minimax
# internals never call it (root wiring only => at most ONE ranker call per MAIN
# decision, on <=3 options). Flag-off => byte-identical (helper never called, no DIAG
# keys; src/agent/test_sibranker_byte_identical.py proves it against a pre-patch
# snapshot).
# DIAG FIRES: sib_ranker (ran) / sib_ranker_reord (order changed) / sib_ranker_front
# (front changed) / sib_ranker_heur_swap (heuristic-path pick actually swapped) /
# sib_ranker_abstain / sib_ranker_err / sib_ranker_dead (armed but module missing).
# =====================================================================================
SIB_RANKER = False     # default OFF — config key "sib_ranker": [enabled, k] (k pinned to 3)
SIB_RANKER_K = 3       # re-rank width; hard-capped at 3 in _sib_rerank regardless of config


def _sib_rerank(obs, cands):
    """Permute the first min(SIB_RANKER_K, 3) entries of `cands` by descending ranker score
    (stable: ties keep the incumbent H._score order). Returns `cands` itself on any abstention,
    error, or missing module — the incumbent order is always the fallback, never a partial one."""
    if _RANK_PRIOR is None:
        DIAG["sib_ranker_dead"] = DIAG.get("sib_ranker_dead", 0) + 1   # armed but unbundled — LOUD
        return cands
    k = min(int(SIB_RANKER_K), 3, len(cands))     # containment: never touch beyond the top-3
    if k < 2:
        return cands
    try:
        opts = obs.select.option
        scores = []
        for c in cands[:k]:
            s = _RANK_PRIOR.score(opts[c[0]], obs, H.CARDS, H.ATTACKS)
            if s is None:                          # extraction abstained: do not half-rank
                DIAG["sib_ranker_abstain"] = DIAG.get("sib_ranker_abstain", 0) + 1
                return cands
            scores.append(s)
        DIAG["sib_ranker"] = DIAG.get("sib_ranker", 0) + 1             # FIRES: ranker ran
        order = sorted(range(k), key=lambda j: (-scores[j], j))
        if order == list(range(k)):
            return cands
        DIAG["sib_ranker_reord"] = DIAG.get("sib_ranker_reord", 0) + 1
        if order[0] != 0:
            DIAG["sib_ranker_front"] = DIAG.get("sib_ranker_front", 0) + 1
        return [cands[j] for j in order] + cands[k:]
    except Exception:
        DIAG["sib_ranker_err"] = DIAG.get("sib_ranker_err", 0) + 1     # never silent (dead-lever class)
        return cands


def _candidates(obs):
    """TOP_M single-option first-moves for a MAIN maxCount==1 decision, else None."""
    sel = obs.select
    if sel.context != SelectContext.MAIN or sel.maxCount != 1 or len(sel.option) < 2:
        return None
    # PERF (value-identical): hoist the option-invariant consensus scan out of the per-option sort key
    # so it is computed ONCE per decision, not once per option (O(n^2)->O(n)); same sort keys => same order.
    _pp_on = POLICY_PRIOR or MATCHUP_PRUNE   # S2: MATCHUP_PRUNE opens the prune path standalone
    _pp_rules = _consensus_option_idx(obs) if _pp_on else None
    _ppb = (lambda i: _policy_prior_prune_bonus(obs, i, _pp_rules)) if _pp_on else (lambda i: 0.0)
    idx = sorted(range(len(sel.option)),
                 key=lambda i: H._score(sel.option[i], obs) + _ppb(i), reverse=True)
    # SYM_DEDUPE: drop options that are the SAME MOVE as an earlier (better-scored) one, so the
    # TOP_M cut below buys TOP_M distinct moves. Applied to the already-sorted list and
    # order-preserving, so with the flag off -- or on a menu with no duplicates -- `idx` is
    # unchanged and every downstream path (DIVERSE_CANDS, the [:TOP_M] cut, _search_timeaware's
    # cands.index() lookups) sees exactly what it saw before.
    if SYM_DEDUPE:
        idx = _sym_dedupe(obs, sel, idx)
    if not DIVERSE_CANDS:
        return [[i] for i in idx[:TOP_M]]
    picks, seen = [], set()           # best per OptionType first (structurally distinct lines)...
    for i in idx:
        t = sel.option[i].type
        if t not in seen:
            seen.add(t)
            picks.append(i)
        if len(picks) >= TOP_M:
            break
    for i in idx:                      # ...then fill remaining slots by raw score
        if len(picks) >= TOP_M:
            break
        if i not in picks:
            picks.append(i)
    return [[i] for i in picks]


def _lethal_move(obs):
    """enh 2: the first immediate-LETHAL option ([i]) at a MAIN decision, else None. The heuristic
    scores a knockout (damage >= target HP) at >= 10000; playing it ends our turn with a KO/prize and
    is almost always correct, so we can skip search. Used only when LETHAL_SHORTCUT is on."""
    sel = obs.select
    if sel is None or sel.context != SelectContext.MAIN or sel.maxCount != 1:
        return None
    best_i, best_s = None, 10000.0
    for i, opt in enumerate(sel.option):
        if getattr(opt, "type", None) == OptionType.ATTACK:
            s = H._score(opt, obs)
            if s >= best_s:
                best_i, best_s = i, s
    return [best_i] if best_i is not None else None


G1_CLOSE = False       # G1 slice 2 (WINRATE70 §7): match-point CLOSE ENUMERATOR — multi-step
G1_CLOSE_PREFIX = 6    # this-turn lines (play/attach/ability prefix -> lethal) on the engine
G1_CLOSE_STEPS = 3     # forward model. Facts, not values; flag-off => byte-identical.


def _g1_close_line(obs, deadline):
    """Match-point CLOSE ENUMERATOR (Class-A fact; RESEARCH G1/T1): the 1-ply _lethal_move sees
    only single-attack closes — 13/26 autopsied close-out losses sat at match point where a
    MULTI-STEP line (Boss-then-attack, attach-then-attack, ability-then-attack) closed. BFS over
    non-attack play prefixes via the engine fork; a prefix whose end state offers an immediate
    lethal (or wins outright) returns the PREFIX'S FIRST MOVE. None => no closing line found.
    Cost: <= PREFIX^STEPS forks at match-point decisions only (~tens of search_steps)."""
    st = obs.current
    sel = obs.select
    if sel is None or sel.context != SelectContext.MAIN or sel.maxCount != 1:
        return None
    me = st.yourIndex
    if len(st.players[me].prize) > 3:
        return None                                     # not our match point (one line can't close)
    if _lethal_move(obs) is not None:
        return None                                     # 1-ply close exists; normal path takes it
    preds = _predict(obs)                               # MAP world: prefixes only play OUR cards,
    root = [i for i, o in enumerate(sel.option)         # so world choice barely matters here
            if getattr(o, "type", None) not in (OptionType.ATTACK, OptionType.END)]
    frontier = [[[i]] for i in root[:G1_CLOSE_PREFIX]]
    for depth in range(G1_CLOSE_STEPS):
        nxt = []
        for seq in frontier:
            if deadline is not None and time.time() > deadline:
                return None
            try:
                ss = _replay(obs, preds, seq, deadline)
                try:
                    ss, st2, branchable = _roll_to_branch(ss, deadline)
                    if st2 is None:
                        continue
                    if st2.result != -1:
                        if st2.result == me:
                            return list(seq[0])         # the line WON outright
                        continue
                    if st2.yourIndex != me:
                        continue                        # our turn ended without a close
                    o2 = ss.observation
                    if _lethal_move(o2) is not None:
                        return list(seq[0])             # prefix opens an immediate lethal: CLOSE
                    if branchable and depth < G1_CLOSE_STEPS - 1:
                        s2 = o2.select
                        adds = [i for i, o in enumerate(s2.option)
                                if getattr(o, "type", None) not in (OptionType.ATTACK,
                                                                    OptionType.END)]
                        for a in adds[:G1_CLOSE_PREFIX]:
                            nxt.append(seq + [[a]])
                finally:
                    try:
                        api.search_end()
                    except Exception:
                        pass
            except Exception:
                continue
        frontier = nxt
        if not frontier:
            break
    return None


def _turn_count(st) -> int:
    """Best-effort game turn counter (engine field name varies; 0 if unavailable)."""
    for attr in ("turn", "turnCount", "turnNumber"):
        v = getattr(st, attr, None)
        if isinstance(v, int):
            return v
    return 0


def _stakes_factor(obs, cands, me: int) -> float:
    """Spend MORE on high-stakes/branchy decisions, LESS on forced/decided ones. ∈[STAKES_LO,STAKES_HI].

    Levers (multiplicative around 1.0): near-lethal either way (_ko_potential / _opp_can_ko) bumps
    spend (the decision swings the prize race); a large candidate set (branchy) bumps spend; a tiny
    candidate set (near-forced) cuts it. Clamped to the [LO,HI] band so it only scales the share."""
    st = obs.current
    mine, opp = st.players[me], st.players[1 - me]
    ma = mine.active[0] if mine.active else None
    oa = opp.active[0] if opp.active else None
    f = 1.0
    # near-lethal in EITHER direction = high stakes (we can KO, or we're about to be KO'd)
    if (ma and oa) and (_ko_potential(mine, opp) >= 1.0 or H._opp_can_ko(oa, ma, _opp_blocked(opp))):
        f *= 1.6
    # branchiness: many real candidate lines → more to resolve; very few → near-forced
    nc = len(cands)
    if nc >= 6:
        f *= 1.4
    elif nc >= 4:
        f *= 1.15
    elif nc <= 2:
        f *= 0.6
    return max(STAKES_LO, min(STAKES_HI, f))


def _is_stall(obs, me: int) -> bool:
    """A grindy stall (high turn count, neither side closing) → bank time (SPEEDRUN)."""
    st = obs.current
    mine, opp = st.players[me], st.players[1 - me]
    return (_turn_count(st) >= STALL_TURN
            and len(mine.prize) >= STALL_PRIZES_LEFT and len(opp.prize) >= STALL_PRIZES_LEFT)


def _speedrun(obs, me: int, budget: float) -> bool:
    """SPEEDRUN when the pool is nearly drained OR the game is a stall: take ~min samples and stop,
    banking the remaining pool so a later high-stakes decision can spend it (and so we never time out)."""
    return (budget or 0.0) < POOL_FLOOR_S or _is_stall(obs, me)


def _decision_deadline(obs, me: int, t0: float, budget: float, cands=None) -> float:
    """Adaptive per-decision deadline that SPENDS the overage pool without risking timeout.

    share = BUDGET_RESERVE × remainingOverage / est_SEARCHABLE_decisions, scaled by _stakes_factor,
    where est_searchable ≈ our prizes-left × DECISIONS_PER_PRIZE (floored at EST_DECISIONS_MIN) — only
    the decisions worth searching, NOT all ~88 raw actions, so the share is the real per-move budget.
    per_move is clamped to ≤ HARD_WALL_S (the true safety ceiling) AND ≤ (budget − POOL_FLOOR_S) so we
    never drain the pool below the floor in one move → DeadlineExceeded is structurally impossible.
    On SPEEDRUN (stall / near-empty pool) per_move collapses to a tiny floor (bank time)."""
    st = obs.current
    mine = st.players[me]
    b = budget or 0.0
    if _speedrun(obs, me, b):
        return t0 + 0.05                                   # SPEEDRUN: min-sample then stop, bank time
    est = max(EST_DECISIONS_MIN, len(mine.prize) * DECISIONS_PER_PRIZE)
    share = BUDGET_RESERVE * b / est
    if cands is not None:
        share *= _stakes_factor(obs, cands, me)
    per_move = min(HARD_WALL_S, share, max(0.05, b - POOL_FLOOR_S))   # clamp ≤ ceiling AND keep floor
    return t0 + max(per_move, 0.02)


def _max_world_rounds(minimaxing: bool) -> int:
    """Round cap for the K-world loop — THE clamp TIME_FILL removes.

    Measured 2026-07-30 (n=105 ladder games): with the cap at K_WORLDS=6 we used 2.3s of the 600s
    overage pool per GAME (0.4%, max 4.4s), while every stronger Elo band thinks 10-16x longer
    (700-900: mean 20.9s; >=1050: 37.7s, p90 121s, 16%% of games >60s). The deadline machinery in
    the loop below only ever CUTS SHORT; this cap was the reason nothing ever scaled UP to fill
    the per-decision share (~12s under HARD_WALL_S). Under TIME_FILL, K_WORLDS becomes the MINIMUM
    (SAMPLE_MIN_PER_CAND still guarantees coverage) and the existing per-round deadline check does
    the limiting; TIME_FILL_MAX_ROUNDS is a runaway backstop, not a target. Composes with
    WIN_VALUE: on a bounded P(win) leaf the many-world mean is a real expectation whose noise
    shrinks as 1/sqrt(rounds). Flag-off => byte-identical (cap stays K_WORLDS)."""
    if not minimaxing:
        return 10 ** 9
    return TIME_FILL_MAX_ROUNDS if TIME_FILL else K_WORLDS


TIME_FILL = False           # spend the pool: world loop becomes time-limited, not count-limited
TIME_FILL_MAX_ROUNDS = 64   # backstop cap on belief worlds per decision (deadline is the real limit)


def _search_timeaware(obs, cands, preds, me: int, t0: float, budget: float):
    """Round-robin sample across `cands` until the adaptive deadline (the K-world expectation loop).

    Each round draws a fresh belief WORLD (seed = round index) and evaluates every candidate in it:
      - RECURSE_DEPTH == 0 → the shipped 1-ply leaf (search_begin → step → _rollout), byte-identical;
      - RECURSE_DEPTH  > 0 → _world_value = a depth-d MINIMAX in that world (MIN over opp top-Mo
        replies / MAX our counter). Averaging the per-candidate values over the rounds = a
        belief-averaged (PIMC/ISMCTS) minimaxed value. K_WORLDS caps how many worlds we draw.

    Returns (scored, rolls, stopped_early, info) where info carries per-candidate value lists (world
    spread) and the captured opponent MIN-reply line, for the rich trace. Every candidate is still
    guaranteed SAMPLE_MIN_PER_CAND samples, and the deadline is threaded into every rollout so a
    single deep line can never overrun the pool."""
    deadline = _decision_deadline(obs, me, t0, budget, cands)
    minimaxing = RECURSE_DEPTH > 0
    det = minimaxing or (DETERMINIZATIONS > 1) or BELIEF or ROLL_TO_END or ROLL_2PLY
    nC = len(cands)
    sums = [0.0] * nC
    counts = [0] * nC
    vlists = [[] for _ in range(nC)]      # per-candidate per-world values (trace: spread)
    reply_line = [None]                   # opponent MIN-reply that drove the chosen value (trace)
    # TRAP_TIEBREAK: armed only with the flag ON, a minimaxing search AND a resolved mined
    # entry (_RT_ROOT — the shared REPLY_TENDENCY loader/belief/conf gates). Off => the sink is
    # never threaded (None) and info never carries trap_exp: byte-identical.
    trap_on = TRAP_TIEBREAK and minimaxing and _RT_ROOT[0] is not None
    tsum = [0.0] * nC                     # per-candidate summed per-world expectations
    tn = [0] * nC
    rolls = 0
    rnd = 0
    stopped_early = False
    max_rounds = _max_world_rounds(minimaxing)
    while rnd < max_rounds:
        did = False
        for ci in range(nC):
            if time.time() >= deadline and rnd >= SAMPLE_MIN_PER_CAND:
                break
            try:
                if minimaxing:
                    sink = []
                    tsink = [] if trap_on else None
                    v = _world_value(obs, cands[ci], me, rnd, deadline, reply_sink=sink,
                                     trap_sink=tsink)
                    if tsink:                                # tendency-weighted punish profile
                        z = sum(r for r, _ in tsink)
                        if z > 0:
                            tsum[ci] += sum(r * val for r, val in tsink) / z
                            tn[ci] += 1
                    if sink:
                        reply_line[0] = sink                 # last world's opp reply line (trace)
                else:
                    preds_k = _predict(obs, vary=(rnd > 0), seed=rnd) if det else preds
                    ss = api.search_begin(obs, *preds_k, False)
                    ss = api.search_step(ss.searchId, cands[ci])
                    v = _rollout(ss, me, deadline)
                    api.search_end()
                sums[ci] += v
                vlists[ci].append(v)
                counts[ci] += 1
                rolls += 1
                did = True
            except Exception:
                _DOM_ERR[0] += 1
                try:
                    api.search_end()
                except Exception:
                    pass
                continue
        rnd += 1
        if time.time() >= deadline and rnd >= SAMPLE_MIN_PER_CAND:
            break
        if not did:                                  # all candidates erroring → bail to caller
            break
        if EARLY_STOP_MARGIN is not None and min(counts) >= 3:
            ms = sorted((sums[i] / counts[i] for i in range(nC) if counts[i]), reverse=True)
            if len(ms) >= 2 and ms[0] - ms[1] > _scaled_margin(EARLY_STOP_MARGIN):
                stopped_early = True
                break
    if TIME_FILL and minimaxing:                 # FIRES evidence: mean rounds/decision >> K_WORLDS
        DIAG["tf_rounds"] = DIAG.get("tf_rounds", 0) + rnd
        DIAG["tf_decisions"] = DIAG.get("tf_decisions", 0) + 1
    scored = [(sums[i] / counts[i], cands[i]) for i in range(nC) if counts[i] > 0]
    info = {"vlists": vlists, "reply_line": reply_line[0]}
    if trap_on:                           # candidate INDEX -> mean expectation (None = no signal)
        info["trap_exp"] = {i: (tsum[i] / tn[i] if tn[i] else None) for i in range(nC)}
    return scored, rolls, stopped_early, info


def _describe_opt(obs, cand) -> dict:
    """Human-readable summary of a candidate first-move option (type + target card if resolvable)."""
    try:
        opt = obs.select.option[cand[0]]
    except Exception:
        return {"idx": cand}
    d = {"idx": cand[0], "type": str(getattr(opt, "type", None)).replace("OptionType.", "")}
    card = H._option_card(obs, opt)
    if card is None:
        card = H._resolve(obs, getattr(opt, "area", None), getattr(opt, "index", None),
                          getattr(opt, "playerIndex", None))
        card = H.CARDS.get(card.id) if (card is not None and getattr(card, "id", None)) else None
    if card is not None:
        d["card"] = getattr(card, "name", None)
    aid = getattr(opt, "attackId", None)
    if aid is not None and aid in H.ATTACKS:
        d["attack"] = getattr(H.ATTACKS[aid], "name", None)
    return d


def _belief_guess(obs) -> dict | None:
    """Belief deck-guess for the trace: MAP archetype + posterior top-3 (archetype, weight)."""
    if belief is None:
        return None
    try:
        post = belief.posterior(obs)
        if not post:
            return {"map": None, "top3": []}
        top = sorted(post, key=lambda dw: -dw[1])[:3]
        return {"map": [top[0][0].get("archetype"), round(top[0][1], 3)],
                "top3": [[d.get("archetype"), round(w, 3)] for d, w in top]}
    except Exception:
        return None


def _trace_dp() -> int:
    """Decimal places for traced values. Two is plenty on the material scale (one prize = 1000) and
    destroys the corpus on the probability scale, where every candidate in a real decision differs
    in the 4th place and would round to the same 0.01 bucket — recording "all candidates exactly
    tied" for decisions that were not. The decision trace is the only per-decision record we keep.

    LEAF_V3 AUDIT (site 4 of 5): NO CHANGE REQUIRED, checked arithmetically rather than assumed.
    2 dp resolves 0.01 material points; the SMALLEST non-zero LEAF_V3 delta between two candidates
    is one point of HP, worth LEAF_V3_W * W[hp_me] / SD[hp_me] = 1000 * 0.03222 / 226.885 = 0.142
    points — 14x the trace resolution. One card of hand is 14.9 points, one energy 38.4. Two
    candidates that agree on all 8 features get an EXACTLY equal LEAF_V3 term, so the flag adds no
    sub-0.01 population for 2 dp to collapse. Raising the precision here would only widen the
    corpus for no recovered signal."""
    return 6 if WIN_VALUE else 2


def _emit_trace(obs, me, cands, scored, best, best_v, margin, rolls, stopped_early, info, t0):
    """Write one rich JSONL decision record: the info/judgements that drove THIS choice — our
    candidates with averaged values + per-world spread, the predicted opponent MIN-reply line, the
    belief deck-guess, the chosen move + margin + a short reason, time/rollouts, and caught errors.
    Fully guarded by the caller (TRACE_LEVEL≥1) so it costs nothing when tracing is off."""
    try:
        _TRACE_DP = _trace_dp()
        st = obs.current
        mine, opp = st.players[me], st.players[1 - me]
        ma = mine.active[0] if mine.active else None
        oa = opp.active[0] if opp.active else None
        # map cand index -> its per-world value spread (from the K-world loop), if available
        vlists = (info or {}).get("vlists") or []
        spread = {}
        for ci, cand in enumerate(cands):
            if ci < len(vlists) and vlists[ci]:
                vs = vlists[ci]
                spread[cand[0]] = {"mean": round(sum(vs) / len(vs), _TRACE_DP),
                                   "min": round(min(vs), _TRACE_DP), "max": round(max(vs), _TRACE_DP),
                                   "n": len(vs)}
        cand_records = []
        sv_by_cand = {tuple(c): s for s, c in scored}
        for cand in cands:
            rec = _describe_opt(obs, cand)
            rec["value"] = round(sv_by_cand.get(tuple(cand), float("nan")), _TRACE_DP) \
                if tuple(cand) in sv_by_cand else None
            if cand[0] in spread:
                rec["world_spread"] = spread[cand[0]]
            cand_records.append(rec)
        # the opponent reply line that drove the chosen value (MIN node), rendered if we can
        reply = (info or {}).get("reply_line")
        opp_reply = None
        if reply:
            try:
                opp_reply = [_describe_opt_at(obs, r) for r in reply]
            except Exception:
                opp_reply = reply
        chosen = _describe_opt(obs, best)
        # short reason: why this move won
        if margin is None:
            reason = "only candidate"
        elif margin >= _scaled_margin(1000.0):
            reason = "prize-race swing vs alternatives"
        elif margin >= _scaled_margin(100.0):
            reason = "clear board-value edge"
        else:
            reason = "narrow edge (near-tie)"
        if stopped_early:
            reason += "; early-stopped (converged)"
        trace.TRACER.log({
            "ctx": "MAIN", "turn": _turn_count(st), "stage": _game_stage(mine, opp),
            "prizes": [len(mine.prize), len(opp.prize)],
            "my_active": getattr(H.CARDS.get(getattr(ma, "id", None)), "name", None) if ma else None,
            "my_active_hp": getattr(ma, "hp", None),
            "opp_active": getattr(H.CARDS.get(getattr(oa, "id", None)), "name", None) if oa else None,
            "opp_active_hp": getattr(oa, "hp", None),
            "opp_belief": _belief_guess(obs),
            "recurse_depth": RECURSE_DEPTH, "k_worlds": K_WORLDS,
            # WHICH SCALE these values are on. Without it a reader cannot tell a margin of 0.03
            # (a decisive probability gap) from 0.03 material points (a rounding artifact), and the
            # downstream threshold tools silently report "(none)" forever — see analyze_trace.py.
            "win_value": bool(WIN_VALUE),
            # LEAF_V3 AUDIT (site 5 of 5): the ONE site that needs a real change. `win_value` names
            # the SCALE; this names WHICH LEAF produced the numbers, which a reader cannot otherwise
            # recover — a 0.14 margin means "one HP of resource edge" under LEAF_V3 and "a rounding
            # artifact" without it. Emitted only when armed, so a flag-off trace is byte-identical
            # to the incumbent's and no downstream trace reader sees a new key it must ignore.
            **({"leaf_v3": True} if LEAF_V3 else {}),
            # Same contract as leaf_v3 above: names WHICH leaf produced the numbers, emitted only
            # when armed so a flag-off trace stays byte-identical to the incumbent's.
            **({"leaf_dyn": True} if LEAF_DYN else {}),
            "n_options": len(obs.select.option), "n_cands": len(cands),
            "candidates": cand_records,
            "opp_min_reply": opp_reply,          # the "potential opponent play" that drove the value
            "chosen": chosen, "best_v": round(best_v, _TRACE_DP),
            "margin": round(margin, _TRACE_DP) if margin is not None else None,
            "reason": reason,
            "rollouts": rolls, "early_stop": stopped_early,
            "dom_errors": _DOM_ERR[0],           # caught rollout/minimax failures so far (warnings)
            "time_s": round(time.time() - t0, 4),
        })
    except Exception:
        # tracing must NEVER break a decision; swallow and count it
        _DOM_ERR[0] += 1


def _describe_opt_at(obs, cand) -> dict:
    """Render an opponent reply option index against the CURRENT obs (best-effort label only —
    the option list at the MIN node differs from root, so this is a coarse 'index + type if seen')."""
    try:
        if cand and cand[0] < len(obs.select.option):
            return _describe_opt(obs, cand)
    except Exception:
        pass
    return {"idx": cand[0] if cand else None}


def _is_basic_pokemon(cd) -> bool:
    """A Basic Pokemon = a Pokemon card with no predecessor (engine `evolvesFrom` empty)."""
    try:
        return (cd is not None and cd.cardType == H.CardType.POKEMON
                and not getattr(cd, "evolvesFrom", None))
    except Exception:
        return False


def _benchable_basic_option(obs):
    """Index-list [i] of the first MAIN option that plays a Basic Pokemon from hand to the bench, else
    None. A PLAY option is {type: PLAY, index: <hand index>} — the card is hand[index] (NOT resolvable via
    _option_card, which reads area/inPlayArea that PLAY options omit)."""
    sel = obs.select
    if sel is None or int(sel.context) != int(SelectContext.MAIN) or sel.maxCount != 1:
        return None
    st = obs.current
    me = st.players[st.yourIndex]
    hand = me.hand or []
    for i, o in enumerate(sel.option):
        if o.type != OptionType.PLAY:
            continue
        hi = getattr(o, "index", None)
        if hi is None or not (0 <= hi < len(hand)):
            continue
        if _is_basic_pokemon(H.CARDS.get(getattr(hand[hi], "id", None))):
            return [i]
    return None


def _benchless_is_safe(obs):
    """The user's soundness rule: staying benchless is ALLOWED only if (a) we have a killing blow this
    turn (we win/KO now, so the lone-active cliff never materialises), or (b) we are highly certain the
    opponent cannot KO our active next turn (their FULL realistic fueled burst + archetype booster, via
    _envelope_can_ko, falls short of our active HP). Otherwise benchless is the instant-loss cliff
    (engine loss reason 3) and we MUST develop a backup. Conservative: any plausible KO threat => unsafe."""
    if _lethal_move(obs) is not None:                       # killing blow available -> win now
        return True
    st = obs.current
    me = st.players[st.yourIndex]
    opp = st.players[1 - st.yourIndex]
    arch = (_OPP_ARCH[0] or (None, 0.0))[0]
    return not _envelope_can_ko(me, opp, arch)              # opp can't KO our active -> safe to stay benchless


def _force_bench_dev(obs):
    """Anti-brick development rule (see BENCH_FORCE). At OUR MAIN single-select, if our bench holds fewer
    than BENCH_FORCE_MIN Pokemon AND a legal option plays a Basic Pokemon from hand to the bench AND
    staying benchless is NOT safe (per _benchless_is_safe — no killing blow + opponent can KO our lone
    active), return that bench option (a free action; the search resumes on the next select). Else None
    (defer to the search). Opponent-independent development; never crashes (any error => defer)."""
    try:
        st = obs.current
        me = st.players[st.yourIndex]
        if sum(1 for pk in (me.bench or []) if pk) >= BENCH_FORCE_MIN:
            return None
        bench_opt = _benchable_basic_option(obs)
        if bench_opt is None:                              # no benchable basic in hand (a true deck brick)
            return None
        if _benchless_is_safe(obs):                        # killing blow OR opp can't KO active -> allowed
            return None
        return bench_opt                                   # benchless + threatened + no win now -> MUST bench
    except Exception:
        return None


def _dominance_enabling(obs, opt):
    """DOMINANCE_ENABLING_ONLY (see the flag block): True iff this ATTACH option COMPLETES an attack
    cost on its own target — after the attach the target can pay an attack it cannot pay NOW
    (need - have == 1). That is the narrow, capability-buying subset of the free-attach class; the
    broad "always spend the expiring attach" rule overshot the teacher's 22% ATTACH share to 41%.

    NO cost logic is re-implemented here (the explicit design constraint): `need` is read from
    `_consensus_attack_cost` — the very detector POLICY_PRIOR_PRUNE's `attach_to_attacker` key uses,
    i.e. the min energy COUNT over the card's attacks — and from `H._strongest_attack`'s own
    `.energies` (so a Basic pre-loading toward a costlier top attack also counts as enabling when the
    attach exactly completes it). `have` is the target's attached-energy count, counted exactly as
    that detector counts it. The OVERFILL end of the range is NOT re-checked here: it belongs to
    `H._can_pay_strongest`, which `H._score` early-returns on, so an overfill attach never carries the
    ATTACH_FIRST band and therefore cannot pass the score-diff qualification in _dominant_free_move.
    Target resolution mirrors H._score's ATTACH branch (`_resolve(...) or my_active`). Never raises;
    returns False on any doubt (the caller then defers that option to the search)."""
    try:
        if getattr(opt, "type", None) not in (OptionType.ATTACH, OptionType.ENERGY,
                                              OptionType.ENERGY_CARD):
            return False
        st = obs.current
        me = st.players[st.yourIndex]
        tgt = H._resolve(obs, getattr(opt, "inPlayArea", None),
                         getattr(opt, "inPlayIndex", None), st.yourIndex)
        if tgt is None:                                    # H._score's ATTACH default target
            tgt = me.active[0] if me.active else None
        if tgt is None:
            return False
        cd = H.CARDS.get(getattr(tgt, "id", None))
        if cd is None:
            return False
        have = len(getattr(tgt, "energies", []) or [])
        costs = []
        need = _consensus_attack_cost(cd)                  # cheapest attack — the existing detector
        if need is not None:
            costs.append(need)
        atk = H._strongest_attack(cd)                      # same cost source, the card's top attack
        if atk is not None and getattr(atk, "energies", None) is not None:
            costs.append(len(atk.energies))
        return any(c > 0 and (c - have) == 1 for c in costs)
    except Exception:
        return False


def _dominant_free_move(obs):
    """DOMINANCE_FORCE (see the flag block): return [idx] of a provably-dominant FREE action at OUR
    MAIN single-select, else None (defer to the search).

    Today the proven class is the energy attach: free, once-per-turn, EXPIRING, and it does not end
    the turn, so a BENEFICIAL attach cannot cost us the alternative — it only defers it by one
    decision. Qualification reuses ATTACH_FIRST's vetted exclusions with NO duplicated rule logic, by
    diffing H._score with the flag flipped: a option qualifies iff flipping ATTACH_FIRST raises its
    score by exactly ATTACH_FIRST_BAND, which by construction happens only for a non-overfill,
    non-doomed-sink attach. Among qualifiers we keep the H._score argmax so the bench-vs-active /
    PLAN / TEMPO shaping and the live attach_to_attacker prune weight still pick the target.
    Under DOMINANCE_ENABLING_ONLY (default True) a candidate must ALSO complete an attack cost
    (_dominance_enabling) — the broad class overshot the teacher's ATTACH share; see the flag block.
    Capped at one force per turn (_DOM_FORCED) so the agent can never loop. Never raises."""
    try:
        sel = obs.select
        if sel is None or int(sel.context) != int(SelectContext.MAIN) or sel.maxCount != 1:
            return None
        opts = sel.option
        if not opts or len(opts) < 2:
            return None
        turn = _turn_count(obs.current)
        prev = _DOM_FORCED[0]
        if prev is not None and prev[0] == turn and prev[1] >= 1:
            return None                                    # already spent this turn's free attach
        band = float(getattr(H, "ATTACH_FIRST_BAND", 12000.0))
        was = H.ATTACH_FIRST
        best_i, best_s = None, None
        try:
            for i, o in enumerate(opts):
                if DOMINANCE_ENABLING_ONLY and not _dominance_enabling(obs, o):
                    continue                               # narrow class: must COMPLETE an attack cost
                H.ATTACH_FIRST = False
                lo = H._score(o, obs)
                H.ATTACH_FIRST = True
                hi = H._score(o, obs)
                if abs((hi - lo) - band) > 1.0:            # not a beneficial non-sink attach
                    continue
                if best_s is None or lo > best_s:          # rank qualifiers by their NORMAL score
                    best_i, best_s = i, lo
        finally:
            H.ATTACH_FIRST = was
        if best_i is None:
            return None
        _DOM_FORCED[0] = (turn, 1)
        return [best_i]
    except Exception:
        return None


def _dom_pick_wins_now(obs, opt):
    """True if this ATTACK looks GAME-WINNING right now (its KO takes the opponent's last prize(s)).
    DOMINANCE_CLOSE never defers such a pick — 'never risk a win' outranks closing a wasted resource,
    even though the deferral is provably safe in principle. Conservative: any doubt => True (do not
    override). Cheap: one damage estimate + one prize count, no search."""
    try:
        if int(getattr(opt, "type", -1) or -1) != int(OptionType.ATTACK):
            return False
        st = obs.current
        opp = st.players[1 - st.yourIndex]
        left = len(opp.prize or [])
        oact = (opp.active or [None])[0]
        if oact is None or left <= 0:
            return True                                    # unclear board => do not override
        dmg = _ce_attack_damage(st.players[st.yourIndex].active[0], oact,
                                getattr(opp, "handCount", 0) or 0,
                                getattr(st.players[st.yourIndex], "handCount", 0) or 0)
        if dmg < (getattr(oact, "hp", 0) or 0):
            return False                                   # not even a KO => safe to defer
        return H._prizes_for_ko(oact) >= left              # this KO takes their last prize(s) => WIN
    except Exception:
        return True                                        # fail SAFE: never override on uncertainty


def _dom_close_classes(obs):
    """Option indices of TEACHER-MANDATORY classes still available at this MAIN decision (see the
    DOMINANCE_CLOSE flag block for the compliance table each entry is justified by). Returned in
    priority order = descending measured blunders/game, so the scarcest waste is closed first.

    Every predicate is a structural read of the 100% KNOWNS on our own side (our board, our hand, our
    energy, the once-per-turn flags) — no probability and no opponent model. Cost/payability questions
    reuse H's own helpers (_can_pay_strongest / _strongest_attack / _skill_value) rather than
    re-implementing rules. Never raises (any error => that class simply does not qualify)."""
    out = []
    try:
        st = obs.current
        me = st.players[st.yourIndex]
        hand = me.hand or []
        opts = obs.select.option
    except Exception:
        return out

    def _energy_completes(o, active_only):
        """ATTACH of an ENERGY from hand that makes an attack PAYABLE that is not payable now.
        Colour-aware via H._can_pay_strongest on the projected target (NOT a raw count test — the
        count form measured only 69-70% teacher compliance vs 100% for the payability form)."""
        try:
            if int(getattr(o, "type", -1) or -1) not in _ATTACH_TYPES:
                return False
            if active_only and getattr(o, "inPlayArea", None) != H.AreaType.ACTIVE:
                return False
            tgt = H._resolve(obs, getattr(o, "inPlayArea", None), getattr(o, "inPlayIndex", None),
                             st.yourIndex)
            if tgt is None or H._can_pay_strongest(tgt):       # overfill: nothing to complete
                return False
            cd = H.CARDS.get(getattr(tgt, "id", None))
            atk = H._strongest_attack(cd) if cd is not None else None
            if atk is None or (getattr(atk, "damage", 0) or 0) <= 0:
                return False
            need = len(getattr(atk, "energies", []) or [])
            have = len(getattr(tgt, "energies", []) or [])
            return need > 0 and (need - have) == 1            # this attach completes the cost
        except Exception:
            return False

    for j, o in enumerate(opts):
        try:
            t = int(getattr(o, "type", -1) or -1)
        except Exception:
            continue
        # 1. ATTACH that completes a cost (active first — teacher 100%, then any target — 96.9%)
        if _energy_completes(o, True):
            out.append((0, j)); continue
        if _energy_completes(o, False):
            out.append((1, j)); continue
        # 2. FREE SUPPORTER — once per turn, expiring (teacher 97.3%, ours 80.0%, 1.00 blunders/game)
        # DOM_CLOSE_DECK_GUARD: but NOT when deck-out is the binding constraint — see the flag block.
        if (t == OptionType.PLAY and not bool(getattr(st, "supporterPlayed", False))
                and not (DOM_CLOSE_DECK_GUARD
                         and 0 <= int(getattr(me, "deckCount", 99) or 0) <= DOM_CLOSE_MIN_DECK)):
            try:
                hi = getattr(o, "index", None)
                cd = H.CARDS.get(getattr(hand[hi], "id", None)) if hi is not None and 0 <= hi < len(hand) else None
                if cd is not None and getattr(cd, "cardType", None) == H.CardType.SUPPORTER:
                    out.append((2, j)); continue
            except Exception:
                pass
        # 3. ABILITY with real value (teacher 98.0%, ours 72.2%) — reuse the shipped skill scorer
        if t == OptionType.ABILITY:
            try:
                cd = H._option_card(obs, o)
                if cd is not None and H._skill_value(cd) >= HIGH_VALUE_ABILITY_MIN:
                    out.append((3, j)); continue
            except Exception:
                pass
        # 4. EVOLVE (teacher 100%, ours 97.0% — small but free)
        if t == OptionType.EVOLVE:
            out.append((4, j)); continue
    out.sort()
    return [j for _p, j in out]


def agent(obs_dict) -> list:
    global OPP_MODEL                                        # RISK_SLOW_TAU may set it per-decision (below)
    H.DIAG["calls"] = H.DIAG.get("calls", 0) + 1
    if not isinstance(obs_dict, dict) or obs_dict.get("select") is None:
        return MY_DECK
    try:
        obs = to_observation_class(obs_dict)
        _PP_LEAF_ARMED[0] = bool(POLICY_PRIOR) and any(POLICY_PRIOR_LEAF.values())
        if POLICY_PRIOR:                                 # Layer A: snapshot root prize counts for the
            try:                                         # leaf prize-equal gate (set ONCE per decision)
                _rst = obs.current
                _rme = _rst.players[_rst.yourIndex]
                _rop = _rst.players[1 - _rst.yourIndex]
                _POLICY_PRIOR_ROOT_PRIZES[0] = (len(_rme.prize), len(_rop.prize))
            except Exception:
                _POLICY_PRIOR_ROOT_PRIZES[0] = None
        else:
            _POLICY_PRIOR_ROOT_PRIZES[0] = None
        if DOMINANCE_FORCE:                              # strictly-dominant FREE action (see flag block):
            dm = _dominant_free_move(obs)                # the attach is free/expiring and never ends the
            if dm is not None:                           # turn, so play it — the energy-blind leaf cannot
                H.DIAG["dominance_force"] = H.DIAG.get("dominance_force", 0) + 1
                return dm                                # see its value, and the KO is only deferred
        if BENCH_FORCE:                                  # anti-brick: develop a backup basic off the cliff
            bm = _force_bench_dev(obs)                   # (hard pre-search rule; benching is a free action)
            if bm is not None:
                H.DIAG["bench_force"] = H.DIAG.get("bench_force", 0) + 1
                return bm
        if G1_CLOSE:                                     # G1 slice 2: match-point close enumerator —
            try:                                         # a found multi-step close is a FACT; play it
                cm = _g1_close_line(obs, None)
                if cm is not None:
                    H.DIAG["g1_close"] = H.DIAG.get("g1_close", 0) + 1
                    return cm
            except Exception:
                pass
        if getattr(H, "OPPONENT_ADAPT", False):         # enh 3 (flag-off): classifier -> per-archetype
            try:                                        # POLICY_W shift once the MAP is confident (t>=3)
                H.apply_opponent_shift(obs, _turn_count(obs.current))
            except Exception:
                pass
        if LEARNED_COND and belief is not None and getattr(H, "LEARNED_POLICY", False):
            try:                                        # matchup-conditional blend, set ONCE at the root
                import learned_pilot as _LPc
                _lab = belief.map_archetype(obs)
                _LPc.BLEND = _LPc.blend_for_arch(_lab[0] if isinstance(_lab, tuple) else _lab)
            except Exception:
                pass
        if (PRIZE_RACE or THREAT_ENVELOPE or ROLE_AWARE or SATURATE_VALUE or WINCON_LEAF
                or DENY_EVOLUTION or RACE_CLOCK or RISK_SLOW_TAU > 0.0
                or ROLE_MATCHUP or MATCHUP_PRUNE               # S1/S2 read the root _OPP_ARCH cache
                or REPLY_TENDENCY or TRAP_TIEBREAK             # mined-tendency levers read it too
                or SUMO_EXPOSURE or GUST_SNIPE_PRIORITY        # anti-Hariyama family gate reads it
                or (CLOSING_EXCHANGE and CE_SLOW_ONLY)) and belief is not None:   # cache opp archetype
            # RACE_CLOCK added 2026-07-05 (review of 55b3f19): it was MISSING here, so _OPP_ARCH
            # stayed None for the v2pack N=600 confirm -> the priors/archetype half of the race
            # clock NEVER RAN (both "pace variant" arms were the same archetype-blind path).
            # The rejection stands AS-CONFIGURED; any RC-C recalibration is only valid with this
            # gate fixed. Behavior-neutral when RACE_CLOCK=False.
            try:                                        # ONCE from the REAL root state (our unveiled
                _OPP_ARCH[0] = belief.map_archetype(obs)  # knowledge): race-clock, threat-envelope, ROLE
            except Exception:
                _OPP_ARCH[0] = None
            # slow verdict cached per ROOT (the per-leaf compute confound, 07-06): leaves read the bool
            _CE_ROOT_SLOW[0] = (_ce_opp_is_slow() if (CLOSING_EXCHANGE and CE_SLOW_ONLY) else False)
        # REPLY_TENDENCY / TRAP_TIEBREAK shared root cache: resolve the mined tendencies entry
        # ONCE per decision (reset EVERY decision — the RISK_SLOW_TAU no-leak convention).
        # Both flags off => always None (byte-identical).
        _RT_ROOT[0] = (_rt_resolve_root()
                       if ((REPLY_TENDENCY or TRAP_TIEBREAK) and belief is not None) else None)
        if _RT_ROOT[0] is not None:
            DIAG["reply_tendency_root"] = DIAG.get("reply_tendency_root", 0) + 1
        # SUMO_EXPOSURE / GUST_SNIPE_PRIORITY per-decision arming (reset EVERY decision — the
        # RISK_SLOW_TAU no-leak convention; flag-off both cells stay False/None = byte-identical).
        # GUST_SNIPE arms the H-side cell seat-scoped: only OUR gust-target valuation is skewed;
        # the MIN node's simulated opponent keeps the untouched exact-arithmetic bands.
        _SUMO_ROOT[0] = (SUMO_EXPOSURE and _hariyama_family_gate())
        if _SUMO_ROOT[0]:
            DIAG["sumo_armed"] = DIAG.get("sumo_armed", 0) + 1
        H._GS_ARMED[0] = None
        if GUST_SNIPE_PRIORITY and _hariyama_family_gate():
            _seat = obs.current.yourIndex if obs.current is not None else None
            if _seat is not None:
                H._GS_ARMED[0] = (_seat, GUST_SNIPE_PREEVO_W, GUST_SNIPE_ENGINE_W)
                DIAG["gust_snipe_armed"] = DIAG.get("gust_snipe_armed", 0) + 1
        if RISK_SLOW_TAU > 0.0:                           # SELECTIVE soft-min (risk posture): relax the
            # worst-case MIN toward a likelihood-weighted opponent ONLY vs a confidently-slow, NON-
            # fallback archetype (opp prizes/turn <= cutoff); hard-MIN vs all aggro. Reset EVERY decision
            # so a prior turn's softmax can never leak. Construction guard: H.DIAG['risk_softmin'] MUST
            # stay 0 on every aggro matchup (the falsifier that would have caught RACE_CLOCK).
            _ra = _OPP_ARCH[0] if belief is not None else None
            if (isinstance(_ra, tuple) and len(_ra) >= 3 and not _ra[2]
                    and _opp_prize_rate(_ra[0]) <= RISK_SLOW_CUTOFF):
                OPP_MODEL = {"mode": "softmax", "tau": RISK_SLOW_TAU}
                H.DIAG["risk_softmin"] = H.DIAG.get("risk_softmin", 0) + 1
            else:
                OPP_MODEL = None
        if DENY_EVOLUTION and belief is not None:        # cache the belief-consistent card-id union ONCE
            try:                                         # per root decision (avoid re-posterior in leaf):
                _OPP_CONSIST_IDS[0] = set().union(       # union of card-ids over consistent posterior particles
                    *[(d.get("_set") or set(d.get("deck") or [])) for d, _w in (belief.posterior(obs) or [])]
                ) or None
            except Exception:
                _OPP_CONSIST_IDS[0] = None
        else:
            _OPP_CONSIST_IDS[0] = None
        if ROLE_AWARE and matchup_role is not None:     # assign the matchup ROLE once from the real root:
            try:                                        # OUR deck-speed vs the OPPONENT's (belief), gated
                arch = _OPP_ARCH[0] or (None, 0.0)      # by VEIL confidence (the MAP posterior weight).
                # require a real belief READ, not a prior-only fallback guess (mirror RISK_SLOW_TAU
                # + _deny_evolution_terms); a fallback MAP is just the modal prior (now >CONF_MIN
                # for the current-meta library), so leave the role neutral on fallback.
                if isinstance(arch, tuple) and len(arch) >= 3 and not arch[2]:
                    opp_speed = _opp_prize_rate(arch[0])    # opponent prizes/turn from their archetype label
                    conf = arch[1] if len(arch) > 1 else 0.0
                    _OPP_ROLE[0] = matchup_role.assign_role(_compute_our_speed(), opp_speed, conf)
                else:
                    _OPP_ROLE[0] = (0, 0.0)
            except Exception:
                _OPP_ROLE[0] = (0, 0.0)
            _OPP_AMP_LIVE[0] = True                      # COUNTER DEPLETION: drop the booster threat once
            try:                                        # we've seen all copies of it (they can't boost now)
                lbl = (_OPP_ARCH[0] or (None, 0))[0]
                amp = _ARCH_AMP.get(lbl) if lbl else None
                if amp and amp.get("cid"):
                    _OPP_AMP_LIVE[0] = not belief.counter_depleted(obs, amp["cid"], amp.get("copies", 4))
            except Exception:
                _OPP_AMP_LIVE[0] = True
        if ROLE_MATCHUP and matchup_role is not None:   # S1: archetype-CONDITIONAL role activation —
            # CONTROL posture ONLY when the mined matchup_priors entry for the belief-MAP archetype
            # says wincon=="stall" (data-driven; see the ROLE_MATCHUP flag comment). Reset EVERY
            # decision (the RISK_SLOW_TAU convention) so a stale role never leaks; deliberately
            # AFTER (and overriding) ROLE_AWARE's speed-derived assignment when both are on.
            _OPP_ROLE[0] = _role_matchup_assign()
        cands = _candidates(obs)
        _sr0 = None                                     # SIB_RANKER: incumbent front (flag-off: stays None)
        if SIB_RANKER and cands is not None and len(cands) > 1:
            _sr0 = cands[0][0]                          # remembered for the heuristic-path adoption below
            cands = _sib_rerank(obs, cands)             # top-3 permutation only (see the flag block)
        budget = obs_dict.get("remainingOverageTime", 0) or 0
        if cands is None or not obs_dict.get("search_begin_input") or budget < MIN_BUDGET_S:
            DIAG["heur_path"] = DIAG.get("heur_path", 0) + 1
            if _sr0 is None:
                return H._choose(obs)                   # heuristic path (flag-off: the untouched incumbent)
            hc = H._choose(obs)
            # SIB_RANKER heuristic-path adoption: ONLY when _choose picked the incumbent's own #1
            # (all its s>0/minCount guards intact) swap in the ranker's preferred member of the
            # SAME top-3. Any other _choose outcome stands untouched — never promote outside the
            # incumbent's top-3, never override a guard.
            if hc and len(hc) == 1 and hc[0] == _sr0 and cands[0][0] != _sr0:
                DIAG["sib_ranker_heur_swap"] = DIAG.get("sib_ranker_heur_swap", 0) + 1
                return [cands[0][0]]
            return hc                                   # heuristic path
        me = obs.current.yourIndex
        if LETHAL_SHORTCUT:                             # enh 2 (flag-off): take a winning KO now,
            lm = _lethal_move(obs)                      # skip search (also the all-sims-equal guard)
            if lm is not None:
                DIAG["search_path"] = DIAG.get("search_path", 0) + 1
                return lm
        preds = _predict(obs)
        t0 = time.time()
        if TREE_SEARCH:
            # STRUCTURAL: pick the first move leading to the best achievable OUR-turn LINE.
            deadline = t0 + PER_MOVE_S
            tbest_v, tbest = None, None
            for cand in cands:
                if time.time() > deadline:
                    break
                v = _tree_best(obs, preds, me, cand, deadline)
                if v is not None and (tbest_v is None or v > tbest_v):
                    tbest_v, tbest = v, cand
            DIAG["search_path"] = DIAG.get("search_path", 0) + 1
            return tbest if tbest is not None else H._choose(obs)
        scored = []                       # (value, cand) in evaluation order
        rolls = 0
        stopped_early = False
        info = None
        if TIME_AWARE:
            # adaptive deadline + round-robin sampling-to-budget (spends the overage pool);
            # with RECURSE_DEPTH>0 the per-candidate leaf is a K-world belief minimax (THE DOMINION)
            scored, rolls, stopped_early, info = _search_timeaware(obs, cands, preds, me, t0, budget)
        else:
            for cand in cands:
                if time.time() - t0 > PER_MOVE_S:
                    break
                vals = []
                for k in range(DETERMINIZATIONS):
                    try:
                        preds_k = _predict(obs, vary=(k > 0), seed=k) if DETERMINIZATIONS > 1 else preds
                        ss = api.search_begin(obs, *preds_k, False)
                        ss = api.search_step(ss.searchId, cand)
                        vals.append(_rollout(ss, me))
                        api.search_end()
                        rolls += 1
                    except Exception:
                        try:
                            api.search_end()
                        except Exception:
                            pass
                        continue
                if not vals:
                    continue
                scored.append((sum(vals) / len(vals), cand))  # avg over sampled opp hands (belief)
                # convergence early-stop: if the current leader dominates the rest by a clear
                # margin, stop and bank the remaining budget for a more contested turn.
                if EARLY_STOP_MARGIN is not None and len(scored) >= 2:
                    vs = sorted((s for s, _ in scored), reverse=True)
                    if vs[0] - vs[1] > _scaled_margin(EARLY_STOP_MARGIN):
                        stopped_early = True
                        break
        DIAG["search_path"] = DIAG.get("search_path", 0) + 1
        DIAG["rollouts"] = DIAG.get("rollouts", 0) + rolls
        if not scored:
            return H._choose(obs)
        scored.sort(key=lambda sv: -sv[0])
        best_v, best = scored[0]
        if ROBUST_TIEBREAK and len(scored) > 1 and info and info.get("vlists"):
            # among candidates within ROBUST_BAND of the best MEAN, take the one whose WORST belief
            # world is least bad (maximin) — the robust reconfiguration path that survives the
            # opponent's best/luckiest reply. The deckout/benchless leaf makes a losing world score low,
            # so this actively drops self-losing lines. Needs a world spread (BELIEF + K_WORLDS>1).
            def _worst(cand):
                try:
                    vl = info["vlists"][cands.index(cand)]
                except (ValueError, IndexError):
                    return float("-inf")
                return min(vl) if vl else float("-inf")
            band = [sv for sv in scored if best_v - sv[0] <= _scaled_margin(ROBUST_BAND)]
            if len(band) > 1:
                band.sort(key=lambda sv: (_worst(sv[1]), sv[0]), reverse=True)
                best_v, best = band[0]
        if TRAP_TIEBREAK and len(scored) > 1 and info and info.get("trap_exp"):
            # TRAP_TIEBREAK (MAX-node counterpart of REPLY_TENDENCY — see its flag block):
            # re-rank ONLY the within-eps safety band by the mined tendency-weighted
            # expectation. Applied LAST (after ROBUST_TIEBREAK) — the finest tie-break; the
            # maximin primary key is enforced inside _trap_apply (a move outside the band can
            # never be returned). DIAG: armed vs band vs ACTUALLY-changed.
            DIAG["trap_armed"] = DIAG.get("trap_armed", 0) + 1
            # ONE conversion, used by both the counter and the decision. Converting only the
            # counter would leave _trap_apply comparing a probability gap against 150.0 -- always
            # true, so its band becomes EVERY candidate and the maximin invariant above is void
            # while the DIAG counter still reports a healthy band.
            _trap_eps = _scaled_margin(TRAP_TIEBREAK_EPS)
            if sum(1 for sv in scored if scored[0][0] - sv[0] <= _trap_eps) > 1:
                DIAG["trap_band"] = DIAG.get("trap_band", 0) + 1
            _tv, _tb, _chg = _trap_apply(scored, cands, info["trap_exp"],
                                         _trap_eps, best_v, best)
            if _chg:
                DIAG["trap_changed"] = DIAG.get("trap_changed", 0) + 1
                best_v, best = _tv, _tb
        margin = (best_v - scored[1][0]) if len(scored) > 1 else None
        # arena/dtrace hook: surface the chosen candidate VALUE + margin for per-decision tracing
        # (negligible: two dict writes; does NOT alter the decision). Read by arena/arena.py.
        DIAG["last_value"] = best_v
        DIAG["last_margin"] = margin
        if TRACE_LEVEL >= 1 and trace.TRACER.enabled:
            _emit_trace(obs, me, cands, scored, best, best_v, margin, rolls, stopped_early,
                        info, t0)
        if DOMINANCE_CLOSE:
            # DOMINANCE_CLOSE (see the flag block): the search has settled on ENDING the turn. If a
            # teacher-mandatory class is still available, playing it now is strictly better than
            # wasting it — the leaf simply cannot see its value (energy/abilities score 0.0 with
            # RICH_EVAL off). Fires ONLY on the END pick, so it never pre-empts a searched attack/KO
            # and never reorders plays within a turn.
            try:
                _bi = best[0] if isinstance(best, list) and best else None
                _opts = obs.select.option
                # THE TURN-ENDING PICK, not just END: in this game ATTACKING ends the turn, and our
                # ATTACK compliance is already 97-99% — so the free/expiring resource is almost always
                # wasted on an ATTACK pick, not an END pick. Measured: END is only 2.7% of our MAIN
                # picks (an END-only hook fired 0 times in 3 real games), while the original T3
                # diagnosis found 132/141 of the wasted-attach turns ended on a LETHAL. Deferring is
                # safe: attach/ability/supporter/evolve do NOT end the turn, so the attack is still
                # offered on the very next decision (attach-then-attack observed 116x/40 games).
                if (_bi is not None and 0 <= _bi < len(_opts)
                        and int(getattr(_opts[_bi], "type", -1) or -1)
                            in (int(OptionType.END), int(OptionType.ATTACK))
                        and not _dom_pick_wins_now(obs, _opts[_bi])):
                    _t = _turn_count(obs.current)
                    _prev = _DOM_CLOSED[0]
                    _n = _prev[1] if (_prev is not None and _prev[0] == _t) else 0
                    if _n < DOMINANCE_CLOSE_CAP:
                        _cls = _dom_close_classes(obs)
                        if _cls:
                            _DOM_CLOSED[0] = (_t, _n + 1)
                            DIAG["dominance_close"] = DIAG.get("dominance_close", 0) + 1
                            return [_cls[0]]
            except Exception:
                pass
        return best
    except Exception as _fb_exc:
        global FALLBACK_EXC, LAST_FALLBACK_EXC, LAST_FALLBACK_TB, FALLBACK_BUG
        FALLBACK_EXC += 1
        LAST_FALLBACK_EXC = repr(_fb_exc)
        DIAG["fallback_exc"] = DIAG.get("fallback_exc", 0) + 1
        try:                                   # WHERE, not just what — a repr() alone never told
            import traceback                   # anyone which call site had rotted
            _tb = traceback.extract_tb(_fb_exc.__traceback__)
            LAST_FALLBACK_TB = " <- ".join("%s:%d %s" % (os.path.basename(f.filename), f.lineno,
                                                         f.name) for f in _tb[-2:])
        except Exception:
            LAST_FALLBACK_TB = None
        if isinstance(_fb_exc, _BUG_EXC):      # our bug, not the engine's: surface it loudly
            FALLBACK_BUG += 1
            DIAG["fallback_bug"] = DIAG.get("fallback_bug", 0) + 1
            DIAG["fallback_bug_last"] = "%s @ %s" % (type(_fb_exc).__name__, LAST_FALLBACK_TB)
        return H._legal_fallback(obs_dict)
