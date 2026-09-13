# Agent modules, as submitted on freeze day

These two files are included as **reference, not as a runnable package.** They are the artefact the
writeup describes and audits; they will not execute standing alone.

`search_agent.py` — determinized search: sample the hidden information, search each sampled world a
few plies, vote across worlds. 5,099 lines, of which 2,462 are code and 2,283 are comment and
docstring: roughly 45 % of the file is a written record of what was tried and what it measured.
128 functions, 59 module-level boolean switches (1 on at freeze).

`heuristic_agent.py` — the action-selection layer used at the search leaf and in rollouts.
2,630 lines, 1,345 of them code. 76 functions, 30 switches (3 on at freeze).

## Why it will not run here

Both modules import the competition harness (`from cg import api`) and about a dozen sibling modules
from the original tree that are deliberately **not** published: `belief`, `features`, `value_leaf`,
`learned_pilot`, `archetype_clf`, `matchup_role`, `rank_prior`, `self_model`, `leaf_fe`,
`core.damage`. Several of those carry fitted model weights or mined metagame data; the rest are
scaffolding with no bearing on the writeup's claims. Read these two files to check what the writeup
says about the approach and the switches; do not expect `python search_agent.py` to do anything.

## Known structural problems

Stated plainly, since the writeup audits the agent's behaviour and it would be odd to hide the
code's own defects:

- `heuristic_agent._score` is **434 lines**. It is the leaf evaluation and it should be a dispatch
  table over per-activity scorers.
- `search_agent.agent` is 307 lines and `search_agent._state_value_material` 268. Same problem.
- 89 module-level boolean switches is a configuration surface far too wide to test. The writeup's
  closing section is about precisely this: 85 of them shipped off because no experiment could
  resolve them against ladder noise of 65 rating points.
- Outside those three functions the mean function length is about 19 lines, so the issue is
  concentrated rather than pervasive.

## Sanitisation

Two kinds of edit were made, both confined to comments and docstrings. Absolute paths from the
original working tree and the compute host's name were replaced with `<workdir>` and
`<compute-host>` placeholders in four provenance comments. Three comments named individual competing
entrants whose replays were used as references; those names were replaced with role descriptions
("a top-ranked entrant", "one high-volume entrant"). No executable line was changed — the logic is
byte-for-byte what was submitted.
