# SCOuT Search Algorithms

## Purpose

This document explains the search algorithms that are actually implemented in SCOuT today. It is intentionally code-oriented: each section describes both the optimization idea and the way that idea is realized in this repository.

The implemented strategies live in:

- [src/explore.py](/home/hari/git/SCOuT/src/explore.py)
- [src/searchMethods/wavefront_flags.py](/home/hari/git/SCOuT/src/searchMethods/wavefront_flags.py)
- [src/searchMethods/tabu_flags.py](/home/hari/git/SCOuT/src/searchMethods/tabu_flags.py)
- [src/searchMethods/beam_tabu.py](/home/hari/git/SCOuT/src/searchMethods/beam_tabu.py)
- [src/searchMethods/anneal_flags.py](/home/hari/git/SCOuT/src/searchMethods/anneal_flags.py)

## Common Execution Model

All implemented search methods use the same outer workflow:

1. Construct a candidate compiler/runtime configuration.
2. Build a binary from that candidate.
3. Execute the binary under a chosen measurement backend.
4. Extract one or more metrics.
5. Score the candidate against an objective.
6. Decide what candidate to try next.

The differences between algorithms are mostly about step 1 and step 6: how candidates are generated, and how the search decides where to move next.

Across the codebase, the candidate space is made from these building blocks:

- `compiler_flags_base`: flags that are always present
- `compiler_flags`: selectable flag variants
- `compiler_params`: tunable flag parameters like `-march=native`
- `compiler_flag_pool`: optional on/off flags
- `env`: runtime environment variables, sometimes with `when` predicates

## How SCOuT Chooses A Search Method

Search selection is done in [_run_from_config()` in main.py](/home/hari/git/SCOuT/main.py:74). The value comes from `cfg.search.study`.

Explicitly dispatched values:

- `wavefront`
- `tabu`
- `beam_tabu`
- `anneal`

Anything else falls back to Optuna.

That means:

- `optuna` is the default practical path
- unknown or stale study names silently become Optuna runs

## 1. Optuna Search

Code:

- [explore_optuna()` in src/explore.py](/home/hari/git/SCOuT/src/explore.py:38)

### What it is

Optuna is the most general search strategy in this repository. Instead of hand-coding one specific search path, SCOuT defines a search space and lets an Optuna sampler propose candidate configurations.

This is the only implementation here that fully supports multi-objective optimization end-to-end.

### How the candidate is represented

For each trial, SCOuT samples:

- one variant from `compiler_flags`
- a subset of active parametric flags from `compiler_params`
- concrete values for those active params
- a subset of optional flags from `compiler_flag_pool`
- one consistent environment assignment from `env`

The logic for this lives in:

- [suggest_compiler_flags()` in src/misc.py](/home/hari/git/SCOuT/src/misc.py:97)
- [suggest_env()` in src/misc.py](/home/hari/git/SCOuT/src/misc.py:161)

### How the search behaves

Optuna behavior depends on `search.sampler`:

- `tpe`: Tree-structured Parzen Estimator
- `nsga3`: population-based multi-objective search
- `rs`: random search
- `cmaes`: CMA-ES

For `tpe`, SCOuT also uses a staged complexity schedule in [_complexity_limits()` in src/misc.py](/home/hari/git/SCOuT/src/misc.py:24):

- early trials activate very few params and pool flags
- later trials are allowed to explore more complex combinations

That is a practical design choice: it biases early search toward simpler candidates before opening the larger combinatorial space.

### What gets optimized

Unlike the heuristic methods, Optuna uses all configured objectives in `cfg.objectives`.

That means it can optimize:

- one objective, like `CPI`
- several objectives simultaneously, producing a Pareto front

### Important implementation details

- Duplicate evaluations are cached by `(flags, env)` so identical candidates are not rebuilt.
- Failed builds prune the trial.
- Missing objective metrics prune the trial.
- Multi-objective runs can write a Pareto CSV.
- Trial metadata is stored in `trial.user_attrs`, which is later reused for CSV export.

### Best use case

Use Optuna when:

- you want the most complete implementation in the repo
- you need multi-objective optimization
- the search space includes variants, params, pool flags, and environment variables together
- you want a strong general-purpose default

## 2. Wavefront Search

Code:

- [run_wavefront_study()` in src/searchMethods/wavefront_flags.py](/home/hari/git/SCOuT/src/searchMethods/wavefront_flags.py:410)

### What it is

Wavefront search explores flag combinations by increasing combination size.

The mental model is:

- start from a baseline
- try all single added flags
- then pairs
- then triples
- stop at `max_k`

This is a structured combinational search over `flag_atoms`.

### How the candidate is represented

Wavefront treats the search space mainly as:

- `base_flags`
- `flag_atoms`

Each candidate is the baseline plus a chosen combination of atoms.

Environment settings are not ignored, but they are secondary:

- for each flag combination, SCOuT can evaluate several environment combinations
- the best environment outcome is used to rank that flag combination

### Two modes

Wavefront supports two generation modes:

- `full`: enumerate all combinations of size `k`
- `beam`: expand only the best candidates from the previous wave

The generators are implemented in:

- [_generate_full()`](/home/hari/git/SCOuT/src/searchMethods/wavefront_flags.py:228)
- [_generate_beam_boost()`](/home/hari/git/SCOuT/src/searchMethods/wavefront_flags.py:242)

### How ranking works

Wavefront only uses the first configured objective for ranking.

That first objective is turned into an internal scalar score:

- for `min`, lower is better
- for `max`, higher is better, but internally negated so lower score still wins

### Early stopping

Wavefront supports significance-based stopping.

If the best candidate in a wave does not significantly improve over the current global best, it can stop early. That logic depends on:

- `significance.min_rel_gain`
- `significance.min_abs_gain`
- `wavefront.stop_if_no_improve`

The improvement check uses [is_significant_improvement()` in src/misc.py](/home/hari/git/SCOuT/src/misc.py:211).

### Best use case

Use wavefront when:

- you want interpretable flag-combination search
- you mainly care about combinations of optional flags
- you want to see how quality changes as combination size grows
- the number of flag atoms is small enough to make layered search feasible

### Practical limitation

Wavefront is strongest when the search space is mostly about flag atoms. It is less expressive than Optuna for mixed spaces with many parametric flags and conditional environment choices.

## 3. Tabu Search

Code:

- [run_tabu_study()` in src/searchMethods/tabu_flags.py](/home/hari/git/SCOuT/src/searchMethods/tabu_flags.py:360)

### What it is

Tabu search is a local-search algorithm with memory. It walks from one candidate to nearby candidates, but keeps a tabu list so it does not keep revisiting the same recent states.

This is useful when:

- the search space is large
- full enumeration is too expensive
- greedy local improvement alone would cycle or get stuck too easily

### How the state is represented

In this codebase, the tabu state contains:

- one selected variant
- a dictionary of active parametric flags and their chosen values
- a list of enabled pool flags
- one concrete environment assignment

That is broader than wavefront: it can move through flags and environment settings in the same run.

### How neighbors are generated

Neighbor generation happens in [_neighbors()`](/home/hari/git/SCOuT/src/searchMethods/tabu_flags.py:170).

Possible move types:

- change variant
- add a param
- remove a param
- change a param value
- swap one param for another
- add a pool flag
- remove a pool flag
- switch to a different environment

Which move families are allowed is controlled by config:

- `allow_variant_moves`
- `allow_param_moves`
- `allow_pool_moves`
- `allow_env_moves`

### How the search proceeds

The loop is:

1. Start from a small initial state.
2. Evaluate a neighborhood around the current state.
3. Ignore tabu states unless they satisfy aspiration.
4. Move to the best admissible neighbor.
5. Update the tabu memory.
6. Stop when the iteration budget or no-improvement budget is exhausted.

### Aspiration

Tabu entries can be overridden if a candidate beats the current global best. That is the classic aspiration rule:

- tabu usually blocks revisits
- exceptional improvements are still allowed through

### Important implementation details

- The tabu memory stores serialized configuration keys, not abstract move operators.
- A build/measurement cache avoids reevaluating identical `(flags, env)` pairs.
- The current active implementation compares direct improvement over the best value.
- There is also commented-out significance-aware logic in the file, so the behavior here looks like a partially evolved implementation.

### Best use case

Use tabu when:

- you want local search over a richer mixed state space
- the search space is too large for structured enumeration
- you want explicit move-level control over which parts of the configuration can change

## 4. Beam-Tabu Search

Code:

- [run_beam_tabu_study()` in src/searchMethods/beam_tabu.py](/home/hari/git/SCOuT/src/searchMethods/beam_tabu.py:202)

### What it is

Beam-tabu combines two ideas:

- beam search: keep several promising frontier candidates at once
- tabu search: block recently used moves so the frontier does not keep oscillating

This makes it more global than plain tabu, because it expands a set of current good candidates rather than walking only one active state.

### How the candidate is represented

This module focuses mainly on combinations of flag atoms layered on top of `base_flags`.

The main frontier stores tuples of selected atoms. For each candidate atom-set:

- the code forms full flags as `base_flags + atom_set`
- it evaluates that flag set across environment combinations
- it ranks the candidate by its best environment outcome

### How expansion works

At each iteration:

1. Expand each current beam element into neighbors by adding or removing atoms.
2. Evaluate those neighbors.
3. Sort by objective score.
4. Keep only the top `beam_width` candidates.
5. Mark recent moves as tabu.

The move abstraction here is simpler than in the full tabu search:

- add one atom
- remove one atom

### How it differs from plain beam search

Pure beam search would always expand the best-looking frontier candidates. The tabu layer tries to reduce immediate backtracking by forbidding recently used transitions for some number of iterations.

### Important implementation caveat

This module expects `cfg.beam_tabu`, but [Config in src/config.py](/home/hari/git/SCOuT/src/config.py:211) does not define or populate a `beam_tabu` field.

That means:

- the JSON file may contain a `beam_tabu` block
- but the loaded `Config` object does not currently expose it
- so this search often runs on internal defaults instead of the configured values

This is one of the most important wiring gaps in the repository.

### Best use case

Conceptually, beam-tabu is a good fit when:

- you want flag-atom search that is broader than single-state tabu
- you want to maintain several promising candidates at once
- the search space is still atom-combination oriented

In practice, it should be treated as partially wired until the config integration is fixed.

## 5. Simulated Annealing

Code:

- [run_anneal_study()` in src/searchMethods/anneal_flags.py](/home/hari/git/SCOuT/src/searchMethods/anneal_flags.py:332)

### What it is

Simulated annealing is a stochastic local-search method that sometimes accepts worse moves on purpose. Early in the run, it explores more freely; later, as temperature cools, it becomes more conservative.

The goal is to escape local minima that a purely greedy search would get stuck in.

### How the state is represented

The state includes:

- one variant
- active parametric flags and values
- enabled pool flags

Environment is handled slightly differently here:

- the state itself does not directly carry a single environment variable choice
- instead, each proposed state is evaluated across a set of environment assignments
- the best environment result becomes that state's score

So annealing is really searching over flag states, with environment treated as an inner optimization loop.

### How neighbors are generated

Neighbor generation happens in [_neighbors_sa()`](/home/hari/git/SCOuT/src/searchMethods/anneal_flags.py:241).

Possible moves:

- change variant
- add/remove/change/swap parameter selections
- toggle pool flags

The number of active parameters respects `compiler_params_select.min` and `compiler_params_select.max`.

### Acceptance rule

Annealing compares the current candidate and the proposed candidate in score space:

- always accept if the candidate is better
- sometimes accept if it is worse, with probability based on temperature

The relevant line is the standard exponential rule:

`exp(-(new_score - current_score) / T)`

As `T` decreases, worse solutions become less likely to be accepted.

### Cooling

Temperature starts at `T0` and is multiplied by `alpha` every iteration.

Important parameters:

- `T0`
- `alpha`
- `max_iters`
- `max_no_improve`

### Global-best updates

Annealing uses significance-aware comparison before updating the recorded global best. That makes it a little more noise-tolerant than the active tabu implementation.

### Best use case

Use annealing when:

- you want a lightweight local-search strategy
- you want some ability to escape local basins
- you prefer a single active trajectory instead of a full population or frontier

## Comparing The Implemented Algorithms

## Search shape

- Optuna: model-driven sampling over the full mixed search space
- Wavefront: layered combination growth over flag atoms
- Tabu: single-state local search with explicit memory
- Beam-tabu: multi-frontier local search with move blocking
- Anneal: single-state stochastic local search with probabilistic downhill moves

## Multi-objective support

- Optuna: true multi-objective support
- Wavefront: effectively first objective only
- Tabu: effectively first objective only
- Beam-tabu: effectively first objective only
- Anneal: effectively first objective only

## Environment handling

- Optuna: samples one consistent env assignment per trial
- Wavefront: evaluates each flag combo across env combos and ranks by best env
- Tabu: environment can be part of the state and can also move
- Beam-tabu: evaluates candidates across env combos and ranks by best env
- Anneal: evaluates each flag state across env combos and ranks by best env

## Interpretability

- Wavefront is the easiest to reason about combinationally.
- Tabu and anneal are easier to think of as trajectory-based searches.
- Optuna is the most flexible, but the least hand-structured.

## Current Practical Recommendation

Based on the code as it exists now:

- use Optuna when you want the most complete and best-wired implementation
- use wavefront when you specifically want layer-by-layer flag-combination search
- use tabu when you want richer local search over flags and environment
- treat beam-tabu as experimental until `Config` is extended to carry its config block cleanly
- use annealing when you want a simpler stochastic local search with escape-from-local-minima behavior

## Known Gaps And Caveats

- `explore_synergy()` exists in [src/explore.py](/home/hari/git/SCOuT/src/explore.py:290), but there is no `synergy_flags.py` implementation in the repository.
- `search.study = "synergy"` therefore does not activate a real synergy search path from `main.py`; it falls back to Optuna.
- `beam_tabu` configuration is not fully wired into `Config`.
- Some heuristic modules use significance-aware improvement checks, while others use direct best-value comparisons or contain commented-out alternate logic.

## Related Docs

- [README.md](/home/hari/git/SCOuT/README.md)
- [docs/architecture.md](/home/hari/git/SCOuT/docs/architecture.md)
