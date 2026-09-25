# Model v2 

Neural-guided genetic algorithm for core-to-IO mapping on a mesh-based
many-core chip, with a learned, conflict-aware preemption policy.

A two-phase attention model samples mappings and preemption schedules;
those samples seed and get injected into a genetic algorithm's
population; the GA's best individuals train the model back via
REINFORCE. This is the **v2 architecture** — a full redesign from an
earlier LSTM-pointer-network version (`model_v1_backup.py`, kept for
reference, not imported anywhere). Nothing about the GA itself, the
simulator's scheduling logic, or the REINFORCE math changed; what
changed is *how the model decides mappings and preemptions*, and *why*
those decisions are structured the way they are — that's most of what
this document covers.

---

## Install

```bash
pip install -e .
```

Installs the `lmga_igsa` package and an `lmga-igsa` console script.
Python ≥3.9; PyTorch, NumPy, NetworkX pulled in automatically.

## Quickstart

```bash
# Full NN+GA co-training run
lmga-igsa --mode run --num_cores 32 --num_io 2

# Repeat N times for the mean/std numbers a results table needs
lmga-igsa --mode multirun --num_cores 32 --num_io 2 --num_runs 10
```

`--cuda` uses a GPU if available; both modes default to CPU otherwise.
There is no `--mode pretrain` in v2 (see "What v2 removed and why").

Benchmark files (`bm_<N>cores.txt`) are expected in the working
directory. If missing, a warning prints and synthetic dummy core data
is generated — fine for a smoke test, not for real numbers.

### Ablation flags

| Flag | Effect | Isolates |
|---|---|---|
| `--disable_preemption_head` | random bucket preemptions instead of the conflict-GAT head | phase 1 (mapping)'s contribution alone |
| `--disable_constructive` | no NN-sampled individuals injected into the GA | the constructive operator's contribution |
| `--disable_inloop_updates` | only the end-of-outer-iteration update fires | the in-loop update's contribution |
| `--continuous_preemptions` | revert fix #2 — continuous draws instead of bucket-snapped | whether action/reward consistency matters |

Four ablation rows, not five — v1's `--disable_warmup` had no v2
counterpart to disable (there is no warm-up phase; see below).

---

## Step-by-step: what actually happens in one outer iteration

This is the sequence `run.py` executes, in order, every iteration —
useful to read before the architecture deep-dive, since the deep-dive
explains *why* each step is shaped this way.

1. **Problem setup** (once, at the start of `run_lmga`, not per
   iteration): load the core/IO/topology instance (`data.prep_data`),
   build the per-core input tensor `[row, col, patterns, scan]`
   (`data.create_core_features_v2`), build the IO input tensor from
   each channel's `[src, sink]`.

2. **Initial population**: sample 75% of the population from the
   model — phase 1 first (mapping), then, for each sampled individual,
   build that individual's own conflict table from *its own* sampled
   mapping, then run phase 2 (preemption) using that table. The
   remaining 25% are fully random individuals (random mapping, random
   bucket-snapped preemptions) — this floor of randomness is unchanged
   from v1 and exists to keep the GA exploring outside whatever the
   model currently believes.

3. **GA evolution loop**, `ga_generations_per_iter` generations:
   - Elitism: carry the top 2 individuals forward unchanged.
   - Crossover + mutation fill most of the rest (`ga.py`, untouched by
     the v2 rewrite).
   - Constructive injection: a further `num_constructive` individuals
     sampled fresh from the model each generation (same two-phase
     sample-then-build-table-then-predict sequence as step 2).
   - Every individual is scored by the simulator
     (`ga.evaluate_population_parallel` → `simulator.simulate_single_mapping`).
   - Every `update_interval` generations, an in-loop REINFORCE update
     fires on the current population's top third.

4. **Convergence accounting**: track the best fitness seen so far;
   reset a stability counter on real improvement, increment it
   otherwise; stop once `stability_target` iterations pass with no
   real improvement.

5. **End-of-outer REINFORCE update**: one more update on the final
   generation's population, every iteration (including the first —
   there is no warm-up-only first iteration in v2, unlike v1).

6. **Output**: best mapping/preemptions and the top-100 population
   snapshot written to `top_100_data_<N>cores_<M>io.txt`; model
   checkpoint saved (if `save_model=True`).

---

## Architecture

### Two phases, run in sequence, not interleaved per-core

**Phase 1 (mapping):** every core attends to every IO channel in
parallel via multi-head cross-attention — cores project to queries,
IOs project to keys/values, softmax over IOs gives a categorical
distribution per core, one IO is hard-sampled per core
(`model.CoreIOAttention`). This is standard transformer-style
cross-attention, not a literal graph-library GAT layer over an
explicit bipartite graph: since every core can route to every IO
(fully connected, not sparse), there's no adjacency structure to
encode — the attention mechanism itself *is* the compatibility
function GAT would compute, just applied without a graph object,
because full connectivity needs none.

**Phase 2 (preemption), run only after phase 1 fully resolves:** for
each core, a single-layer Graph Attention Network
(`model.ConflictGAT`) operates over the **conflict graph** — an edge
between two cores exists iff their routing paths would contend for a
shared physical mesh link, per `data.check_path_conflict`, given the
*resolved* mapping. Each core's resulting conflict-embedding is
concatenated with its own (raw-feature) projection and fed through an
MLP for a per-core preemption bucket.

Why sequential, not parallel: phase 2's whole point is to react to
*actual* co-tenancy and conflict structure on the mesh, which only
exists once phase 1 has committed to real assignments. A version that
reused phase 1's pre-sampling attention *scores* instead of resampled,
resolved assignments would be cheaper (one pass instead of two) but
would condition preemption on a soft belief about who *might* share a
channel rather than the ground truth of who *actually* does — and
since the simulator's scheduler operates on the actual resolved
mapping, that mismatch would teach the preemption head the wrong
thing.

### Why the conflict graph, not mesh-topology proximity or per-IO grouping

Three designs were considered for phase 2, and it's worth recording
why the middle two were rejected, not just that the third was chosen:

- **Mesh-topology proximity** (a GNN over the physical mesh graph,
  independent of the sampled mapping): captures "what's physically
  near me," which is a *proxy* for conflict, not conflict itself. Two
  cores can be mesh-adjacent yet never conflict (their routes never
  cross), and two cores far apart on the mesh can still conflict (long
  routing paths crossing). A proxy signal here is strictly weaker than
  the real thing, and the real thing is cheap to compute.
- **Per-IO grouping** (attend only over cores sharing the *same* IO
  channel): conflates "shares my IO channel" with "conflicts with
  me," which `simulator.py`'s own scheduling loop disproves —
  `check_path_conflict` is checked against every other core's
  schedule log regardless of IO assignment, not restricted to
  same-channel cores. `tests/test_data.py`'s
  `test_build_conflict_table_can_include_cross_io_conflicts` is a
  permanent regression guard for this: real cross-IO conflicts do
  occur, and a per-IO-grouped design would silently miss them.
- **The conflict graph** (chosen): built directly from
  `check_path_conflict`, the exact predicate the simulator's scheduler
  uses to decide when two cores must serialize. No proxy, no
  IO-channel restriction — this is the actual structure that
  determines whether fine or coarse preemption helps a given core.

### Why `patterns` and `scan` are model inputs now (they weren't in v1)

v1's core input was position only (`[row, col]`). v2's is
`[row, col, patterns, scan]`. Both `patterns` (test-pattern count —
the workload a core must chop into subtasks) and `scan` (scan-chain
shift-in/shift-out length — a fixed per-subtask overhead) are real,
independently-varying, per-core properties in standard NoC test
benchmarks (ITC'02-style format; confirmed against two papers using
the same `T = (1 + max(Si,So))*TP + min(Si,So)` per-core test-time
formula this repo's simulator mirrors). Both materially affect the
*optimal* preemption value for a core:

- `scan` sets the overhead-repeat cost: finer preemption creates more
  subtasks, and `scan`'s fixed cost is paid once per subtask — so a
  core with large `scan` generally prefers coarser preemption to avoid
  re-paying that overhead repeatedly, while `scan` near 0 cores are
  close to overhead-indifferent to preemption granularity.
- `patterns` sets how much work there is to chop up at all, hence how
  much room finer-grained scheduling flexibility even has to matter.

Neither existed in the model's input in v1 — this is a genuine input
gap the v2 rewrite closes, independent of the two-phase restructuring
itself. Note: `data.py`'s synthetic dummy-data fallback (used when no
benchmark file exists) generates identical `patterns`/`scan` for every
core — this is an artifact of the placeholder generator, not a
modeling assumption; real benchmark files carry per-core-varying
values, parsed independently per line.

### No gradient detaching between phases (v1's fix #6 is retired)

v1 detached the preemption head's input (`output.detach()`) so
preemption-loss gradients could never reach the mapping backbone. That
existed for a specific, structural reason: v1's backbone was
*pretrained separately* (`pretrain.py`, now removed — see below)
before the preemption head existed in any trained state, so an early,
noisy, randomly-initialized head's REINFORCE gradient could otherwise
degrade an already-good backbone. v2 has no such asymmetry — nothing
is pretrained separately, and both phases train together via one
REINFORCE update from the first outer iteration. Cutting the gradient
path here would do something different from what it did in v1: it
would prevent the shared `core_proj` embedding from ever learning
*which mapping decisions lead to good or bad preemption/scheduling
outcomes downstream* — information only the preemption loss's
gradient carries, and information the mapping phase should arguably be
sensitive to (a mapping that creates a conflict-heavy channel is bad,
even if the mapping loss alone never says why).

Concretely: `core_proj` (the shared per-core feature projection) is
the *only* component genuinely shared between phases, and it receives
gradient from **both** `mapping_lp` and `preempt_lp` in the same
backward pass. `phase1`'s attention weights and `conflict_gat` +
`preempt_mlp` are each phase-private — but that's a structural
consequence of which computation actually uses them (phase 2 never
calls phase 1's attention module, and vice versa), not the result of
any `detach()` call. `tests/test_model_shapes.py`'s
`test_no_detach_preempt_loss_reaches_shared_core_proj` is the
regression guard for this; `test_no_detach_mapping_loss_reaches_shared_and_phase1_only`
checks the same property from the mapping side.

### What v2 removed, and why

- **`pretrain.py`, and `--mode pretrain`** — v1 pretrained the mapping
  backbone alone before the preemption head existed in any trained
  form. v2 has no separate backbone to pretrain in isolation; both
  phases are meant to co-train from initialization.
- **Fix #3's supervised head warm-up
  (`warmup_preemption_head`), and `--disable_warmup` /
  `--head_warmup_epochs`** — existed to get v1's freshly-bolted-on,
  randomly-initialized head roughly sane via supervised imitation
  before it was trusted to contribute constructively to the GA
  population. v2 has no "freshly bolted on after the fact" head — both
  phases start randomly and get REINFORCE signal together immediately,
  so there's no asymmetry left for a warm-up to correct. If
  simultaneous-random-init training turns out noisier in practice than
  a warm-started head would have been, that's a new, empirical
  question specific to v2 — worth watching, but not something
  pre-solved by re-importing a mechanism whose justification was tied
  to a different architecture's structure.

These are described here as removed with reasons attached, not
silently absent — if either turns out to matter in practice, the
reasoning above is exactly what should be revisited first.

---

## Architecture map

```
src/lmga_igsa/
    config.py       constants -- v1 dims retained for reference, v2 dims added  (leaf)
    data.py          problem instances, check_path_conflict, build_conflict_table,
                     create_core_features_v2 (vectorized; ~27x faster than the
                     original nested-loop conflict check at 64 cores)             (leaf)
    simulator.py      makespan fitness function; accepts an optional precomputed
                      conflict table (build_conflict_table's output) and looks
                      values up instead of recomputing check_path_conflict inline (leaf)
    ga.py              Individual / Population / genetic operators, unchanged
                       from v1 except where v2's model needed new call shapes     (leaf)
    model.py             PointerNetV2: CoreIOAttention (phase 1) + ConflictGAT
                         (phase 2) + preemption MLP. model_v1_backup.py is the
                         retired LSTM version, kept for reference, not imported.
    reinforce.py           REINFORCE update only -- no warm-up function; model x ga glue
    run.py                   outer-loop orchestration (run_lmga), no warm-up branch
    cli.py                     argparse + dispatch -- run/multirun only
scripts/cli.py    thin wrapper -- `python scripts/cli.py ...` without installing
tests/            one file per leaf module, plus model.py and reinforce.py
```

Dependency direction is still the thing to understand first, and it's
unchanged in spirit from v1: `model.py` and `ga.py` don't import each
other. `reinforce.py` is still the only place they're combined. The
new piece is `data.build_conflict_table` sitting *between* them: it's
computed once, from a resolved mapping, and threaded into **both**
`model.py`'s phase 2 and `simulator.py`'s scheduling loop as a shared
argument -- this was a deliberate choice over letting the model and
the simulator each build their own copy, since the two are exactly the
same O(N^2) computation on the same fixed mapping and building it
twice would be pure waste. `model.py` itself never calls
`build_conflict_table` -- that would require branching inside
`forward()` on a just-sampled value, which breaks the pure
tensor-in/tensor-out shape that let this be unit-tested with fake
tensors in the first place. `run.py`'s `_sample_individuals` helper is
where the two-phase sample -> build-table -> predict sequence actually
lives.

## Testing

```bash
python -m pytest tests/ -v
```

35 tests. The ones worth reading first, in order of how central they
are to this architecture's actual correctness claims:

1. `tests/test_model_shapes.py::test_no_detach_preempt_loss_reaches_shared_core_proj`
   and `test_no_detach_mapping_loss_reaches_shared_and_phase1_only` -- the
   central architectural property distinguishing v2 from v1.
2. `tests/test_data.py::test_build_conflict_table_can_include_cross_io_conflicts`
   -- the empirical justification for building phase 2 around the
   conflict graph rather than per-IO grouping.
3. `tests/test_model_shapes.py::test_predict_preemption_handles_fully_isolated_cores_without_nan`
   and `test_predict_preemption_handles_all_isolated_batch` -- a core
   with zero conflict-graph edges is common, not an edge case, and
   produces `NaN` from a naive softmax without explicit handling.
4. `tests/test_data.py::test_build_conflict_table_matches_scalar_reference_across_many_instances`
   -- the vectorized conflict-table implementation checked against an
   obviously-correct nested-loop reference across 200 random
   instances; a broadcasting mistake here would be easy to miss
   without an exhaustive check.
5. `tests/test_simulator.py::test_simulate_with_precomputed_conflict_table_matches_inline`
   -- confirms the simulator's lookup-based fast path is byte-identical
   to its original inline-recomputation path.

---

## Development log

### v1 (LSTM pointer-network) -- retired, see model_v1_backup.py
Built as a restructuring of a single research script into an
installable package: `config.py`/`data.py`/`simulator.py`/`ga.py`
split as independent leaves, `model.py` as an LSTM encoder-decoder
with attention plus an IO-conditioned preemption head
(`output.detach()`-isolated, fix #6), `reinforce.py` combining a
REINFORCE update with a supervised warm-up phase (fix #3),
`pretrain.py` for backbone-only pretraining. Fully tested (30 tests),
verified end-to-end including a clean install/reinstall cycle and the
installed console script.

### v2 (two-phase attention/GAT) -- current
Complete architectural redesign, driven by wanting the model to
reason about *why* a given preemption value is good, not just *what*
value to predict:

1. **Conflict table as a shared, vectorized primitive**
   (`data.build_conflict_table`): built once per resolved mapping,
   consumed by both the model's phase 2 and the simulator's scheduling
   loop. Vectorized from an O(N^2)-Python-calls nested loop to
   O(N^2) NumPy broadcasting (~27x faster at 64 cores), verified
   exactly equivalent to the scalar reference across 200 randomized
   instances before being trusted.
2. **`simulator.py`** extended (not rewritten) to accept an optional
   precomputed conflict table via a new 7-tuple argument form,
   verified byte-identical to the original inline-computation path
   across 20+ random mappings on a 16-core instance before being
   trusted.
3. **`model.py`** fully rewritten: `CoreIOAttention` (phase 1,
   multi-head cross-attention, cores as queries / IOs as keys-values)
   and `ConflictGAT` (phase 2, standard GAT attention coefficients
   over the conflict graph, with explicit isolated-node handling to
   avoid the NaN a naive softmax would produce). Both phases share
   `core_proj`; no gradient detaching between them (a deliberate,
   explained departure from v1's fix #6 -- see "Architecture" above).
   Core input grew from `[row, col]` to `[row, col, patterns, scan]`,
   closing a real input gap v1 had (checked against two NoC-testing
   papers' per-core test-time formulas to confirm `patterns`/`scan`
   are genuinely per-core-varying properties in the standard benchmark
   format, not constants).
4. **`reinforce.py`** rewritten around a single `_do_model_update`
   entry point -- v1's `warmup_preemption_head` (fix #3) is retired,
   with the reasoning recorded in the module docstring and in this
   README's "What v2 removed" section, not silently dropped.
5. **`run.py`** rewritten: no warm-up-only first iteration; every
   outer iteration follows the same sample -> evolve -> update sequence
   via a new `_sample_individuals` helper that threads the two-phase
   sampling (mapping, then per-individual conflict table, then
   preemption) through both initial-population and constructive
   injection.
6. **`pretrain.py` and `--mode pretrain` removed entirely** (not
   deprecated-and-ignored) -- v2 has no separate backbone to pretrain,
   so keeping a mode that silently did nothing would be worse than
   removing it and letting a stale script fail loudly with argparse's
   normal error.
7. **Full test suite rewritten for v2's contracts**: `test_data.py`
   added (conflict-table correctness, new in v2);
   `test_model_shapes.py` and `test_reinforce.py` rewritten around
   `PointerNetV2`'s actual methods and the no-detach property, in
   place of v1's fix-#6-specific detach checks. 35/35 passing,
   verified after a clean uninstall/reinstall and against the
   installed console script end-to-end (both a 4-core smoke test and a
   16-core stress test with real cross-IO conflicts present).

---

## What's left to do

A few things are genuinely open, not yet decided or verified, and
worth flagging explicitly rather than letting the README's polish
imply everything is settled:

- **Phase 1's IO input features.** `run.py` currently feeds each IO
  channel's raw `[src, sink]` mesh coordinates into `io_proj` as phase
  1's key/value input. This was a pragmatic choice (it's the most
  natural numeric feature already available) made while wiring
  `run.py`, not something explicitly discussed and agreed on the way
  the rest of the architecture was -- v1 had no analogous IO *feature*
  at all (only an index-keyed embedding table). Worth revisiting
  deliberately rather than treating as settled.
- **No empirical results yet.** Every verification in this repo is
  shape/gradient/logic correctness (does the code do what it's
  supposed to), not whether v2 actually maps and schedules better than
  v1, or better than heuristic baselines. Nothing here has been run
  long enough, or on real (non-synthetic-fallback) benchmark data, to
  produce a real makespan comparison.
- **No warm-up's practical consequences are unverified.** The
  reasoning for retiring fix #3 is structural and sound, but whether
  simultaneous random-init training of both phases is actually stable
  and sample-efficient in practice -- versus subtly worse than v1's
  staged approach in ways the reasoning didn't anticipate -- is an
  empirical question with no data behind it yet.
- **`mapping_scores.mean(dim=1)`** (averaging attention heads down to
  one distribution for sampling, in `CoreIOAttention`) was a stated,
  flagged design choice, not a forced consequence of "multi-head
  attention" -- an alternative (a learned linear reduction from H heads
  to 1) was named but not built or compared against.
- **GPU/scale behavior is untested.** Every run in this session was
  CPU, small (4-16 cores). Nothing about batch sizes, memory, or
  wall-clock behavior at the paper's actual scale (32-64 cores, full
  `pop_size=100`/`ga_generations_per_iter=100`/`stability_target=20`
  defaults) has been exercised end-to-end -- only smoke-tested at a
  fraction of that scale with short timeouts.
- **`model_v1_backup.py`** sits in the package directory unused and
  unimported -- fine as a reference during this transition, but worth a
  deliberate decision later on whether it stays in the shipped
  package, moves to a `legacy/` or docs location, or gets deleted once
  v2 is trusted.