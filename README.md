# LMGA_IGSA

Neural-guided genetic algorithm for core-to-IO mapping on a mesh-based
many-core chip, with a learned, IO-conditioned preemption policy.

A pointer-network policy samples mappings and preemption schedules;
those samples seed and get injected into a genetic algorithm's
population; the GA's best individuals train the policy back via
REINFORCE. The two search processes run in a loop, each making the
other better.

This repo is a restructuring of a single research script
(`lmga_igsa.py`) into an installable package. The algorithm itself is
unchanged — the split preserves four correctness fixes the original
script needed (see below) and adds tests that check each fix directly
rather than only by re-reading the code.

---

## Install

```bash
pip install -e .
```

This installs the `lmga_igsa` package and a `lmga-igsa` console
script. Requires Python ≥3.9; PyTorch, NumPy, and NetworkX are pulled
in automatically (see `pyproject.toml`).

## Quickstart

```bash
# Pretrain the mapping backbone only (preemption head stays at random init)
lmga-igsa --mode pretrain --num_cores 32 --num_io 2 --epoch 1000

# Full NN+GA co-training run
lmga-igsa --mode run --num_cores 32 --num_io 2

# Repeat a run N times for the mean/std numbers a results table needs
lmga-igsa --mode multirun --num_cores 32 --num_io 2 --num_runs 10
```

Add `--cuda` to any mode to use a GPU if one is available; all three
modes default to CPU otherwise.

Benchmark input files (`bm_<N>cores.txt` for `run`, `data_<N>cores.txt`
for `pretrain`) are expected in the working directory you run from. If
missing, a warning prints and synthetic dummy core data is generated
instead — useful for a quick smoke test, not for real numbers.

### Ablation flags (paper Table 5: "what part of IGSA matters?")

| Flag | Effect | Isolates |
|---|---|---|
| `--disable_preemption_head` | random bucket preemptions instead of the learned head | the mapping head's contribution |
| `--disable_constructive` | no NN-sampled individuals injected into the GA | the constructive operator's contribution |
| `--disable_inloop_updates` | only the end-of-outer-iteration update fires | the in-loop update's contribution |
| `--disable_warmup` | skip the supervised head warm-up | whether warm-up is actually needed |
| `--continuous_preemptions` | revert fix #2 — continuous draws instead of bucket-snapped | whether action/reward consistency matters |

Run once with no flags (full model), then once per flag, for the five
rows of an ablation table.

---

## Architecture

```
src/lmga_igsa/
    config.py       constants: preemption buckets, dims, defaults        (leaf)
    data.py          problem instance loading: cores, IO pairs, topology  (leaf)
    simulator.py      makespan fitness function / environment              (leaf)
    model.py            PointerNet: sampling + teacher-forced log-probs     (leaf)
    ga.py                 Individual / Population / genetic operators        (leaf)
    reinforce.py            REINFORCE update + head warm-up  — model × ga glue
    pretrain.py                mapping-backbone-only pretraining
    run.py                        outer-loop orchestration (run_lmga)
    cli.py                           argparse + mode dispatch
scripts/cli.py    thin wrapper — lets `python scripts/cli.py ...` work without installing
tests/            one test file per leaf module, plus reinforce.py
```

**Dependency direction is the thing to understand first.** `model.py`
and `ga.py` do not import each other. `PointerNet` has no notion of a
genetic algorithm; `Individual`/`Population` have no notion of a
neural net. They are introduced to each other in exactly one place —
`reinforce.py` — which takes a scored `Population` and produces
gradient updates on a `PointerNet`. This mirrors the algorithm's real
structure rather than an arbitrary file split, and it's why the model
can be unit-tested with fake tensors and the GA with fake genes,
independently of one another.

---

## Codebase Deep Dive: For Researchers & Developers

If you want to extend this implementation (e.g., swapping the attention mechanism, adding new GA operators, modifying the simulator's logic, or parsing a different dataset), this section describes how the code blocks come together and the usage of various helper functions. 

### How Everything Comes Together (The Pipeline)

The entire algorithm is orchestrated inside `run_lmga` (located in `run.py`). Here is the step-by-step lifecycle of one outer-loop iteration:

1. **Initialization:** `data.py` loads the geometric grid, core workloads, and IO pairs. 
2. **Initial Population:** The Neural Network (`PointerNet`) auto-regressively samples mappings and conditional preemptions for 75% of the initial population. The remaining 25% are generated randomly. 
3. **Evaluation:** `ga.evaluate_population_parallel` passes the raw mapping and preemption genes to the `simulator.py`, which computes a schedule makespan. The population is then sorted by makespan.
4. **GA Evolution (In-Loop):** Over several generations, the GA executes `selection`, `crossover`, and `mutate`. Crucially, at each generation, the NN injects `NUM_CONSTRUCTIVE` fresh samples into the GA (the constructive operator).
5. **In-Loop Model Update:** Periodically during evolution, the top ~30% of the GA population is fed into `reinforce._do_model_update`. The NN calculates teacher-forced log-probabilities (`model.get_log_prob_components`) on the GA's top trajectories, and updates the NN weights using an Advantage Actor-Critic (REINFORCE) loss.
6. **Convergence & Head Warm-up:** The loop monitors for fitness stagnation. In the very first outer loop (if warmup is enabled), a special `warmup_preemption_head` is triggered on the top GA individuals to imitate the GA's schedule using supervised Negative Log Likelihood (NLL).

### Module Breakdown & Helper Functions

#### `config.py`
Stores all global constants to prevent circular imports. 
* **`PREEMPTION_BUCKETS`**: The fixed discrete buckets `[0.20, 0.30, ..., 0.90]` that the NN preemption head predicts, which ensures action/reward consistency.
* **Architecture Dimensions**: e.g., `INPUT_DIM`, `HIDDEN_DIM`, `IO_EMBED_DIM`, etc.

#### `data.py`
Answers the question: *"What does this mapping problem look like?"*
* **`Core` class**: A simple container for workload attributes (patterns, scan, core ID, etc.).
* **`prep_data()`**: Parses the benchmark file, generates a synthetic fallback if the file is missing, and computes all-pairs shortest paths using NetworkX over the chip's mesh topology.
* **`check_path_conflict(...)`**: A geometric collision test used by the simulator. It determines if two cores' subtasks will attempt to use a shared physical link on the mesh.

#### `model.py`
The neural policy consisting of an `Encoder`, a `Decoder`, an `Attention` (mapping) head, and a conditional `preemption_head`.
* **`PointerNet`**: The main model container. 
* **`forward(...)`**: Performs autoregressive *sampling*. Used when generating candidates to inject into the GA.
* **`get_log_prob_components(...)`**: Performs *teacher-forcing*. It replays a fixed sequence (from the GA) and asks "What log-prob would the current policy assign to this?" This is used to compute gradients.
* **Helper `_preempt_logits(...)`**: Contains the logic for detaching the backbone (Fix #6) and conditioning the preemption head on the selected IO channel (Fix #1).

#### `ga.py`
Independent GA mechanics holding `Individual` and `Population`. 
* **Genetic Operators**: `selection(pop, k)` (tournament), `crossover(parent1, parent2)` (single-segment contiguous crossover), and `mutate(ind, ...)` (uniform chance to reroll genes/preemptions).
* **Helper `preemptions_to_bucket_indices(...)`**: Rounds continuous preemption values back to their nearest bucket index. 
* **Helper `bucket_indices_to_preemptions(...)`**: Converts bucket indices to actual real-valued preemptions.
* **Helper `evaluate_population_parallel(...)`**: Efficiently runs the Python multiprocessing pool across all unevaluated individuals, updates their fitness, and sorts the population.

#### `simulator.py`
The fitness environment mapping inputs to makespans.
* **`simulate_single_mapping(args)`**: The core scheduling engine. It builds a Shortest Job First (SJF) queue of subtasks and inserts them into a global schedule layout, deferring tasks via the geometric `check_path_conflict` if needed.
* **Helper `_insert_by_finish(lst, ev)`**: Binary-inserts a task into an active timeline, keeping the timeline sorted by finish time.
* **Helper `_merge_sorted(intervals)`**: Merges overlapping contiguous execution intervals, enabling the simulator to find empty scheduling gaps. 

#### `reinforce.py`
The glue between the Neural Network and the GA.
* **`_do_model_update(...)`**: Takes the top-K population, pulls their genomes and rewards, computes an Exponential Moving Average (EMA) baseline advantage, and performs a gradient step on `PointerNet`. 
* **`warmup_preemption_head(...)`**: Runs supervised Negative Log-Likelihood optimization strictly on the preemption head and IO embeddings using the GA's warmup output.

### File Interactions (Dependency Flow)

Understanding how the files rely on each other is crucial for extending the codebase safely. The architecture adheres to a strict uni-directional flow:

1. **Leaves (`config.py`, `data.py`, `model.py`, `ga.py`, `simulator.py`)**: 
   - These modules **never** import `run.py` or `reinforce.py`.
   - `model.py` and `ga.py` **do not import each other**. The PointerNet has no notion of "genes", and the GA has no notion of "tensors".
   - `simulator.py` only imports `data.py` (for the conflict checker) and `config.py` (for buckets).

2. **The Glue (`reinforce.py`)**:
   - This is the **only** place where `model.py` and `ga.py` interact directly. It translates GA populations (genes/rewards) into PyTorch tensors, computes the REINFORCE loss via `model.get_log_prob_components`, and performs the gradient step.

3. **The Orchestrator (`run.py` & `pretrain.py`)**:
   - These files sit at the very top. They import all the leaves and the glue (`reinforce.py`) to manage the outer training loop, multiprocessing pools, ablation flags, and model I/O.

## The four fixes

The original script had four correctness bugs; this repo preserves
the fixes, each with a direct test (not just a comment) verifying the
property it guarantees:

**#1 — IO-conditioned preemption head** (`model.py`). The preemption
head takes `cat(decoder_output, io_embedding(selected_io))`, not just
`decoder_output`. The right preemption schedule for a core genuinely
depends on which IO channel it was routed to; without this the head
can only represent the marginal `P(preempt | core)`, not the joint
`P(preempt | core, IO)` the problem actually needs.
→ tested in `tests/test_model_shapes.py::test_fix_1_preemption_logits_depend_on_selected_io`

**#2 — Action/reward consistency via bucket-snapped preemptions**
(`simulator.py`, `ga.py`). Every fresh preemption draw — random
initial population, the simulator's fallback, gene-driven or
independent mutation — samples from a fixed discrete bucket set
instead of a continuous range. This guarantees the bucket-index
*action* used in REINFORCE always refers to the same preemption
*value* that produced the observed reward.
→ tested in `tests/test_buckets.py` and `tests/test_simulator.py`

**#3 — Head warm-up via supervised imitation** (`reinforce.py`,
orchestrated from `run.py`). The first outer iteration runs with no
constructive injection and no in-loop updates; afterward, the
preemption head is trained via supervised NLL on the top-K
individuals' `(genes, bucket-index)` pairs. From outer iteration 2
onward the head is in-distribution before its first sample is ever
used constructively, instead of contributing noise from a random init.
→ tested in `tests/test_reinforce.py::test_warmup_preemption_head_only_updates_head_and_io_embed`

**#6 — Backbone gradient isolation via `output.detach()`** (`model.py`).
The preemption head's *training* path receives `output.detach()` as
its feature input. Gradients from the preemption loss can reach
`io_embed` and `preemption_head`, but never the encoder, decoder, or
attention — so the pretrained mapping backbone can't be contaminated
by noisy preemption-head gradients. The mapping loss has no such
detach, and does reach the backbone; the asymmetry is the whole point.
→ tested directly via autograd in
`tests/test_model_shapes.py::test_fix_6_preemption_gradient_does_not_reach_backbone`
and `test_fix_6_mapping_gradient_does_reach_backbone`

## Testing

```bash
python -m pytest tests/ -v
```

30 tests across five files, one per leaf module plus `reinforce.py`.
The two tests worth reading first if you want to understand what
actually matters in this codebase are the fix #6 pair above — they
backprop through each loss term in isolation and assert exactly which
parameters moved.
