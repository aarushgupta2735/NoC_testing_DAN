"""
Global configuration for LMGA_IGSA.

Kept as a single leaf module (no imports from the rest of the package)
so that every other module can depend on it without creating cycles.
"""

# ---- Preemption action space -------------------------------------------
# The preemption head predicts a *discrete* bucket index, not a
# continuous value. This is what makes fix #2 (action/reward
# consistency) possible: every preemption value that ever appears in
# the population is guaranteed to be one of these buckets, so
# `preemptions_to_bucket_indices` round-trips exactly. Still live and
# used by both v1 (model_v1_backup.py) and v2 (model.py).
PREEMPTION_BUCKETS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
NUM_PREEMPTION_BUCKETS = len(PREEMPTION_BUCKETS)

# ---- GA / outer-loop defaults -------------------------------------------
NUM_CONSTRUCTIVE = 15       # NN-sampled individuals injected per GA generation
UPDATE_INTERVAL = 10        # in-loop REINFORCE update every N generations

# ---- Model dims (v2, attention/GAT architecture — LIVE) -------------------
# Per-core input: [row, col, patterns, scan]. patterns/scan are real,
# per-core-varying properties in standard NoC test benchmarks (ITC'02
# format) — see data.py's Core class and prep_data's per-line parsing.
CORE_INPUT_DIM_V2 = 4
IO_INPUT_DIM_V2 = 2           # IO pairs currently carry (src, sink) as their raw feature
HIDDEN_DIM_V2 = 128
NUM_ATTN_HEADS_V2 = 4         # phase 1 (core->IO cross-attention) and phase 2 (conflict GAT) both use this
GAT_LEAKY_SLOPE_V2 = 0.2      # standard GAT negative slope for its LeakyReLU

# ---- REINFORCE (LIVE) ------------------------------------------------------
EMA_DECAY = 0.9
IMPROVEMENT_THRESHOLD = 0.005
GRAD_CLIP_NORM = 1.0

# =============================================================================
# ARCHIVAL — v1 (LSTM pointer-net) constants below this line.
# Used only by model_v1_backup.py, which is not imported anywhere in
# the live package (kept for reference; see model_v1_backup.py and the
# README's "Development log" for why v1 was replaced). Nothing in
# model.py, reinforce.py, run.py, or cli.py reads these. Kept here
# rather than deleted so model_v1_backup.py remains runnable in
# isolation if someone wants to diff v1 against v2 directly.
# =============================================================================

# v1 model dims
INPUT_DIM = 2                # (row, col) core coordinates -- v1 had no patterns/scan input
HIDDEN_DIM = 128
NUM_RNN_LAYERS = 2
DROPOUT_P = 0.1

# v1 IO conditioning (fix #1) -- v1's io_embed was an index-keyed
# embedding table; v2's io_proj instead projects each IO's raw
# [src, sink] features directly (see IO_INPUT_DIM_V2 above).
MAX_NUM_IO = 16
IO_EMBED_DIM = 16

# v1 head warm-up (fix #3) -- retired in v2; see reinforce.py's module
# docstring for why the warm-up concept doesn't apply to this
# architecture. Not read by any v1 OR v2 code path currently (v1's
# pretrain.py, which used these, was also removed) -- kept only as a
# record of what the original values were.
HEAD_WARMUP_EPOCHS = 30
HEAD_WARMUP_LR = 3e-4