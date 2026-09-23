"""
Global configuration for LMGA-IGSA-FIXED.

Kept as a single leaf module (no imports from the rest of the package)
so that every other module can depend on it without creating cycles.
"""

# ---- Preemption action space -------------------------------------------
# The preemption head predicts a *discrete* bucket index, not a
# continuous value. This is what makes fix #2 (action/reward
# consistency) possible: every preemption value that ever appears in
# the population is guaranteed to be one of these buckets, so
# `preemptions_to_bucket_indices` round-trips exactly.
PREEMPTION_BUCKETS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
NUM_PREEMPTION_BUCKETS = len(PREEMPTION_BUCKETS)

# ---- GA / outer-loop defaults -------------------------------------------
NUM_CONSTRUCTIVE = 15       # NN-sampled individuals injected per GA generation
UPDATE_INTERVAL = 10        # in-loop REINFORCE update every N generations

# ---- Head warm-up defaults (fix #3) --------------------------------------
HEAD_WARMUP_EPOCHS = 30     # supervised NLL epochs on warm-up top-K
HEAD_WARMUP_LR = 3e-4       # higher LR since the head starts at random init

# ---- IO conditioning (fix #1) --------------------------------------------
MAX_NUM_IO = 16
IO_EMBED_DIM = 16

# ---- Model dims -----------------------------------------------------------
INPUT_DIM = 2                # (row, col) core coordinates
HIDDEN_DIM = 128
NUM_RNN_LAYERS = 2
DROPOUT_P = 0.1

# ---- REINFORCE ------------------------------------------------------------
EMA_DECAY = 0.9
IMPROVEMENT_THRESHOLD = 0.005
GRAD_CLIP_NORM = 1.0