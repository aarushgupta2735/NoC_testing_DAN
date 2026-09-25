"""
The bridge between model.py and ga.py: turns a scored Population into
gradient updates on PointerNetV2.

ONE entry point in v2: _do_model_update. v1's warmup_preemption_head
(fix #3) is retired here — see the note below.

Why fix #3's warm-up doesn't carry over: it existed because v1's
preemption head started at random init AFTER the mapping backbone was
already separately pretrained (pretrain.py) and good — an untrained
head producing noisy REINFORCE actions early would otherwise
contribute nothing useful (or actively harmful signal) to the GA
population before it had ever seen supervised signal. v2 has no such
asymmetry: nothing is pretrained separately, phase 1 and phase 2 both
start at random init and train together via one REINFORCE update from
the very first outer iteration. There is no "backbone that's already
good" for an undertrained head to contaminate, and no separate
pretraining phase before which a warm-up would even make sense to run.
If early-training noise from simultaneous random-init learning turns
out to be a real problem in practice, that's a new, empirical question
for this architecture — not something to solve by re-importing v1's
mechanism, whose justification was specific to v1's structure.

Relies on fix #2 (population preemptions are bucket-aligned, so
preemptions_to_bucket_indices gives exact action indices matching the
reward that was actually evaluated) and requires data.py's
build_conflict_table to bridge phase 1's sampled mapping into phase
2's conflict-aware preemption prediction (see model.py's module
docstring for why conflict-table construction is NOT done inside the
model itself).
"""

import numpy as np
import torch

from .config import EMA_DECAY, GRAD_CLIP_NORM
from .data import build_conflict_table
from .ga import preemptions_to_bucket_indices


def _build_batch_conflict_adj(dir_np, genes_batch, io_pairs):
    """
    Build a (batch, num_cores, num_cores) conflict adjacency tensor,
    one table per individual, from each individual's own mapping
    (genes). Each individual has a different mapping, hence a
    genuinely different conflict graph — there is no cross-individual
    reuse to be had here (see the conversation history in run.py's
    module docstring for why this is computed once per mapping,
    shared between the model's phase 2 and the simulator, rather than
    cached at any coarser granularity).
    """
    tables = [
        build_conflict_table(dir_np, genes, io_pairs)
        for genes in genes_batch
    ]
    stacked = np.stack(tables).astype("float32")
    return torch.from_numpy(stacked)


def _do_model_update(model, optimizer, population, core_features, io_features,
                      dir_np, io_pairs, ema_baseline, ema_decay=EMA_DECAY,
                      train_preempt_head=True):
    """
    Combined REINFORCE update on the top 1/3 of the population.

    Reward is each individual's simulator fitness (already computed by
    ga.evaluate_population_parallel). Baseline is an EMA of the mean
    reward among the training slice; advantage = reward - baseline,
    standardized.

    core_features: (1, num_cores, 4) — [row, col, patterns, scan] for
        this problem instance (from data.create_core_features_v2),
        repeated across the training batch below.
    io_features: (1, num_io, 2) — same repetition pattern.
    dir_np: (num_cores, 2) numpy array of grid positions, needed to
        build each individual's conflict table.
    io_pairs: the ioArray (list of [src, sink]) for this problem.

    train_preempt_head=False (the --disable_preemption_head ablation)
    skips computing/using the preemption log-prob entirely, so the
    update becomes pure mapping-phase REINFORCE — and skips building
    conflict tables at all, since nothing would consume them.

    Returns the updated ema_baseline (session state, threaded by the
    caller across calls — not stored on the model).
    """
    population.sort()
    top_k = max(5, len(population.individuals) // 3)
    train_inds = population.individuals[:top_k]
    batch_size_train = len(train_inds)

    train_genes = torch.tensor([ind.genes for ind in train_inds], dtype=torch.long)
    train_rewards = torch.tensor([ind.fitness for ind in train_inds], dtype=torch.float32)

    current_mean = train_rewards.mean().item()
    if ema_baseline is None:
        ema_baseline = current_mean
    else:
        ema_baseline = ema_decay * ema_baseline + (1 - ema_decay) * current_mean

    advantages = train_rewards - ema_baseline
    adv_std = advantages.std()
    if adv_std > 1e-6:
        advantages = advantages / (adv_std + 1e-6)

    model.train()
    optimizer.zero_grad()

    batch_core_features = core_features.repeat(batch_size_train, 1, 1)
    batch_io_features = io_features.repeat(batch_size_train, 1, 1)

    conflict_adj = None
    target_preempts = None
    if train_preempt_head:
        genes_list = [ind.genes for ind in train_inds]
        conflict_adj = _build_batch_conflict_adj(dir_np, genes_list, io_pairs)
        train_preemption_indices = torch.tensor(
            [preemptions_to_bucket_indices(ind.preemptions) for ind in train_inds],
            dtype=torch.long,
        )
        target_preempts = train_preemption_indices

    mapping_lp, preempt_lp = model.get_log_prob_components(
        batch_core_features, batch_io_features, train_genes, conflict_adj, target_preempts
    )

    total_lp = mapping_lp + (preempt_lp if preempt_lp is not None else 0)
    pg_loss = -(total_lp * advantages).mean()
    pg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
    optimizer.step()

    return ema_baseline