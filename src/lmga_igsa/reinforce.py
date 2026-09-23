"""
The bridge between model.py and ga.py: turns a scored Population into
gradient updates on the PointerNet.

Two entry points:

  _do_model_update        — the standard REINFORCE step, run either
                             in-loop (every UPDATE_INTERVAL generations)
                             or once at the end of each outer iteration.
  warmup_preemption_head  — fix #3: supervised NLL pretraining of the
                             preemption head on the warm-up GA pass's
                             top-K, run once, before the head's first
                             constructive sample is ever used.

Both rely on fix #2 (population preemptions are bucket-aligned, so
`preemptions_to_bucket_indices` gives exact action indices) and fix #6
(model.get_log_prob_components internally detaches the backbone from
the preemption loss, so calling it here never risks contaminating
encoder/decoder/attn with preemption-head gradient).
"""

import torch

from .config import EMA_DECAY, GRAD_CLIP_NORM, HEAD_WARMUP_EPOCHS, HEAD_WARMUP_LR
from .ga import preemptions_to_bucket_indices


def _do_model_update(model, optimizer, population, data_tensor, mask, num_io_local,
                      ema_baseline, ema_decay=EMA_DECAY, train_preempt_head=True):
    """
    Combined REINFORCE update on the top 1/3 of the population.

    Reward is each individual's simulator fitness (already computed by
    ga.evaluate_population_parallel). Baseline is an EMA of the mean
    reward among the training slice, used to reduce variance
    (advantage = reward - baseline, then standardized).

    train_preempt_head=False (the --disable_preemption_head ablation)
    skips computing/using the preemption log-prob entirely, so the
    update becomes pure mapping-head REINFORCE.

    Returns the updated ema_baseline (the caller threads this back in
    on the next call — it's session state, not model state).
    """
    population.sort()
    top_k = max(5, len(population.individuals) // 3)
    train_inds = population.individuals[:top_k]

    train_genes = torch.tensor([ind.genes for ind in train_inds], dtype=torch.long)
    train_preemption_indices = torch.tensor(
        [preemptions_to_bucket_indices(ind.preemptions) for ind in train_inds],
        dtype=torch.long,
    )
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

    batch_size_train = len(train_inds)
    batch_input = data_tensor[:, 0:1, :].repeat(1, batch_size_train, 1)
    batch_mask = mask.repeat(batch_size_train, 1)

    target_preempts = train_preemption_indices if train_preempt_head else None
    mapping_lp, preempt_lp = model.get_log_prob_components(
        batch_input, batch_mask, num_io_local, train_genes, target_preempts
    )

    total_lp = mapping_lp + (preempt_lp if preempt_lp is not None else 0)
    pg_loss = -(total_lp * advantages).mean()
    pg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
    optimizer.step()

    return ema_baseline


def warmup_preemption_head(model, train_samples, data_tensor, mask, num_io_local,
                            epochs=HEAD_WARMUP_EPOCHS, lr=HEAD_WARMUP_LR):
    """
    Fix #3: supervised NLL pretraining of the preemption head on
    (genes, preemptions) pairs from the warm-up GA pass's top-K, before
    the head is ever used to produce a constructive sample.

    The optimizer is restricted to io_embed + preemption_head params
    only (belt-and-suspenders on top of model.get_log_prob_components's
    internal output.detach() — fix #6 — which already prevents any
    gradient from reaching the backbone here).

    train_samples: list of (genes, preemptions, fitness) tuples. Only
    genes and preemptions are used; fitness is accepted for call-site
    convenience (callers typically build this list alongside fitness
    for logging) but ignored here — this is supervised imitation of
    the GA's outcome, not a reward-weighted objective.
    """
    if not train_samples:
        return

    head_params = list(model.io_embed.parameters()) + list(model.preemption_head.parameters())
    head_opt = torch.optim.Adam(head_params, lr=lr)

    genes_list = [s[0] for s in train_samples]
    preempt_list = [s[1] for s in train_samples]
    bucket_targets = [preemptions_to_bucket_indices(p) for p in preempt_list]

    genes_t = torch.tensor(genes_list, dtype=torch.long)
    bucket_t = torch.tensor(bucket_targets, dtype=torch.long)

    bs = len(train_samples)
    batch_input = data_tensor[:, 0:1, :].repeat(1, bs, 1)
    batch_mask = mask.repeat(bs, 1)

    model.train()
    for e in range(epochs):
        head_opt.zero_grad()
        _, preempt_lp = model.get_log_prob_components(
            batch_input, batch_mask, num_io_local, genes_t, bucket_t
        )
        loss = -preempt_lp.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params, max_norm=GRAD_CLIP_NORM)
        head_opt.step()
        if (e + 1) % 10 == 0:
            print(f"  [Head Warmup] epoch {e + 1}/{epochs}, NLL: {-preempt_lp.mean().item():.4f}")