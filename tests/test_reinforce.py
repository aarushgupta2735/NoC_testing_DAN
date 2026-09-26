"""
Tests for reinforce.py (v2) — _do_model_update is the sole entry
point; v1's warmup_preemption_head is retired (see reinforce.py's
module docstring for why). These tests exercise the function directly
against a real PointerNetV2 and a fake Population, checking which
parameters move under the full update versus the
--disable_preemption_head ablation.
"""

import random

import numpy as np
import torch

from lmga_igsa.config import PREEMPTION_BUCKETS
from lmga_igsa.ga import Individual, Population
from lmga_igsa.model import PointerNetV2
from lmga_igsa.reinforce import _do_model_update

DEVICE = torch.device("cpu")


def _tiny_model():
    return PointerNetV2(hidden_dim=32, num_heads=4, device=DEVICE)


def _fake_population(num_cores=8, num_io=2, size=10, seed=0):
    rng = random.Random(seed)
    inds = []
    for _ in range(size):
        genes = [rng.randint(0, num_io - 1) for _ in range(num_cores)]
        preemptions = [rng.choice(PREEMPTION_BUCKETS) for _ in range(num_cores)]
        fitness = -rng.uniform(100, 1000)
        inds.append(Individual(genes, fitness=fitness, preemptions=preemptions))
    return Population(inds)


def _problem_setup(num_cores=8, num_io=2, seed=0):
    rng = np.random.default_rng(seed)
    core_features = torch.randn(1, num_cores, 4)
    io_features = torch.randn(1, num_io, 2)
    dir_np = rng.integers(0, 4, size=(num_cores, 2)).astype(float)
    io_pairs = [[1, 2], [num_cores // 2, num_cores]]
    return core_features, io_features, dir_np, io_pairs


def _snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _changed(before, after, prefix):
    return any(
        not torch.allclose(before[n], after[n])
        for n in before if n.startswith(prefix)
    )


def test_do_model_update_trains_all_components_together():
    # No warm-up in v2: a single _do_model_update call should move
    # EVERY component -- shared core_proj, phase-1-private attention,
    # AND phase-2-private conflict_gat/preempt_mlp -- since nothing is
    # pretrained separately and both phases learn from the start.
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    population = _fake_population()
    core_features, io_features, dir_np, io_pairs = _problem_setup()

    before = _snapshot(model)
    ema = _do_model_update(model, optimizer, population, core_features, io_features,
                            dir_np, io_pairs, ema_baseline=None, train_preempt_head=True)
    after = _snapshot(model)

    assert ema is not None
    assert _changed(before, after, "core_proj")
    assert _changed(before, after, "phase1")
    assert _changed(before, after, "conflict_gat")