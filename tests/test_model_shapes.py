"""
Tests for PointerNetV2 (model.py): shape correctness for phase 1
(sample_mapping/mapping_log_prob) and phase 2
(predict_preemption/preemption_log_prob), plus the two properties that
matter most in this architecture:

  - NO-DETACH: unlike v1's fix #6, preempt_lp's gradient DOES reach
    the shared core_proj parameters (confirmed in both directions —
    mapping_lp and preempt_lp each reach their own phase's private
    parameters and the shared core_proj, and do NOT reach the other
    phase's private parameters, purely because the other phase's
    computation is never invoked, not via any detach()).
  - ISOLATED CORES: a core with zero conflict-graph neighbors must not
    produce NaN (softmax over an all -inf row is undefined without
    explicit handling) — this is an expected, common case (many cores
    won't conflict with anyone), not a rare edge case.
"""

import numpy as np
import torch

from src.lmga_igsa.data import build_conflict_table
from src.lmga_igsa.model import PointerNetV2

DEVICE = torch.device("cpu")


def _tiny_model(hidden_dim=32, num_heads=4):
    return PointerNetV2(hidden_dim=hidden_dim, num_heads=num_heads, device=DEVICE)


def _tiny_inputs(batch=2, num_cores=8, num_io=2):
    core_raw = torch.randn(batch, num_cores, 4)
    io_raw = torch.randn(batch, num_io, 2)
    return core_raw, io_raw


def _conflict_adj_for(mapping, num_cores, num_io, batch, seed=0):
    rng = np.random.default_rng(seed)
    dir_np = rng.integers(0, 4, size=(num_cores, 2)).astype(float)
    io = [[1, 2], [num_cores // 2, num_cores]] if num_io == 2 else \
        [[k * (num_cores // num_io) + 1, min(num_cores, k * (num_cores // num_io) + 6)] for k in range(num_io)]
    tables = [
        torch.from_numpy(build_conflict_table(dir_np, mapping[b].detach().numpy(), io).astype("float32"))
        for b in range(batch)
    ]
    return torch.stack(tables)


def _grad_norm(params):
    return sum((p.grad.abs().sum().item() if p.grad is not None else 0.0) for p in params)


# ---------------------------------------------------------------------
# Phase 1: mapping
# ---------------------------------------------------------------------

def test_sample_mapping_shapes():
    model = _tiny_model()
    core_raw, io_raw = _tiny_inputs(batch=2, num_cores=8, num_io=2)
    mapping, mapping_lp, core_h = model.sample_mapping(core_raw, io_raw)
    assert mapping.shape == (2, 8)
    assert mapping_lp.shape == (2,)
    assert core_h.shape == (2, 8, 32)


def test_sample_mapping_indices_in_valid_range():
    model = _tiny_model()
    core_raw, io_raw = _tiny_inputs(batch=2, num_cores=8, num_io=3)
    mapping, _, _ = model.sample_mapping(core_raw, io_raw)
    assert mapping.min().item() >= 0
    assert mapping.max().item() < 3


def test_mapping_log_prob_matches_teacher_forced_target():
    model = _tiny_model()
    core_raw, io_raw = _tiny_inputs(batch=3, num_cores=6, num_io=2)
    target = torch.randint(0, 2, (3, 6))
    lp = model.mapping_log_prob(core_raw, io_raw, target)
    assert lp.shape == (3,)
    # log-probs of a valid categorical distribution are <= 0
    assert (lp <= 0).all()


# ---------------------------------------------------------------------
# Phase 2: preemption
# ---------------------------------------------------------------------

def test_predict_preemption_shapes():
    model = _tiny_model()
    core_raw, io_raw = _tiny_inputs(batch=2, num_cores=8, num_io=2)
    mapping, _, _ = model.sample_mapping(core_raw, io_raw)
    conflict_adj = _conflict_adj_for(mapping, 8, 2, batch=2)
    preempt, preempt_lp = model.predict_preemption(core_raw, conflict_adj)
    assert preempt.shape == (2, 8)
    assert preempt_lp.shape == (2,)
    assert preempt.min().item() >= 0
    assert preempt.max().item() < model.num_preemption_buckets


def test_predict_preemption_handles_fully_isolated_cores_without_nan():
    # Regression guard: a core with zero conflict-graph edges produced
    # NaN via naive softmax before this was handled explicitly.
    model = _tiny_model()
    core_raw = torch.randn(1, 6, 4)
    adj = torch.zeros(1, 6, 6)
    adj[0, 0, 1] = 1
    adj[0, 1, 0] = 1
    adj[0, 4, 5] = 1
    adj[0, 5, 4] = 1
    # cores 2 and 3 remain fully isolated -- no edges at all

    preempt, preempt_lp = model.predict_preemption(core_raw, adj)
    assert not torch.isnan(preempt_lp).any()
    assert not torch.isnan(preempt.float()).any()


def test_predict_preemption_handles_all_isolated_batch():
    # Extreme case: every core isolated (empty conflict graph entirely).
    model = _tiny_model()
    core_raw = torch.randn(1, 5, 4)
    adj = torch.zeros(1, 5, 5)
    preempt, preempt_lp = model.predict_preemption(core_raw, adj)
    assert not torch.isnan(preempt_lp).any()


# ---------------------------------------------------------------------
# The no-detach property: gradient reaches shared params from BOTH
# losses; each phase's private params only get gradient from their
# own loss (because the other phase's forward pass never touches
# them, not via any detach()).
# ---------------------------------------------------------------------

def test_no_detach_mapping_loss_reaches_shared_and_phase1_only():
    model = _tiny_model()
    core_raw, io_raw = _tiny_inputs(batch=2, num_cores=8, num_io=2)
    mapping, mapping_lp, _ = model.sample_mapping(core_raw, io_raw)

    model.zero_grad()
    mapping_lp.sum().backward()

    assert _grad_norm(model.core_proj.parameters()) > 0
    assert _grad_norm(model.phase1.parameters()) > 0
    assert _grad_norm(model.conflict_gat.parameters()) == 0
    assert _grad_norm(model.preempt_mlp.parameters()) == 0


def test_no_detach_preempt_loss_reaches_shared_core_proj():
    # The central architectural property distinguishing v2 from v1's
    # fix #6: preemption loss gradient DOES reach the shared core_proj
    # parameters (v1 detached this path deliberately; v2 does not, by
    # design -- see model.py's module docstring for why).
    model = _tiny_model()
    core_raw, io_raw = _tiny_inputs(batch=2, num_cores=8, num_io=2)
    mapping, _, _ = model.sample_mapping(core_raw, io_raw)
    conflict_adj = _conflict_adj_for(mapping, 8, 2, batch=2)
    _, preempt_lp = model.predict_preemption(core_raw, conflict_adj)

    model.zero_grad()
    preempt_lp.sum().backward()

    assert _grad_norm(model.core_proj.parameters()) > 0, \
        "no-detach violated: preempt_lp should reach shared core_proj"
    assert _grad_norm(model.conflict_gat.parameters()) > 0
    assert _grad_norm(model.preempt_mlp.parameters()) > 0
    # phase1's attention params are untouched by preempt_lp -- not
    # because of a detach, but because predict_preemption() never
    # calls self.phase1(...) at all.
    assert _grad_norm(model.phase1.parameters()) == 0


# ---------------------------------------------------------------------
# get_log_prob_components: the contract reinforce.py depends on
# ---------------------------------------------------------------------

def test_get_log_prob_components_without_preempt_target_returns_none():
    model = _tiny_model()
    core_raw, io_raw = _tiny_inputs(batch=3, num_cores=6, num_io=2)
    target_mapping = torch.randint(0, 2, (3, 6))
    mapping_lp, preempt_lp = model.get_log_prob_components(core_raw, io_raw, target_mapping)
    assert preempt_lp is None
    assert mapping_lp.shape == (3,)


def test_get_log_prob_components_with_preempt_target_returns_both():
    model = _tiny_model()
    core_raw, io_raw = _tiny_inputs(batch=3, num_cores=6, num_io=2)
    target_mapping = torch.randint(0, 2, (3, 6))
    target_preempt = torch.randint(0, model.num_preemption_buckets, (3, 6))
    conflict_adj = _conflict_adj_for(target_mapping, 6, 2, batch=3)

    mapping_lp, preempt_lp = model.get_log_prob_components(
        core_raw, io_raw, target_mapping, conflict_adj, target_preempt
    )
    assert mapping_lp.shape == (3,)
    assert preempt_lp.shape == (3,)