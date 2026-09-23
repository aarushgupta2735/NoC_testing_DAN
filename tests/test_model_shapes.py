"""
Tests for PointerNet: shape correctness for both the sampling path
(forward) and the teacher-forced path (get_log_prob_components), plus
the two most important correctness properties in the whole codebase:

  fix #1 — the preemption head is conditioned on the selected IO
           (tested indirectly: different io_indices for the same
           decoder output produce different logits).
  fix #6 — backpropagating through the preemption log-prob touches
           ONLY io_embed + preemption_head; encoder/decoder/attn
           receive exactly zero gradient from it. Backpropagating
           through the mapping log-prob DOES reach the backbone —
           the asymmetry is the point, and both halves are checked.
"""

import torch

from src.lmga_igsa.config import NUM_PREEMPTION_BUCKETS
from src.lmga_igsa.model import PointerNet

DEVICE = torch.device("cpu")


def _tiny_model(hidden_dim=16):
    return PointerNet(input_dim=2, hidden_dim=hidden_dim, n_layers=1, p=0.0, device=DEVICE)


def _tiny_input(num_cores=6, batch=1):
    return torch.randn(num_cores, batch, 2), torch.ones(batch, num_cores)


def test_forward_shapes():
    model = _tiny_model()
    input_tensor, mask = _tiny_input(num_cores=6, batch=1)
    mappings, preemptions, log_probs_sum = model(input_tensor, num_io=2, mask=mask, num_samples=4)
    assert mappings.shape == (4, 6)
    assert preemptions.shape == (4, 6)
    assert log_probs_sum.shape == (4,)


def test_forward_preemption_indices_in_valid_range():
    model = _tiny_model()
    input_tensor, mask = _tiny_input()
    _, preemptions, _ = model(input_tensor, num_io=2, mask=mask, num_samples=3)
    assert preemptions.min().item() >= 0
    assert preemptions.max().item() < NUM_PREEMPTION_BUCKETS


def test_disable_preemption_head_still_returns_valid_indices():
    model = _tiny_model()
    input_tensor, mask = _tiny_input()
    _, preemptions, _ = model(input_tensor, num_io=2, mask=mask, num_samples=3, use_preemption_head=False)
    assert preemptions.min().item() >= 0
    assert preemptions.max().item() < NUM_PREEMPTION_BUCKETS


def test_get_log_prob_components_shapes():
    model = _tiny_model()
    input_tensor, mask = _tiny_input(num_cores=6, batch=1)
    mappings, preemptions, _ = model(input_tensor, num_io=2, mask=mask, num_samples=1)
    mapping_lp, preempt_lp = model.get_log_prob_components(input_tensor, mask, 2, mappings, preemptions)
    assert mapping_lp.shape == (1,)
    assert preempt_lp.shape == (1,)


def test_get_log_prob_components_no_preemption_targets_returns_none():
    model = _tiny_model()
    input_tensor, mask = _tiny_input(num_cores=6, batch=1)
    mappings, _, _ = model(input_tensor, num_io=2, mask=mask, num_samples=1)
    mapping_lp, preempt_lp = model.get_log_prob_components(input_tensor, mask, 2, mappings, None)
    assert preempt_lp is None
    assert mapping_lp.shape == (1,)


def _grad_norm(params):
    return sum((p.grad.abs().sum().item() if p.grad is not None else 0.0) for p in params)


def test_fix_6_preemption_gradient_does_not_reach_backbone():
    model = _tiny_model()
    input_tensor, mask = _tiny_input(num_cores=6, batch=1)
    mappings, preemptions, _ = model(input_tensor, num_io=2, mask=mask, num_samples=1)

    mapping_lp, preempt_lp = model.get_log_prob_components(input_tensor, mask, 2, mappings, preemptions)

    model.zero_grad()
    preempt_lp.sum().backward()

    assert _grad_norm(model.encoder.parameters()) == 0.0
    assert _grad_norm(model.decoder.parameters()) == 0.0
    assert _grad_norm(model.attn.parameters()) == 0.0
    # The head itself, and io_embed, MUST receive gradient — otherwise
    # the head could never learn anything.
    assert _grad_norm(model.preemption_head.parameters()) > 0.0
    assert _grad_norm(model.io_embed.parameters()) > 0.0


def test_fix_6_mapping_gradient_does_reach_backbone():
    # The other half of the asymmetry: unlike preemption loss, mapping
    # loss is NOT detached, and should update the backbone normally.
    model = _tiny_model()
    input_tensor, mask = _tiny_input(num_cores=6, batch=1)
    mappings, preemptions, _ = model(input_tensor, num_io=2, mask=mask, num_samples=1)

    mapping_lp, _ = model.get_log_prob_components(input_tensor, mask, 2, mappings, preemptions)

    model.zero_grad()
    mapping_lp.sum().backward()

    assert _grad_norm(model.encoder.parameters()) > 0.0
    assert _grad_norm(model.decoder.parameters()) > 0.0
    assert _grad_norm(model.attn.parameters()) > 0.0


def test_fix_1_preemption_logits_depend_on_selected_io():
    # Same decoder output, different IO index -> different preemption
    # logits. If the head weren't IO-conditioned, these would be
    # identical (same Linear applied to the same feature vector).
    model = _tiny_model(hidden_dim=16)
    fake_output = torch.randn(1, 16)
    io_a = torch.tensor([0])
    io_b = torch.tensor([1])
    logits_a = model._preempt_logits(fake_output, io_a, detach_backbone=False)
    logits_b = model._preempt_logits(fake_output, io_b, detach_backbone=False)
    assert not torch.allclose(logits_a, logits_b)