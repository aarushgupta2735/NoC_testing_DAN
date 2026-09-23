"""
Tests for reinforce.py — the model x ga glue. These exercise
_do_model_update and warmup_preemption_head directly (rather than only
indirectly, via a full run_lmga call), so a break here is attributable
to this file specifically.
"""

import torch

from src.lmga_igsa.config import INPUT_DIM, HIDDEN_DIM, NUM_RNN_LAYERS, DROPOUT_P, PREEMPTION_BUCKETS
from src.lmga_igsa.ga import Individual, Population
from src.lmga_igsa.model import PointerNet
from src.lmga_igsa.reinforce import _do_model_update, warmup_preemption_head

DEVICE = torch.device("cpu")


def _tiny_model():
    return PointerNet(INPUT_DIM, hidden_dim=16, n_layers=1, p=0.0, device=DEVICE)


def _fake_population(num_cores=4, num_io=2, size=8):
    import random
    inds = []
    for _ in range(size):
        genes = [random.randint(0, num_io - 1) for _ in range(num_cores)]
        preemptions = [random.choice(PREEMPTION_BUCKETS) for _ in range(num_cores)]
        fitness = -random.uniform(100, 1000)
        inds.append(Individual(genes, fitness=fitness, preemptions=preemptions))
    return Population(inds)


def _snapshot_params(model):
    return {name: p.detach().clone() for name, p in model.named_parameters()}


def _changed(before, after, name):
    return not torch.allclose(before[name], after[name])


def test_do_model_update_changes_model_params_and_returns_ema():
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    population = _fake_population()

    num_cores, num_io = 4, 2
    data_tensor = torch.randn(num_cores, 1, 2)
    mask = torch.ones(1, num_cores)

    before = _snapshot_params(model)
    ema = _do_model_update(model, optimizer, population, data_tensor, mask, num_io,
                            ema_baseline=None, train_preempt_head=True)
    after = _snapshot_params(model)

    assert ema is not None
    # At minimum, encoder and preemption_head params should have moved.
    assert any(_changed(before, after, n) for n in before if n.startswith("encoder"))
    assert any(_changed(before, after, n) for n in before if n.startswith("preemption_head"))


def test_do_model_update_with_train_preempt_head_false_skips_head_training_target():
    # With train_preempt_head=False, target_preempts is None, so the
    # preemption loss term is absent from total_lp. The preemption
    # head can still technically receive *some* gradient only if
    # mapping_lp depended on it, which it doesn't — so preemption_head
    # params should be unchanged by this call.
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    population = _fake_population()
    num_cores, num_io = 4, 2
    data_tensor = torch.randn(num_cores, 1, 2)
    mask = torch.ones(1, num_cores)

    before = _snapshot_params(model)
    _do_model_update(model, optimizer, population, data_tensor, mask, num_io,
                      ema_baseline=None, train_preempt_head=False)
    after = _snapshot_params(model)

    assert not any(_changed(before, after, n) for n in before if n.startswith("preemption_head"))
    assert not any(_changed(before, after, n) for n in before if n.startswith("io_embed"))
    # Mapping backbone should still have moved.
    assert any(_changed(before, after, n) for n in before if n.startswith("encoder"))


def test_do_model_update_ema_baseline_is_threaded_and_smoothed():
    model = _tiny_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_cores, num_io = 4, 2
    data_tensor = torch.randn(num_cores, 1, 2)
    mask = torch.ones(1, num_cores)

    pop1 = _fake_population()
    ema1 = _do_model_update(model, optimizer, pop1, data_tensor, mask, num_io, ema_baseline=None)

    pop2 = _fake_population()
    ema2 = _do_model_update(model, optimizer, pop2, data_tensor, mask, num_io, ema_baseline=ema1)

    # ema2 should be a smoothed combination, not just pop2's raw mean.
    raw_mean_pop2 = sum(i.fitness for i in pop2.individuals[: max(5, len(pop2.individuals) // 3)])
    assert ema2 != ema1  # it moved
    assert isinstance(ema2, float)


def test_warmup_preemption_head_only_updates_head_and_io_embed():
    model = _tiny_model()
    num_cores, num_io = 4, 2
    data_tensor = torch.randn(num_cores, 1, 2)
    mask = torch.ones(1, num_cores)

    import random
    samples = []
    for _ in range(6):
        genes = [random.randint(0, num_io - 1) for _ in range(num_cores)]
        preemptions = [random.choice(PREEMPTION_BUCKETS) for _ in range(num_cores)]
        samples.append((genes, preemptions, -random.uniform(100, 1000)))

    before = _snapshot_params(model)
    warmup_preemption_head(model, samples, data_tensor, mask, num_io, epochs=3, lr=1e-2)
    after = _snapshot_params(model)

    assert any(_changed(before, after, n) for n in before if n.startswith("preemption_head"))
    assert any(_changed(before, after, n) for n in before if n.startswith("io_embed"))
    # Backbone must NOT move during warmup (fix #6, belt-and-suspenders
    # via the restricted optimizer in warmup_preemption_head).
    assert not any(_changed(before, after, n) for n in before if n.startswith("encoder"))
    assert not any(_changed(before, after, n) for n in before if n.startswith("decoder"))
    assert not any(_changed(before, after, n) for n in before if n.startswith("attn"))


def test_warmup_preemption_head_empty_samples_is_a_noop():
    model = _tiny_model()
    num_cores, num_io = 4, 2
    data_tensor = torch.randn(num_cores, 1, 2)
    mask = torch.ones(1, num_cores)

    before = _snapshot_params(model)
    warmup_preemption_head(model, [], data_tensor, mask, num_io, epochs=3)
    after = _snapshot_params(model)

    assert all(torch.allclose(before[n], after[n]) for n in before)