import numpy as np

from src.lmga_igsa.data import prep_data
from src.lmga_igsa.simulator import simulate_single_mapping


def _tiny_problem(num_cores=6, num_io=2):
    data, cores, io, all_hops = prep_data(num_cores, num_io, test=True)
    dir_np = data[0].numpy()
    return dir_np, cores[0], io, all_hops


def test_simulate_returns_nonnegative_makespan():
    dir_np, core_config, io, all_hops = _tiny_problem()
    mapping = [i % len(io) for i in range(6)]
    preemptions = [0.5] * 6
    neg_makespan, schedule_log, preemptions_used = simulate_single_mapping(
        (mapping, dir_np, core_config, io, all_hops, preemptions)
    )
    assert -neg_makespan >= 0


def test_simulate_echoes_back_stored_preemptions_unchanged():
    dir_np, core_config, io, all_hops = _tiny_problem()
    mapping = [i % len(io) for i in range(6)]
    preemptions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    _, _, preemptions_used = simulate_single_mapping(
        (mapping, dir_np, core_config, io, all_hops, preemptions)
    )
    assert preemptions_used == preemptions


def test_simulate_fallback_draws_bucket_aligned_preemptions_when_none_given():
    # Fix #2's fallback: no stored preemptions -> draw from the bucket
    # set, not a continuous distribution.
    from lmga_igsa.config import PREEMPTION_BUCKETS

    dir_np, core_config, io, all_hops = _tiny_problem()
    mapping = [i % len(io) for i in range(6)]
    _, _, preemptions_used = simulate_single_mapping(
        (mapping, dir_np, core_config, io, all_hops, None)
    )
    assert all(p in PREEMPTION_BUCKETS for p in preemptions_used)


def test_simulate_higher_preemption_generally_changes_makespan():
    # Sanity: preemption value should actually influence the schedule
    # (not silently ignored) — low vs high preemption should not
    # always give identical makespans across many trials.
    dir_np, core_config, io, all_hops = _tiny_problem()
    mapping = [i % len(io) for i in range(6)]

    low = simulate_single_mapping((mapping, dir_np, core_config, io, all_hops, [0.2] * 6))[0]
    high = simulate_single_mapping((mapping, dir_np, core_config, io, all_hops, [0.9] * 6))[0]
    assert low != high


def test_simulate_with_precomputed_conflict_table_matches_inline():
    # v2 architecture: build_conflict_table + the 7-arg form let the
    # simulator look up conflict results instead of recomputing
    # check_path_conflict inline. Since check_path_conflict is a pure
    # function of a fixed mapping, this MUST be behavior-preserving —
    # same makespan, same preemptions_used, every time.
    import random
    from src.lmga_igsa.data import build_conflict_table

    num_cores, num_io = 16, 4
    dir_np, core_config, io, all_hops = _tiny_problem(num_cores, num_io)

    random.seed(42)
    for _ in range(10):
        mapping = [random.randint(0, num_io - 1) for _ in range(num_cores)]
        preemptions = [random.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) for _ in range(num_cores)]

        result_inline = simulate_single_mapping(
            (mapping, dir_np, core_config, io, all_hops, preemptions)
        )
        table = build_conflict_table(dir_np, mapping, io)
        result_table = simulate_single_mapping(
            (mapping, dir_np, core_config, io, all_hops, preemptions, table)
        )

        assert result_inline[0] == result_table[0]
        assert result_inline[2] == result_table[2]