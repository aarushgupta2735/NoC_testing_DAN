"""
Tests for data.py — focused on build_conflict_table, which is a fully
vectorized (NumPy broadcasting) reimplementation of check_path_conflict
evaluated pairwise. Vectorized boolean logic is easy to get subtly
wrong (broadcasting direction, min/max argument order, etc.), so this
file checks it against a straightforward scalar reference that calls
check_path_conflict directly, across many randomized instances rather
than one or two hand-picked cases.
"""

import numpy as np

from lmga_igsa.data import build_conflict_table, check_path_conflict


def _scalar_conflict_table(dir_np, mapping, io_array):
    """Reference implementation: nested loop, one check_path_conflict
    call per pair. Deliberately kept simple/obviously-correct rather
    than fast, as the ground truth build_conflict_table is checked
    against."""
    num_cores = len(mapping)
    table = np.zeros((num_cores, num_cores), dtype=bool)
    src_sink = []
    for core_id in range(num_cores):
        io_pair = io_array[int(mapping[core_id])]
        src_sink.append((io_pair[0] - 1, io_pair[1] - 1))
    for i in range(num_cores):
        src_i, sink_i = src_sink[i]
        for j in range(i + 1, num_cores):
            src_j, sink_j = src_sink[j]
            conflict = check_path_conflict(dir_np, i, src_i, sink_i, j, src_j, sink_j)
            table[i, j] = conflict
            table[j, i] = conflict
    return table


def _random_instance(rng, num_cores, num_io):
    dir_np = rng.integers(0, 6, size=(num_cores, 2)).astype(float)
    mapping = rng.integers(0, num_io, size=num_cores)
    io = []
    for k in range(num_io):
        s = (k * (num_cores // num_io)) + 1
        d = min(num_cores, s + 5)
        io.append([s, d])
    return dir_np, mapping, io


def test_build_conflict_table_matches_scalar_reference_across_many_instances():
    rng = np.random.default_rng(0)
    for _ in range(200):
        num_cores = int(rng.choice([4, 8, 12, 16, 20]))
        num_io = int(rng.choice([2, 3, 4]))
        dir_np, mapping, io = _random_instance(rng, num_cores, num_io)

        expected = _scalar_conflict_table(dir_np, mapping, io)
        actual = build_conflict_table(dir_np, mapping, io)

        assert np.array_equal(expected, actual)


def test_build_conflict_table_is_symmetric():
    rng = np.random.default_rng(1)
    dir_np, mapping, io = _random_instance(rng, 16, 4)
    table = build_conflict_table(dir_np, mapping, io)
    assert np.array_equal(table, table.T)


def test_build_conflict_table_diagonal_is_false():
    rng = np.random.default_rng(2)
    dir_np, mapping, io = _random_instance(rng, 10, 2)
    table = build_conflict_table(dir_np, mapping, io)
    assert not table.diagonal().any()


def test_build_conflict_table_can_include_cross_io_conflicts():
    # Regression guard for the specific claim that motivated the
    # conflict-graph design: conflicts are NOT restricted to cores
    # sharing an IO channel (check_path_conflict is checked against
    # every other core's schedule regardless of IO assignment). This
    # test doesn't force a cross-IO conflict to exist (that depends on
    # geometry), but across enough random instances at least one
    # should show up, confirming the table isn't silently only
    # capturing same-IO pairs.
    rng = np.random.default_rng(3)
    found_cross_io = False
    for _ in range(50):
        num_cores, num_io = 16, 4
        dir_np, mapping, io = _random_instance(rng, num_cores, num_io)
        table = build_conflict_table(dir_np, mapping, io)
        for i in range(num_cores):
            for j in range(i + 1, num_cores):
                if table[i, j] and mapping[i] != mapping[j]:
                    found_cross_io = True
                    break
            if found_cross_io:
                break
        if found_cross_io:
            break
    assert found_cross_io