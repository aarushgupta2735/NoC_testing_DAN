"""
Fix #2 (action/reward consistency): every preemption value that ever
reaches REINFORCE training must be bucket-aligned, so that
bucket-index actions match the rewards that were actually evaluated.
"""

from src.lmga_igsa.config import PREEMPTION_BUCKETS
from src.lmga_igsa.ga import (
    bucket_indices_to_preemptions,
    preemptions_to_bucket_indices,
    random_preemption_vector,
)


def test_bucket_snapped_round_trips_exactly():
    v = random_preemption_vector(50, continuous=False)
    idx = preemptions_to_bucket_indices(v)
    back = bucket_indices_to_preemptions(idx)
    assert v == back


def test_bucket_snapped_values_are_all_in_bucket_set():
    v = random_preemption_vector(50, continuous=False)
    assert all(p in PREEMPTION_BUCKETS for p in v)


def test_continuous_draws_generally_do_not_round_trip():
    # This is the bug fix #2 addresses, reproduced on demand: continuous
    # draws get silently altered by bucket snapping. We don't assert
    # every value changes (a continuous draw could land exactly on a
    # bucket by chance), but with 50 draws at least some should differ.
    v = random_preemption_vector(50, continuous=True)
    idx = preemptions_to_bucket_indices(v)
    back = bucket_indices_to_preemptions(idx)
    mismatches = sum(1 for a, b in zip(v, back) if a != b)
    assert mismatches > 0


def test_preemptions_to_bucket_indices_picks_nearest():
    # 0.21 -> nearest bucket is 0.20 (dist 0.01) not 0.30 (dist 0.09).
    # 0.39 -> nearest bucket is 0.40 (dist 0.01) not 0.30 (dist 0.09).
    # 0.85 -> nearest bucket is 0.80 or 0.90, both dist 0.05; the
    #         nearest-so-far scan picks whichever it hits first, 0.80.
    indices = preemptions_to_bucket_indices([0.21, 0.39, 0.85])
    assert indices == [0, 2, 6]  # buckets 0.20, 0.40, 0.80


def test_bucket_indices_to_preemptions_is_exact_lookup():
    assert bucket_indices_to_preemptions([0, 7]) == [0.20, 0.90]