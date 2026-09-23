"""
Genetic algorithm mechanics: Individual, Population, and the
selection/crossover/mutation operators, plus population-level fitness
evaluation via the simulator.

This module deliberately has NO dependency on model.py. An Individual
is just genes (a mapping) + preemptions + a fitness score; nothing here
knows a neural net exists. The two are only introduced to each other in
reinforce.py.
"""

import random

import numpy as np

from .config import PREEMPTION_BUCKETS
from .simulator import simulate_single_mapping


def preemptions_to_bucket_indices(preemptions):
    """Round continuous preemption values to nearest bucket indices."""
    indices = []
    for p in preemptions:
        best_idx = 0
        min_dist = float("inf")
        for i, b in enumerate(PREEMPTION_BUCKETS):
            dist = abs(p - b)
            if dist < min_dist:
                min_dist = dist
                best_idx = i
        indices.append(best_idx)
    return indices


def bucket_indices_to_preemptions(indices):
    """Convert bucket indices to actual preemption values."""
    return [PREEMPTION_BUCKETS[idx] for idx in indices]


def _continuous_preemption():
    """Continuous preemption draw in [0.20, 0.90]-ish (pre-#2 behavior)."""
    p = 0.2 + 0.7 * random.random()
    return max(0.05, min(0.99, p))


def random_preemption_vector(num_cores, continuous=False):
    """
    Fresh preemption vector.

    continuous=False (default, fix #2): draw from the discrete bucket
        set, so the bucket index used in REINFORCE matches the value
        that was actually evaluated.
    continuous=True (fix #2 REMOVED, for ablation): draw uniformly from
        a continuous range. Finer resolution for the GA search, but
        REINFORCE still rounds to buckets, so the action/reward
        mismatch this fix addresses returns.
    """
    if continuous:
        return [_continuous_preemption() for _ in range(num_cores)]
    return [random.choice(PREEMPTION_BUCKETS) for _ in range(num_cores)]


class Individual:
    def __init__(self, genes, fitness=None, preemptions=None):
        self.genes = genes
        self.fitness = fitness
        self.preemptions = preemptions


class Population:
    def __init__(self, individuals):
        self.individuals = individuals

    def sort(self):
        self.individuals.sort(
            key=lambda x: x.fitness if x.fitness is not None else float("-inf"),
            reverse=True,
        )

    def get_fittest(self):
        return self.individuals[0]


def crossover(parent1, parent2):
    """Single-segment crossover: a contiguous [start, end] slice of
    genes (and, if present, preemptions) is copied from parent2 into
    a copy of parent1."""
    genesN = len(parent1.genes)
    p1 = random.randint(0, genesN - 1)
    p2 = random.randint(0, genesN - 1)
    start, end = min(p1, p2), max(p1, p2)
    child_genes = parent1.genes[:]
    for i in range(start, end + 1):
        child_genes[i] = parent2.genes[i]

    child_preemptions = None
    if parent1.preemptions is not None and parent2.preemptions is not None:
        child_preemptions = parent1.preemptions[:]
        for i in range(start, end + 1):
            child_preemptions[i] = parent2.preemptions[i]

    return Individual(child_genes, preemptions=child_preemptions)


def mutate(ind, num_io, rate=0.05, preemption_rate=0.10, continuous=False):
    """
    Mutate genes and preemptions in place (on a copy).

    Fix #2: every fresh preemption draw here snaps to PREEMPTION_BUCKETS
    so the bucket index later used in REINFORCE matches the value that
    gets evaluated by the simulator. continuous=True reverts this (for
    the --continuous_preemptions ablation).
    """

    def _fresh():
        return _continuous_preemption() if continuous else random.choice(PREEMPTION_BUCKETS)

    new_genes = ind.genes[:]
    new_preemptions = ind.preemptions[:] if ind.preemptions is not None else None
    for i in range(len(new_genes)):
        if random.random() < rate:
            new_genes[i] = random.randint(0, num_io - 1)
            if new_preemptions is not None:
                new_preemptions[i] = _fresh()
        elif new_preemptions is not None and random.random() < preemption_rate:
            new_preemptions[i] = _fresh()
    return Individual(new_genes, preemptions=new_preemptions)


def selection(pop, k=3):
    """Tournament selection: sample k individuals, return the fittest."""
    candidates = random.sample(pop.individuals, k)
    return max(candidates, key=lambda x: x.fitness)


def objfunct_wrapper(mappings, dir_coords, cores, io_pairs, all_hops, pool=None, preemptions_list=None):
    """Evaluate a batch of raw mappings (e.g. straight from model.forward)
    against the simulator. Returns (penalties_tensor, None)."""
    import torch

    if hasattr(mappings, "cpu"):
        mappings_list = mappings.cpu().tolist()
    else:
        mappings_list = mappings
    dir_np = dir_coords.cpu().numpy() if hasattr(dir_coords, "cpu") else np.asarray(dir_coords)
    tasks = []
    core_config = cores[0]
    for i, m in enumerate(mappings_list):
        p = preemptions_list[i] if preemptions_list is not None else None
        tasks.append((m, dir_np, core_config, io_pairs, all_hops, p))
    if not tasks:
        return torch.tensor([], dtype=torch.float32), None
    if pool is not None:
        results = pool.map(simulate_single_mapping, tasks)
    else:
        results = [simulate_single_mapping(t) for t in tasks]
    penalties = [r[0] for r in results]
    return torch.tensor(penalties, dtype=torch.float32), None


def evaluate_population_parallel(pop, dir_coords, cores, io_pairs, all_hops, pool=None):
    """
    Evaluate every not-yet-scored Individual in pop against the
    simulator, in parallel if a pool is given. Fills in .fitness and
    .preemptions (the simulator echoes back the preemptions it used,
    which matter when an individual started with preemptions=None).
    Sorts the population in place afterward.
    """
    dir_np = dir_coords.cpu().numpy() if hasattr(dir_coords, "cpu") else np.asarray(dir_coords)
    tasks = []
    unevaluated_indices = []

    for idx, ind in enumerate(pop.individuals):
        if ind.fitness is not None:
            continue
        tasks.append((ind.genes, dir_np, cores[0], io_pairs, all_hops, ind.preemptions))
        unevaluated_indices.append(idx)

    if not tasks:
        return

    if pool is not None:
        results = pool.map(simulate_single_mapping, tasks)
    else:
        results = [simulate_single_mapping(t) for t in tasks]

    for i, idx in enumerate(unevaluated_indices):
        penalty, _, preemptions = results[i]
        pop.individuals[idx].fitness = penalty
        pop.individuals[idx].preemptions = preemptions
    pop.sort()