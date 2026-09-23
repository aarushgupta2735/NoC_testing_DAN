import random

from src.lmga_igsa.config import PREEMPTION_BUCKETS
from src.lmga_igsa.ga import Individual, Population, crossover, mutate, selection


def _ind(genes=None, preemptions=None, fitness=None):
    genes = genes or [0, 1, 0, 1]
    preemptions = preemptions or [0.5, 0.5, 0.5, 0.5]
    return Individual(genes, fitness=fitness, preemptions=preemptions)


def test_crossover_keeps_gene_length():
    p1 = _ind([0, 0, 0, 0], [0.2, 0.2, 0.2, 0.2])
    p2 = _ind([1, 1, 1, 1], [0.9, 0.9, 0.9, 0.9])
    child = crossover(p1, p2)
    assert len(child.genes) == 4
    assert len(child.preemptions) == 4


def test_crossover_child_genes_come_only_from_parents():
    p1 = _ind([0, 0, 0, 0], [0.2, 0.2, 0.2, 0.2])
    p2 = _ind([1, 1, 1, 1], [0.9, 0.9, 0.9, 0.9])
    child = crossover(p1, p2)
    assert all(g in (0, 1) for g in child.genes)
    assert all(p in (0.2, 0.9) for p in child.preemptions)


def test_crossover_without_preemptions_is_none():
    p1 = Individual([0, 1], preemptions=None)
    p2 = Individual([1, 0], preemptions=None)
    child = crossover(p1, p2)
    assert child.preemptions is None


def test_mutate_preserves_length():
    ind = _ind([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    mutated = mutate(ind, num_io=2, rate=1.0)  # force mutation on every gene
    assert len(mutated.genes) == 4
    assert len(mutated.preemptions) == 4


def test_mutate_forced_preemptions_stay_bucket_aligned():
    ind = _ind([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    mutated = mutate(ind, num_io=2, rate=1.0, continuous=False)
    assert all(p in PREEMPTION_BUCKETS for p in mutated.preemptions)


def test_mutate_zero_rate_can_leave_genes_unchanged():
    random.seed(0)
    ind = _ind([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])
    mutated = mutate(ind, num_io=2, rate=0.0, preemption_rate=0.0)
    assert mutated.genes == ind.genes
    assert mutated.preemptions == ind.preemptions


def test_population_sort_orders_by_fitness_descending():
    pop = Population([_ind(fitness=-5), _ind(fitness=-1), _ind(fitness=-10)])
    pop.sort()
    assert [i.fitness for i in pop.individuals] == [-1, -5, -10]
    assert pop.get_fittest().fitness == -1


def test_selection_returns_an_individual_from_population():
    inds = [_ind(fitness=i) for i in range(10)]
    pop = Population(inds)
    chosen = selection(pop, k=3)
    assert chosen in inds