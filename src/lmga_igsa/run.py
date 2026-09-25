"""
The outer loop: run_lmga. Orchestrates the full NN+GA co-training
cycle — initial population (NN + random), GA evolution with
constructive injection and in-loop REINFORCE updates, and an
end-of-outer REINFORCE update.

v2 change from v1: no warm-up phase (fix #3 is retired — see
reinforce.py's module docstring for why it doesn't apply to this
architecture). Every outer iteration, including the first, runs the
same way: sample from the model (both phases), evolve, update.

This module wires together every other module in the package:
  data.py       -> problem instance, core features, conflict tables
  model.py      -> PointerNetV2 (phase 1 mapping + phase 2 preemption)
  ga.py         -> Individual/Population/operators/evaluation
  reinforce.py  -> the REINFORCE update

Ablation flags let you isolate the contribution of the preemption
head, the constructive operator, and the in-loop updates
independently. --disable_warmup and --head_warmup_epochs are retained
in the CLI surface (cli.py) for backward-compatible invocation but are
no-ops here, since there is no warm-up phase to disable in v2 — see
cli.py for how this is surfaced to the user.
"""

import copy
import multiprocessing as mp
import os
import random
from time import time

import numpy as np
import torch

from .config import (
    EMA_DECAY, HIDDEN_DIM_V2, IMPROVEMENT_THRESHOLD, NUM_ATTN_HEADS_V2,
    NUM_CONSTRUCTIVE, PREEMPTION_BUCKETS, UPDATE_INTERVAL,
)
from .data import build_conflict_table, create_core_features_v2, create_data, prep_data
from .ga import (
    Individual, Population, bucket_indices_to_preemptions, crossover,
    evaluate_population_parallel, mutate, random_preemption_vector, selection,
)
from .model import PointerNetV2
from .reinforce import _do_model_update


def _sample_individuals(model, core_features, io_features, dir_np, io_pairs,
                         num_samples, use_preemption_head, continuous_preemptions,
                         num_cores):
    """
    Sample num_samples individuals from the model: phase 1 (mapping),
    then — unless use_preemption_head is False — build each sample's
    conflict table from its own sampled mapping and run phase 2.

    Returns a list of (genes, preemptions) tuples, not Individuals
    directly, so callers can decide fitness/id bookkeeping.
    """
    model.eval()
    with torch.no_grad():
        batch_core = core_features.repeat(num_samples, 1, 1)
        batch_io = io_features.repeat(num_samples, 1, 1)

        sampled_mapping, _, _ = model.sample_mapping(batch_core, batch_io)  # (num_samples, num_cores)

        if use_preemption_head:
            tables = [
                build_conflict_table(dir_np, sampled_mapping[i].numpy(), io_pairs)
                for i in range(num_samples)
            ]
            conflict_adj = torch.from_numpy(np.stack(tables).astype("float32"))
            sampled_preempt, _ = model.predict_preemption(batch_core, conflict_adj)

    results = []
    for i in range(num_samples):
        genes = sampled_mapping[i].tolist()
        if use_preemption_head:
            preemptions = bucket_indices_to_preemptions(sampled_preempt[i].tolist())
        else:
            preemptions = random_preemption_vector(num_cores, continuous=continuous_preemptions)
        results.append((genes, preemptions))
    return results


def run_lmga(num_cores, num_io, ga_generations_per_iter=100, pop_size=100,
             stability_target=20, save_model=True, output_suffix="",
             device=None,
             # Ablation flags
             disable_preemption_head=False,
             disable_constructive=False,
             disable_inloop_updates=False,
             continuous_preemptions=False,
             num_constructive=NUM_CONSTRUCTIVE,
             update_interval=UPDATE_INTERVAL):

    device = device or torch.device("cpu")
    start_time = time()

    print(f"--- LMGA-IGSA v2 Run: {num_cores}c {num_io}io ---")
    print(f"    Architecture: two-phase attention/GAT (no LSTM, no warm-up)")
    print(f"    Preemption buckets: {PREEMPTION_BUCKETS}")
    print(f"    Preemption draws: {'CONTINUOUS (#2 removed)' if continuous_preemptions else 'BUCKET-SNAPPED (#2 on)'}")
    print(f"    Constructive: {0 if disable_constructive else num_constructive}/gen")
    print(f"    In-loop updates: {'OFF' if disable_inloop_updates else f'every {update_interval} gens'}")
    print(f"    Preemption head: {'OFF (random)' if disable_preemption_head else 'ON (conflict-GAT + MLP)'}")

    position_data, cores, io, all_hops = prep_data(num_cores, num_io, test=True)
    core_feat_full = create_core_features_v2(position_data, cores)  # (batch, num_cores, 4)
    core_features = core_feat_full[0:1].to(device)                   # this run uses instance 0
    dir_np = position_data[0].numpy()

    io_arr = np.asarray(io, dtype="float32")  # (num_io, 2) — [src, sink], used as phase-1's raw IO features
    io_features = torch.from_numpy(io_arr).unsqueeze(0).to(device)   # (1, num_io, 2)
    num_io_local = len(io)

    model = PointerNetV2(hidden_dim=HIDDEN_DIM_V2, num_heads=NUM_ATTN_HEADS_V2, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpoint_path = f"lmga_v2_model_{num_cores}cores_{num_io}io.pt"
    if os.path.exists(checkpoint_path):
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded checkpoint from {checkpoint_path}.")
        except Exception as e:
            print(f"Error loading checkpoint ({e}), starting fresh.")
    else:
        print("No checkpoint found. Starting from scratch (no warm-up, no separate pretraining).")

    best_global_fitness = float("-inf")
    best_global_mapping = None
    best_global_preemptions = None
    best_population_snapshot = []
    stability_counter = 0
    outer_loop_iter = 0

    ema_baseline = None

    num_workers = mp.cpu_count()
    pool = mp.Pool(processes=num_workers)

    try:
        while stability_counter < stability_target:
            outer_loop_iter += 1

            current_constructive = 0 if disable_constructive else num_constructive
            do_inloop = not disable_inloop_updates

            # ====================================================
            # A. Initial population
            # ====================================================
            pop_list = []
            num_from_model = int(pop_size * 0.75)

            sampled = _sample_individuals(
                model, core_features, io_features, dir_np, io,
                num_samples=num_from_model,
                use_preemption_head=not disable_preemption_head,
                continuous_preemptions=continuous_preemptions,
                num_cores=num_cores,
            )
            for genes, preemptions in sampled:
                pop_list.append(Individual(genes, preemptions=preemptions))

            num_random = pop_size - num_from_model
            for _ in range(num_random):
                genes = [random.randint(0, num_io_local - 1) for _ in range(num_cores)]
                preemptions = random_preemption_vector(num_cores, continuous=continuous_preemptions)
                pop_list.append(Individual(genes, preemptions=preemptions))

            population = Population(pop_list)
            evaluate_population_parallel(population, position_data[0], cores, io, all_hops, pool=pool)

            # ====================================================
            # B. GA evolution loop
            # ====================================================
            for g in range(ga_generations_per_iter):
                next_gen = population.individuals[:2]  # elitism

                num_ga_children = pop_size - 2 - current_constructive
                while len(next_gen) < 2 + num_ga_children:
                    p1 = selection(population)
                    p2 = selection(population)
                    child = crossover(p1, p2)
                    child = mutate(child, num_io_local, rate=0.05, continuous=continuous_preemptions)
                    next_gen.append(child)

                if current_constructive > 0:
                    sampled = _sample_individuals(
                        model, core_features, io_features, dir_np, io,
                        num_samples=current_constructive,
                        use_preemption_head=not disable_preemption_head,
                        continuous_preemptions=continuous_preemptions,
                        num_cores=num_cores,
                    )
                    for genes, preemptions in sampled:
                        next_gen.append(Individual(genes, preemptions=preemptions))

                population.individuals = next_gen
                evaluate_population_parallel(population, position_data[0], cores, io, all_hops, pool=pool)

                if do_inloop and ((g + 1) % update_interval == 0):
                    ema_baseline = _do_model_update(
                        model, optimizer, population, core_features, io_features,
                        dir_np, io, ema_baseline, EMA_DECAY,
                        train_preempt_head=not disable_preemption_head,
                    )

            # ====================================================
            # C. Convergence accounting
            # ====================================================
            current_best_ind = population.get_fittest()
            current_fitness = current_best_ind.fitness

            if best_global_fitness == float("-inf"):
                is_real_improvement = True
                relative_improvement = 0
            else:
                relative_improvement = (current_fitness - best_global_fitness) / abs(best_global_fitness)
                is_real_improvement = relative_improvement > IMPROVEMENT_THRESHOLD

            if is_real_improvement and current_fitness > best_global_fitness:
                tag = "FIRST BEST" if best_global_fitness == float("-inf") else "NEW BEST"
                pct = f" ({relative_improvement * 100:+.2f}%)" if best_global_fitness != float("-inf") else ""
                print(f"[Iter {outer_loop_iter}] {tag}: {current_fitness:.2f}{pct}")
                best_global_fitness = current_fitness
                best_global_mapping = current_best_ind.genes
                best_global_preemptions = current_best_ind.preemptions
                best_population_snapshot = copy.deepcopy(population.individuals)
                stability_counter = 0
            else:
                stability_counter += 1
                print(f"[Iter {outer_loop_iter}] Found {current_fitness:.2f} "
                      f"(Best: {best_global_fitness:.2f}) "
                      f"[stable: {stability_counter}/{stability_target}]")

            # ====================================================
            # D. End-of-outer REINFORCE update
            # ====================================================
            ema_baseline = _do_model_update(
                model, optimizer, population, core_features, io_features,
                dir_np, io, ema_baseline, EMA_DECAY,
                train_preempt_head=not disable_preemption_head,
            )

    finally:
        pool.close()
        pool.join()

    total_time = time() - start_time

    print(f"\n--- Convergence Reached ---")
    print(f"Final Best Fitness: {best_global_fitness}")
    print(f"Best Mapping: {best_global_mapping}")
    print(f"Best Preemptions: {best_global_preemptions}")
    print(f"Total Execution Time: {total_time:.2f} seconds")

    output_filename = f"top_100_data_{num_cores}cores_{num_io}io{output_suffix}.txt"
    if best_population_snapshot:
        best_population_snapshot.sort(key=lambda x: x.fitness, reverse=True)
        top_100 = best_population_snapshot[:100]
    else:
        population.sort()
        top_100 = population.individuals[:100]

    with open(output_filename, "w") as f:
        f.write(f"# Algorithm: LMGA-IGSA v2 (two-phase attention/GAT)\n")
        f.write(f"# Ablation flags: disable_preemption_head={disable_preemption_head}, "
                f"disable_constructive={disable_constructive}, "
                f"disable_inloop_updates={disable_inloop_updates}\n")
        f.write(f"# Preemption Buckets: {PREEMPTION_BUCKETS}\n")
        f.write(f"# Overall Best Fitness: {best_global_fitness}\n")
        f.write(f"# Total Execution Time: {total_time:.4f} seconds\n")
        f.write(f"# Overall Best Mapping: {best_global_mapping}\n")
        f.write(f"# Overall Best Preemptions: {best_global_preemptions}\n\n")
        for ind in top_100:
            f.write(f"{ind.genes}\n")
            f.write(f"{ind.preemptions}\n")
    print(f"Top 100 individuals saved to {output_filename}")

    if save_model:
        torch.save(model.state_dict(), checkpoint_path)
    return best_global_fitness