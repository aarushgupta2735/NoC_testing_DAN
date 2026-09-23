"""
The outer loop: run_lmga. Orchestrates the full NN+GA co-training
cycle — initial population (NN + random), GA evolution with
constructive injection and in-loop REINFORCE updates, and end-of-outer
either a supervised head warm-up (fix #3, first outer iter only) or a
full REINFORCE update.

This module wires together every other module in the package:
  data.py       -> problem instance
  model.py      -> PointerNet (sampling + training)
  ga.py         -> Individual/Population/operators/evaluation
  reinforce.py  -> the REINFORCE + warm-up updates

Ablation flags (see module-level ABLATION NOTES below each block) let
you isolate the contribution of the preemption head, the constructive
operator, the in-loop updates, and the warm-up independently — this is
what feeds Table 5 in the paper.
"""

import copy
import multiprocessing as mp
import os
import random
from time import time

import torch

from .config import (
    DROPOUT_P, EMA_DECAY, HEAD_WARMUP_EPOCHS, HIDDEN_DIM, IMPROVEMENT_THRESHOLD,
    INPUT_DIM, NUM_CONSTRUCTIVE, NUM_RNN_LAYERS, PREEMPTION_BUCKETS, UPDATE_INTERVAL,
)
from .data import prep_data
from .ga import (
    Individual, Population, bucket_indices_to_preemptions, crossover,
    evaluate_population_parallel, mutate, random_preemption_vector, selection,
)
from .model import PointerNet
from .reinforce import _do_model_update, warmup_preemption_head


def run_lmga(num_cores, num_io, ga_generations_per_iter=100, pop_size=100,
             stability_target=20, save_model=True, output_suffix="",
             device=None,
             # Ablation flags
             disable_preemption_head=False,
             disable_constructive=False,
             disable_inloop_updates=False,
             disable_warmup=False,
             continuous_preemptions=False,
             head_warmup_epochs=HEAD_WARMUP_EPOCHS,
             num_constructive=NUM_CONSTRUCTIVE,
             update_interval=UPDATE_INTERVAL):

    device = device or torch.device("cpu")
    start_time = time()

    print(f"--- LMGA-IGSA-FIXED Run: {num_cores}c {num_io}io ---")
    print(f"    Preemption buckets: {PREEMPTION_BUCKETS}")
    print(f"    Preemption draws: {'CONTINUOUS (#2 removed)' if continuous_preemptions else 'BUCKET-SNAPPED (#2 on)'}")
    print(f"    Constructive: {0 if disable_constructive else num_constructive}/gen")
    print(f"    In-loop updates: {'OFF' if disable_inloop_updates else f'every {update_interval} gens'}")
    print(f"    Head warm-up: {'OFF' if disable_warmup else f'{head_warmup_epochs} epochs at end of outer 1'}")
    print(f"    Preemption head: {'OFF (random)' if disable_preemption_head else 'ON (IO-conditioned)'}")

    data, cores, io, all_hops = prep_data(num_cores, num_io, test=True)
    data_tensor = data.permute(1, 0, 2).to(device)
    dir_coords = data_tensor[:, 0, :]
    num_io_local = len(io)
    mask = torch.ones(1, num_cores, device=device)

    model = PointerNet(INPUT_DIM, HIDDEN_DIM, NUM_RNN_LAYERS, DROPOUT_P, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Load pretrained backbone. Head + io_embed start random if no
    # checkpoint has warmed them up yet — that's fine, because
    # warmup_preemption_head trains them before they affect search.
    pretrained_path = f"lmga_model_{num_cores}cores_{num_io}io.pt"
    if os.path.exists(pretrained_path):
        try:
            state_dict = torch.load(pretrained_path, map_location=device)
            missing, _ = model.load_state_dict(state_dict, strict=False)
            new_keys = [m for m in missing if any(k in m for k in ["preemption", "io_embed"])]
            print(f"Loaded backbone from {pretrained_path}; "
                  f"{len(new_keys)} head/io_embed keys at random init (will be warmed up).")
        except Exception as e:
            print(f"Error loading model ({e}), starting fresh.")
    else:
        print("No pretrained model found. Starting from scratch.")

    best_global_fitness = float("-inf")
    best_global_mapping = None
    best_global_preemptions = None
    best_population_snapshot = []
    stability_counter = 0
    outer_loop_iter = 0

    ema_baseline = None

    num_workers = mp.cpu_count()
    pool = mp.Pool(processes=num_workers)

    # Skip warm-up bookkeeping entirely if warm-up is disabled or the
    # head itself is disabled (nothing to warm up).
    head_warmed_up = disable_warmup or disable_preemption_head

    try:
        while stability_counter < stability_target:
            outer_loop_iter += 1
            # Only the FIRST outer iter is treated as warm-up. During
            # warm-up: no constructive injection, no in-loop updates,
            # random bucket preemptions for NN-sampled individuals. The
            # head is then trained on the warm-up's top-K (fix #3)
            # before outer iter 2 begins.
            in_warmup = (outer_loop_iter == 1) and not head_warmed_up

            current_constructive = 0 if (disable_constructive or in_warmup) else num_constructive
            do_inloop = (not disable_inloop_updates) and (not in_warmup)

            # ====================================================
            # A. Initial population
            # ====================================================
            pop_list = []
            num_from_model = int(pop_size * 0.75)
            model.eval()
            with torch.no_grad():
                use_head_for_init = not (in_warmup or disable_preemption_head)
                sampled_mappings, sampled_preemptions, _ = model(
                    data_tensor[:, 0:1, :], num_io_local, mask,
                    num_samples=num_from_model,
                    use_preemption_head=use_head_for_init,
                )
                for i in range(num_from_model):
                    genes = sampled_mappings[i].tolist()
                    if use_head_for_init:
                        preemptions = bucket_indices_to_preemptions(sampled_preemptions[i].tolist())
                    else:
                        preemptions = random_preemption_vector(num_cores, continuous=continuous_preemptions)
                    pop_list.append(Individual(genes, preemptions=preemptions))

            num_random = pop_size - num_from_model
            for _ in range(num_random):
                genes = [random.randint(0, num_io_local - 1) for _ in range(num_cores)]
                preemptions = random_preemption_vector(num_cores, continuous=continuous_preemptions)
                pop_list.append(Individual(genes, preemptions=preemptions))

            population = Population(pop_list)
            evaluate_population_parallel(population, dir_coords, cores, io, all_hops, pool=pool)

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

                # Constructive operator (skipped during warm-up or when disabled)
                if current_constructive > 0:
                    model.eval()
                    with torch.no_grad():
                        nn_mappings, nn_preemptions, _ = model(
                            data_tensor[:, 0:1, :], num_io_local, mask,
                            num_samples=current_constructive,
                            use_preemption_head=not disable_preemption_head,
                        )
                        for i in range(current_constructive):
                            genes = nn_mappings[i].tolist()
                            if disable_preemption_head:
                                preemptions = random_preemption_vector(num_cores, continuous=continuous_preemptions)
                            else:
                                preemptions = bucket_indices_to_preemptions(nn_preemptions[i].tolist())
                            next_gen.append(Individual(genes, preemptions=preemptions))

                population.individuals = next_gen
                evaluate_population_parallel(population, dir_coords, cores, io, all_hops, pool=pool)

                # In-loop model update (skipped during warm-up or when disabled)
                if do_inloop and ((g + 1) % update_interval == 0):
                    ema_baseline = _do_model_update(
                        model, optimizer, population, data_tensor, mask,
                        num_io_local, ema_baseline, EMA_DECAY,
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
                wtag = " [warmup]" if in_warmup else ""
                print(f"[Iter {outer_loop_iter}] {tag}: {current_fitness:.2f}{pct}{wtag}")
                best_global_fitness = current_fitness
                best_global_mapping = current_best_ind.genes
                best_global_preemptions = current_best_ind.preemptions
                best_population_snapshot = copy.deepcopy(population.individuals)
                stability_counter = 0
            else:
                stability_counter += 1
                wtag = " [warmup]" if in_warmup else ""
                print(f"[Iter {outer_loop_iter}] Found {current_fitness:.2f} "
                      f"(Best: {best_global_fitness:.2f}) "
                      f"[stable: {stability_counter}/{stability_target}]{wtag}")

            # ====================================================
            # D. End-of-outer: head warm-up OR full REINFORCE update
            # ====================================================
            if in_warmup and not head_warmed_up and not disable_preemption_head:
                population.sort()
                top_k = max(5, len(population.individuals) // 3)
                top_samples = [
                    (ind.genes, ind.preemptions, ind.fitness)
                    for ind in population.individuals[:top_k]
                ]
                print(f"--- End of warm-up: training preemption head on top-{top_k} samples ---")
                warmup_preemption_head(
                    model, top_samples, data_tensor, mask, num_io_local,
                    epochs=head_warmup_epochs,
                )
                head_warmed_up = True
                print("--- Head warm-up complete; outer 2+ will use trained head ---")
                # Don't run the mapping REINFORCE update on this iter —
                # the warm-up GA pop is closer to random than to a
                # learned signal. Also reset stability so the warm-up
                # iter doesn't count against convergence budget.
                stability_counter = 0
            else:
                ema_baseline = _do_model_update(
                    model, optimizer, population, data_tensor, mask,
                    num_io_local, ema_baseline, EMA_DECAY,
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
        f.write(f"# Algorithm: LMGA-IGSA-FIXED\n")
        f.write(f"# Ablation flags: disable_preemption_head={disable_preemption_head}, "
                f"disable_constructive={disable_constructive}, "
                f"disable_inloop_updates={disable_inloop_updates}, "
                f"disable_warmup={disable_warmup}\n")
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
        torch.save(model.state_dict(), f"lmga_model_{num_cores}cores_{num_io}io.pt")
    return best_global_fitness