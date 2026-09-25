"""
Command-line entry point for LMGA_IGSA (v2: two-phase attention/GAT
architecture).

Two modes:
  run      — a single full NN+GA co-training run (run.run_lmga)
  multirun — repeat `run` N times with the same ablation flags, for the
             mean/std numbers that go in a paper table

v1 had a third mode, `pretrain` (pretrain.pretrain_model), and two
warm-up-related flags (--disable_warmup, --head_warmup_epochs). Both
are removed here, not deprecated-and-ignored: v2 has no separate
backbone-pretraining step and no supervised warm-up phase at all (see
reinforce.py's and run.py's module docstrings for why those concepts
don't apply to this architecture — briefly, both existed to protect an
already-good pretrained backbone from a fresh, noisy, bolted-on head,
and v2 has no such asymmetry, since every component trains together
via REINFORCE from the first outer iteration). A script or doc that
still passes --mode pretrain, --disable_warmup, or
--head_warmup_epochs will get argparse's normal "unrecognized
argument" error rather than a silently-ignored flag.

Kept thin and free of algorithmic logic: this file only parses
arguments and dispatches. All behavior lives in the package.
"""

import argparse
import os
from time import time

import numpy as np
import torch

from .config import NUM_CONSTRUCTIVE, UPDATE_INTERVAL
from .run import run_lmga


def build_parser():
    parser = argparse.ArgumentParser(
        description="LMGA_IGSA: NN-guided GA for core-to-IO mapping (v2: attention + conflict-GAT)."
    )
    parser.add_argument("--mode", type=str, default="run", choices=["run", "multirun"])
    parser.add_argument("--num_cores", type=int, default=32)
    parser.add_argument("--num_io", type=int, default=2)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--constructive", type=int, default=NUM_CONSTRUCTIVE)
    parser.add_argument("--update_interval", type=int, default=UPDATE_INTERVAL)
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available.")

    # Ablation flags (paper Table 5 — now four rows, not five: no
    # warm-up ablation, since there's no warm-up to disable)
    parser.add_argument("--disable_preemption_head", action="store_true",
                         help="Use random bucket preemptions instead of the conflict-GAT "
                              "head. Isolates the mapping phase's contribution.")
    parser.add_argument("--disable_constructive", action="store_true",
                         help="Set constructive injection to 0. Isolates the constructive "
                              "operator's contribution.")
    parser.add_argument("--disable_inloop_updates", action="store_true",
                         help="Skip in-loop NN updates. Isolates the in-loop update's "
                              "contribution.")
    parser.add_argument("--continuous_preemptions", action="store_true",
                         help="Revert fix #2: continuous preemptions instead of "
                              "bucket-snapped. Finer GA resolution, but the preemption "
                              "head's REINFORCE actions no longer match rewards.")
    return parser


def _resolve_device(args):
    if args.cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ablation_tag(args):
    if args.disable_preemption_head:
        return "_noHead"
    if args.disable_constructive:
        return "_noConstruct"
    if args.disable_inloop_updates:
        return "_noInloop"
    if args.continuous_preemptions:
        return "_contPreempt"
    return "_full"


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    device = _resolve_device(args)

    if args.mode == "run":
        run_lmga(
            args.num_cores, args.num_io, device=device,
            disable_preemption_head=args.disable_preemption_head,
            disable_constructive=args.disable_constructive,
            disable_inloop_updates=args.disable_inloop_updates,
            continuous_preemptions=args.continuous_preemptions,
            num_constructive=args.constructive,
            update_interval=args.update_interval,
        )

    elif args.mode == "multirun":
        _run_multirun(args, device)


def _run_multirun(args, device):
    ablation_tag = _ablation_tag(args)

    print(f"=== MULTI-RUN ({ablation_tag.strip('_').upper()}): {args.num_runs} runs for "
          f"{args.num_cores}c {args.num_io}io ===")

    all_results = []
    for run_num in range(1, args.num_runs + 1):
        print(f"\n{'=' * 60}\nRUN {run_num}/{args.num_runs}\n{'=' * 60}")
        run_start = time()
        fitness = run_lmga(
            args.num_cores, args.num_io, device=device,
            save_model=False, output_suffix=f"_run{run_num}{ablation_tag}",
            disable_preemption_head=args.disable_preemption_head,
            disable_constructive=args.disable_constructive,
            disable_inloop_updates=args.disable_inloop_updates,
            continuous_preemptions=args.continuous_preemptions,
            num_constructive=args.constructive,
            update_interval=args.update_interval,
        )
        run_time = time() - run_start
        all_results.append((run_num, fitness, run_time))

    fitnesses = [r[1] for r in all_results]
    times = [r[2] for r in all_results]
    best_fitness = max(fitnesses)
    worst_fitness = min(fitnesses)
    mean_fitness = float(np.mean(fitnesses))
    std_fitness = float(np.std(fitnesses))
    best_run = all_results[fitnesses.index(best_fitness)][0]
    worst_run = all_results[fitnesses.index(worst_fitness)][0]
    best_hit_count = sum(1 for f in fitnesses if abs(f - best_fitness) / abs(best_fitness) < 0.001)

    print(f"\n{'=' * 60}\nMULTI-RUN SUMMARY ({ablation_tag.strip('_').upper()}): "
          f"{args.num_cores}c {args.num_io}io, {args.num_runs} runs")
    print(f"{'=' * 60}")
    print(f"Best:  {best_fitness:.2f} (run {best_run})")
    print(f"Worst: {worst_fitness:.2f} (run {worst_run})")
    print(f"Mean:  {mean_fitness:.2f} \u00b1 {std_fitness:.2f}")
    print(f"Best hit count: {best_hit_count}/{args.num_runs} (within 0.1%)")
    print(f"Mean Runtime: {np.mean(times):.2f}s \u00b1 {np.std(times):.2f}s")
    for run_num, fitness, rt in all_results:
        marker = " <-- BEST" if fitness == best_fitness else (" <-- WORST" if fitness == worst_fitness else "")
        print(f"  Run {run_num:2d}: fitness={fitness:.2f}, time={rt:.2f}s{marker}")

    summary_filename = f"multirun_{args.num_cores}cores_{args.num_io}io_{args.num_runs}runs_v2{ablation_tag}.txt"
    with open(summary_filename, "w") as f:
        f.write(f"# Algorithm: LMGA_IGSA v2 ({ablation_tag.strip('_')})\n")
        f.write(f"# Ablation: disable_preemption_head={args.disable_preemption_head}, "
                f"disable_constructive={args.disable_constructive}, "
                f"disable_inloop_updates={args.disable_inloop_updates}\n")
        f.write(f"# Num Runs: {args.num_runs}\n")
        f.write(f"# Best: {best_fitness:.2f} (run {best_run})\n")
        f.write(f"# Worst: {worst_fitness:.2f} (run {worst_run})\n")
        f.write(f"# Mean \u00b1 Std: {mean_fitness:.2f} \u00b1 {std_fitness:.2f}\n")
        f.write(f"# Best Hit Count: {best_hit_count}/{args.num_runs}\n")
        f.write(f"# Mean Runtime: {np.mean(times):.2f}s \u00b1 {np.std(times):.2f}s\n\n")
        f.write("run,fitness,runtime\n")
        for run_num, fitness, rt in all_results:
            f.write(f"{run_num},{fitness:.2f},{rt:.4f}\n")
    print(f"\nSummary saved to {summary_filename}")

    for run_num in range(1, args.num_runs + 1):
        run_file = f"top_100_data_{args.num_cores}cores_{args.num_io}io_run{run_num}{ablation_tag}.txt"
        if os.path.exists(run_file):
            os.remove(run_file)
    print("Cleaned up per-run output files.")


if __name__ == "__main__":
    main()