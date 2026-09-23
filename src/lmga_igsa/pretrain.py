"""
Pretrains the mapping backbone only. The preemption head and io_embed
are left at random init here — they get warmed up in run.run_lmga's
first outer iteration (fix #3), not here.

To guarantee zero gradient reaches the head during pretraining, this
uses use_preemption_head=False (uniform random bucket preemptions from
model.forward) and never calls get_log_prob_components with preemption
targets — so there's no preemption loss term at all in this file, not
even one that happens to isolate itself via detach.
"""

import multiprocessing as mp
from time import time

import torch

from .config import DROPOUT_P, HIDDEN_DIM, INPUT_DIM, NUM_RNN_LAYERS, PREEMPTION_BUCKETS
from .data import prep_data
from .ga import objfunct_wrapper
from .model import PointerNet


def pretrain_model(num_cores, num_io, epoch, lr=1e-4, device=None):
    device = device or torch.device("cpu")
    print(f"--- Pretraining (IGSA-FIXED, mapping head only): {num_cores}c {num_io}io ---")

    data, cores, io, all_hops = prep_data(num_cores, num_io, test=False)
    data_tensor = data.permute(1, 0, 2).to(device)
    num_io_local = len(io)
    mask = torch.ones(data_tensor.size(1), num_cores, device=device)

    model = PointerNet(INPUT_DIM, HIDDEN_DIM, NUM_RNN_LAYERS, DROPOUT_P, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_workers = mp.cpu_count()
    pool = mp.Pool(processes=num_workers)

    try:
        for e in range(epoch):
            model.train()
            optimizer.zero_grad()

            # use_preemption_head=False: NN samples mappings only;
            # preemptions come back as uniform-random bucket indices
            # that carry no gradient (see module docstring).
            mappings, _, log_probs_sum = model(
                data_tensor, num_io_local, mask, num_samples=1, use_preemption_head=False
            )

            preempt_values = [
                [_rand_bucket() for _ in range(num_cores)] for _ in range(mappings.shape[0])
            ]

            penalties, _ = objfunct_wrapper(
                mappings, data_tensor[:, 0, :], cores, io, all_hops,
                pool=pool, preemptions_list=preempt_values,
            )

            baseline = penalties.mean()
            advantage = penalties - baseline
            loss = -(log_probs_sum * advantage).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            print(f"[Pretrain] Epoch {e + 1}/{epoch} | Loss: {loss.item():.4f} | Avg Reward: {baseline.item():.2f}")
    finally:
        pool.close()
        pool.join()

    torch.save(model.state_dict(), f"lmga_model_{num_cores}cores_{num_io}io.pt")
    print("Pretraining complete (mapping backbone saved).")


def _rand_bucket():
    import random
    return random.choice(PREEMPTION_BUCKETS)