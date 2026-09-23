"""
Problem instance definition: cores, IO pairs, chip topology, and the
geometric path-conflict test used by the simulator's scheduler.

This module has no knowledge of the GA or the neural net — it only
answers "what does this mapping problem look like."
"""

import math
import os

import networkx as nx
import numpy as np
import torch


class Core:
    def __init__(self, id, benchmark, core_no, patterns, scan, preemption):
        self.id = id
        self.benchmark = benchmark
        self.core_no = core_no
        self.patterns = patterns
        self.scan = scan
        self.preemption = preemption


def create_data(batch_size, num_cores, cols):
    """Grid (row, col) coordinates for each core, repeated batch_size times."""
    data = []
    dir = np.zeros(shape=[num_cores, 2])
    for i in range(num_cores):
        dir[i, 0] = i // cols
        dir[i, 1] = i % cols
    for _ in range(batch_size):
        data.append(dir)
    return torch.from_numpy(np.array(data).astype("float32"))


def prep_data(num_cores, num_io, test=True):
    """
    Load a problem instance: core configs, IO src/sink pairs, and the
    all-pairs hop-distance table over the chip's mesh topology.

    Falls back to synthetic dummy cores if the expected benchmark file
    is not found on disk (useful for tests / CI without real data).
    """
    filename = f"bm_{num_cores}cores.txt" if test else f"data_{num_cores}cores.txt"

    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Generating dummy core data.")
        lines = [f"{i} app 0 100 10" for i in range(num_cores)]
    else:
        with open(filename, "r") as file:
            lines = file.readlines()

    batch_size = max(1, len(lines) // num_cores)
    cores = [[] for _ in range(batch_size)]

    for i, line in enumerate(lines):
        if i >= batch_size * num_cores:
            break
        d = line.split()
        cores[i // num_cores].append(
            Core(int(d[0]), d[1], int(d[2]), int(d[3]), int(d[4]), -1)
        )

    cols = int(math.ceil(math.sqrt(num_cores)))
    # Hand-tuned mesh aspect ratios for specific core counts (kept
    # verbatim from the original — these match the benchmark layouts).
    _cols_overrides = {
        7: 3, 8: 4, 10: 4, 14: 7, 16: 4, 28: 4,
        32: 4, 40: 5, 48: 6, 56: 7, 64: 8,
    }
    cols = _cols_overrides.get(num_cores, cols)

    io = []
    if num_io == 2:
        io = [[1, 2], [num_cores // 2, num_cores]]
    else:
        for k in range(num_io):
            s = (k * (num_cores // num_io)) + 1
            d = min(num_cores, s + 5)
            io.append([s, d])

    graph = nx.Graph()
    for i in range(1, num_cores + 1):
        graph.add_node(i)
        if i - cols > 0:
            graph.add_edge(i, i - cols)
        if i + cols <= num_cores:
            graph.add_edge(i, i + cols)
        if i % cols != 0 and i + 1 <= num_cores:
            graph.add_edge(i, i + 1)
        if (i - 1) % cols != 0 and i - 1 > 0:
            graph.add_edge(i, i - 1)

    data = create_data(batch_size, num_cores, cols)
    all_hops = dict(nx.all_pairs_shortest_path_length(graph))

    return data, cores, io, all_hops


def check_path_conflict(dir_np, core1, src1, sink1, core2, src2, sink2):
    """
    Geometric test: do the (src->core->sink) routing paths of two
    cores' IO traffic share a physical link on the mesh? Used by the
    simulator to detect when two cores' subtasks cannot execute
    concurrently.
    """
    ys1 = dir_np[src1][0]; xs1 = dir_np[src1][1]
    yc1 = dir_np[core1][0]; xc1 = dir_np[core1][1]
    yk1 = dir_np[sink1][0]; xk1 = dir_np[sink1][1]
    ys2 = dir_np[src2][0]; xs2 = dir_np[src2][1]
    yc2 = dir_np[core2][0]; xc2 = dir_np[core2][1]
    yk2 = dir_np[sink2][0]; xk2 = dir_np[sink2][1]

    if (xs1 - xc1) * (xs2 - xc2) > 0:
        if not ((xc1 <= min(xc2, xs2) and xs1 <= min(xc2, xs2)) or (xc1 >= max(xc2, xs2) and xs1 >= max(xc2, xs2))) and ys1 == ys2:
            return True
    if (xs1 - xc1) * (xc2 - xk2) > 0:
        if not ((xc1 <= min(xc2, xk2) and xs1 <= min(xc2, xk2)) or (xc1 >= max(xc2, xk2) and xs1 >= max(xc2, xk2))) and ys1 == yc2:
            return True
    if (xc1 - xk1) * (xs2 - xc2) > 0:
        if not ((xc1 <= min(xc2, xs2) and xk1 <= min(xc2, xs2)) or (xc1 >= max(xc2, xs2) and xk1 >= max(xc2, xs2))) and yc1 == ys2:
            return True
    if (xc1 - xk1) * (xc2 - xk2) > 0:
        if not ((xc1 <= min(xc2, xk2) and xk1 <= min(xc2, xk2)) or (xc1 >= max(xc2, xk2) and xk1 >= max(xc2, xk2))) and yc1 == yc2:
            return True
    if (ys1 - yc1) * (ys2 - yc2) > 0:
        if not ((yc1 <= min(yc2, ys2) and ys1 <= min(yc2, ys2)) or (yc1 >= max(yc2, ys2) and ys1 >= max(yc2, ys2))) and xc1 == xc2:
            return True
    if (ys1 - yc1) * (yc2 - yk2) > 0:
        if not ((yc1 <= min(yc2, yk2) and ys1 <= min(yc2, yk2)) or (yc1 >= max(yc2, yk2) and ys1 >= max(yc2, yk2))) and xc1 == xk2:
            return True
    if (yc1 - yk1) * (ys2 - yc2) > 0:
        if not ((yc1 <= min(yc2, ys2) and yk1 <= min(yc2, ys2)) or (yc1 >= max(yc2, ys2) and yk1 >= max(yc2, ys2))) and xk1 == xc2:
            return True
    if (yc1 - yk1) * (yc2 - yk2) > 0:
        if not ((yc1 <= min(yc2, yk2) and yk1 <= min(yc2, yk2)) or (yc1 >= max(yc2, yk2) and yk1 >= max(yc2, yk2))) and xk1 == xk2:
            return True
    return False