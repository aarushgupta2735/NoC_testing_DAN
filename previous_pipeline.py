"""
LMGA-IGSA-FIXED
===============

Patched version of lmga_igsa.py that fixes the four correctness issues
needed for honest paper publication. The IGSA framework (NN + GA hybrid
with constructive operator and in-loop updates) is preserved. Only the
broken parts of the preemption head and its training loop are corrected.

FIXED ISSUES (matched to the analysis labels):

  #1  IO-conditioned preemption head.
      The head now takes cat(decoder_output, io_embedding(selected_io_t))
      after the IO is sampled, so it can represent P(preempt | core, IO)
      instead of the wrong marginal P(preempt | core).

  #2  Action/reward consistency via bucket-snapped preemptions.
      Every fresh preemption draw — random initial pop, simulator
      fallback, and gene-driven or independent mutation — now samples
      from PREEMPTION_BUCKETS directly. The fitness reward and the
      bucket-index action used in REINFORCE now refer to the same
      preemption value.

  #3  Head warm-up via supervised imitation.
      Outer iter 1 runs as a warm-up: no constructive injection, no
      in-loop updates. After its GA loop, the head is trained via NLL
      on the top-K's (genes, bucket-index) targets for HEAD_WARMUP_EPOCHS
      epochs (backbone frozen). From outer iter 2 onwards, the head is
      in-distribution before its first constructive sample is used.

  #6  Backbone gradient isolation via output.detach().
      The preemption head receives output.detach() as its feature
      input. Gradients from the preemption loss can update io_embed
      and the head, but cannot flow back into encoder/decoder/attn.
      The pretrained mapping backbone is now protected from
      contamination by noisy preemption gradients.

ABLATION FLAGS for paper Table 5 (the "what part of IGSA matters?"
experiment):

  --disable_preemption_head : preemption head's outputs are ignored;
                              random bucket preemptions used instead.
                              Isolates the mapping head's contribution.
  --disable_constructive    : NUM_CONSTRUCTIVE = 0.
                              Isolates the constructive operator's
                              contribution.
  --disable_inloop_updates  : in-loop NN updates are skipped; only the
                              end-of-outer update fires.
                              Isolates the in-loop update's contribution.
  --disable_warmup          : skip the supervised head warm-up.
                              Reproduces the original broken behavior.
                              Useful for "is the warm-up actually needed?"

Run the full IGSA-FIXED with no flags. Run each ablation with one flag
at a time. Five rows for Table 5: full / no head / no constructive /
no in-loop / no warmup.
"""

import argparse
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from queue import PriorityQueue
import networkx as nx
import copy
import os
import multiprocessing as mp
from time import time


# ==========================================
# SECTION 1: CONFIGURATION
# ==========================================

PREEMPTION_BUCKETS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
NUM_PREEMPTION_BUCKETS = len(PREEMPTION_BUCKETS)

NUM_CONSTRUCTIVE = 15
UPDATE_INTERVAL = 10

# Head-warmup defaults
HEAD_WARMUP_EPOCHS = 30          # supervised NLL epochs on warm-up top-K
HEAD_WARMUP_LR = 3e-4            # higher LR for the fresh head

# Embedding table size for IO conditioning (#1 fix)
MAX_NUM_IO = 16
IO_EMBED_DIM = 16


def preemptions_to_bucket_indices(preemptions):
    """Round continuous preemption values to nearest bucket indices."""
    indices = []
    for p in preemptions:
        best_idx = 0
        min_dist = float('inf')
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
    """Continuous preemption draw in [0.05, 0.99] (the pre-#2 behavior,
    same formula as the seed/GA simulator)."""
    p = 0.2 + 0.7 * random.random()
    return max(0.05, min(0.99, p))


def random_preemption_vector(num_cores, continuous=False):
    """Fresh preemption vector.
    continuous=False (default, #2 fix): draw from the discrete bucket set,
        so the bucket index used in REINFORCE matches the value evaluated.
    continuous=True (#2 REMOVED): draw uniformly from [0.20, 0.90] like the
        seed/GA version. Finer resolution for the GA, but REINFORCE still
        rounds to buckets, so the action/reward mismatch returns."""
    if continuous:
        return [_continuous_preemption() for _ in range(num_cores)]
    return [random.choice(PREEMPTION_BUCKETS) for _ in range(num_cores)]


class Core():
    def __init__(self, id, benchmark, core_no, patterns, scan, preemption):
        self.id = id
        self.benchmark = benchmark
        self.core_no = core_no
        self.patterns = patterns
        self.scan = scan
        self.preemption = preemption


def create_data(batch_size, num_cores, cols):
    data = []
    dir = np.zeros(shape=[num_cores, 2])
    for i in range(num_cores):
        dir[i, 0] = i // cols
        dir[i, 1] = i % cols
    for _ in range(batch_size):
        data.append(dir)
    return torch.from_numpy(np.array(data).astype('float32'))


def prep_data(num_cores, num_io, test=True):
    filename = f"bm_{num_cores}cores.txt" if test else f"data_{num_cores}cores.txt"

    if not os.path.exists(filename):
        print(f"Warning: {filename} not found. Generating dummy core data.")
        lines = [f"{i} app 0 100 10" for i in range(num_cores)]
    else:
        with open(filename, 'r') as file:
            lines = file.readlines()

    batch_size = max(1, len(lines) // num_cores)
    cores = [[] for _ in range(batch_size)]

    for i, line in enumerate(lines):
        if i >= batch_size * num_cores:
            break
        d = line.split()
        cores[i // num_cores].append(Core(int(d[0]), d[1], int(d[2]), int(d[3]), int(d[4]), -1))

    cols = int(math.ceil(math.sqrt(num_cores)))
    if num_cores == 7: cols = 3
    elif num_cores == 8: cols = 4
    elif num_cores == 10: cols = 4
    elif num_cores == 14: cols = 7
    elif num_cores == 16: cols = 4
    elif num_cores == 28: cols = 4
    elif num_cores == 32: cols = 4
    elif num_cores == 40: cols = 5
    elif num_cores == 48: cols = 6
    elif num_cores == 56: cols = 7
    elif num_cores == 64: cols = 8

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
        if i - cols > 0: graph.add_edge(i, i - cols)
        if i + cols <= num_cores: graph.add_edge(i, i + cols)
        if i % cols != 0 and i + 1 <= num_cores: graph.add_edge(i, i + 1)
        if (i - 1) % cols != 0 and i - 1 > 0: graph.add_edge(i, i - 1)

    data = create_data(batch_size, num_cores, cols)
    all_hops = dict(nx.all_pairs_shortest_path_length(graph))

    return data, cores, io, all_hops


def check_path_conflict(dir_np, core1, src1, sink1, core2, src2, sink2):
    ys1 = dir_np[src1][0]; xs1 = dir_np[src1][1]
    yc1 = dir_np[core1][0]; xc1 = dir_np[core1][1]
    yk1 = dir_np[sink1][0]; xk1 = dir_np[sink1][1]
    ys2 = dir_np[src2][0]; xs2 = dir_np[src2][1]
    yc2 = dir_np[core2][0]; xc2 = dir_np[core2][1]
    yk2 = dir_np[sink2][0]; xk2 = dir_np[sink2][1]

    if (xs1 - xc1) * (xs2 - xc2) > 0:
        if not ((xc1 <= min(xc2, xs2) and xs1 <= min(xc2, xs2)) or (xc1 >= max(xc2, xs2) and xs1 >= max(xc2, xs2))) and ys1 == ys2: return True
    if (xs1 - xc1) * (xc2 - xk2) > 0:
        if not ((xc1 <= min(xc2, xk2) and xs1 <= min(xc2, xk2)) or (xc1 >= max(xc2, xk2) and xs1 >= max(xc2, xk2))) and ys1 == yc2: return True
    if (xc1 - xk1) * (xs2 - xc2) > 0:
        if not ((xc1 <= min(xc2, xs2) and xk1 <= min(xc2, xs2)) or (xc1 >= max(xc2, xs2) and xk1 >= max(xc2, xs2))) and yc1 == ys2: return True
    if (xc1 - xk1) * (xc2 - xk2) > 0:
        if not ((xc1 <= min(xc2, xk2) and xk1 <= min(xc2, xk2)) or (xc1 >= max(xc2, xk2) and xk1 >= max(xc2, xk2))) and yc1 == yc2: return True
    if (ys1 - yc1) * (ys2 - yc2) > 0:
        if not ((yc1 <= min(yc2, ys2) and ys1 <= min(yc2, ys2)) or (yc1 >= max(yc2, ys2) and ys1 >= max(yc2, ys2))) and xc1 == xc2: return True
    if (ys1 - yc1) * (yc2 - yk2) > 0:
        if not ((yc1 <= min(yc2, yk2) and ys1 <= min(yc2, yk2)) or (yc1 >= max(yc2, yk2) and ys1 >= max(yc2, yk2))) and xc1 == xk2: return True
    if (yc1 - yk1) * (ys2 - yc2) > 0:
        if not ((yc1 <= min(yc2, ys2) and yk1 <= min(yc2, ys2)) or (yc1 >= max(yc2, ys2) and yk1 >= max(yc2, ys2))) and xk1 == xc2: return True
    if (yc1 - yk1) * (yc2 - yk2) > 0:
        if not ((yc1 <= min(yc2, yk2) and yk1 <= min(yc2, yk2)) or (yc1 >= max(yc2, yk2) and yk1 >= max(yc2, yk2))) and xk1 == xk2: return True
    return False


# ==========================================
# SECTION 2: SIMULATOR
# ==========================================
#
# CHANGE vs original IGSA simulator: when stored_preemptions is None,
# the fallback draws from PREEMPTION_BUCKETS instead of uniform
# continuous in [0.20, 0.90]. This guarantees that any preemption value
# present in the population is a bucket value, so the bucket index used
# during REINFORCE training corresponds exactly to the value that
# produced the reward. (#2 fix)

def _insert_by_finish(lst, ev):
    f = ev[2]; lo, hi = 0, len(lst)
    while lo < hi:
        mid = (lo + hi) // 2
        if lst[mid][2] < f: lo = mid + 1
        else: hi = mid
    lst.insert(lo, ev)


def _merge_sorted(intervals):
    if not intervals: return []
    merged = []; ps, pf = intervals[0]
    for s, f in intervals[1:]:
        if s <= pf: pf = max(pf, f)
        else: merged.append([ps, pf]); ps, pf = s, f
    merged.append([ps, pf])
    return merged


def simulate_single_mapping(args):
    if len(args) == 6:
        mapping, dir_np, core_config, ioArray, all_hops, stored_preemptions = args
    else:
        mapping, dir_np, core_config, ioArray, all_hops = args
        stored_preemptions = None

    num_cores = len(mapping)
    numIo = len(ioArray)

    if stored_preemptions is not None and len(stored_preemptions) == num_cores:
        preemptions_map = list(stored_preemptions)
    else:
        # #2 fix: draw from bucket set so reward matches a discrete action.
        # NOTE: in run_lmga every Individual always carries preemptions, so
        # this fallback is not on the critical path; it only fires if an
        # individual is ever evaluated without stored preemptions.
        preemptions_map = [random.choice(PREEMPTION_BUCKETS) for _ in range(num_cores)]

    ioAssignments = [[] for _ in range(numIo)]
    for coreId, ioIndex in enumerate(mapping):
        ioAssignments[int(ioIndex)].append(coreId)

    allSubtaskTimes = [[] for _ in range(numIo)]

    for ioIndex in range(numIo):
        sjfQueue = []
        for coreId in ioAssignments[ioIndex]:
            io_pair = ioArray[int(mapping[coreId])]
            src, sink = io_pair[0], io_pair[1]
            currentCore = core_config[coreId]
            preemption = preemptions_map[coreId]

            hsc = all_hops[src][coreId + 1]
            hck = all_hops[coreId + 1][sink]
            l, p_count = currentCore.scan, currentCore.patterns

            subtaskDurations = []
            patternsRemaining = p_count

            while patternsRemaining > 10:
                patternsToProcess = math.floor(patternsRemaining * preemption)
                if patternsToProcess == 0: patternsToProcess = 1
                duration = (max(hsc, hck) + l) * patternsToProcess + (min(hsc, hck) + l - 1)
                subtaskDurations.append(duration)
                patternsRemaining -= patternsToProcess

            if patternsRemaining > 0:
                duration = (max(hsc, hck) + l) * patternsRemaining + (min(hsc, hck) + l - 1)
                subtaskDurations.append(duration)

            allSubtaskTimes[ioIndex].append(subtaskDurations)
            sjfQueue.append((sum(subtaskDurations), coreId))

        sjfQueue.sort()
        originalIoAssignments = list(ioAssignments[ioIndex])
        originalSubtaskTimes = list(allSubtaskTimes[ioIndex])
        reorderMap = {coreId: i for i, coreId in enumerate(originalIoAssignments)}
        ioAssignments[ioIndex] = [coreId for _, coreId in sjfQueue]
        allSubtaskTimes[ioIndex] = [originalSubtaskTimes[reorderMap[coreId]] for _, coreId in sjfQueue]

    startTime = [0] * num_cores
    globalTime = [0] * numIo
    isAllocated = [False] * num_cores
    scheduleLog = []
    passiveQueue = PriorityQueue()
    coreSubtaskIndex = {c: 0 for c in range(num_cores)}

    for coresInIo in ioAssignments:
        for c in coresInIo:
            passiveQueue.put((0, c))

    while not all(isAllocated):
        processQueue = []
        while not passiveQueue.empty():
            st, c = passiveQueue.get()
            if not isAllocated[c]: processQueue.append((st, c))
        processQueue.sort()
        made_progress = False

        for _, coreId in processQueue:
            ioIndex = int(mapping[coreId])
            src1, sink1 = ioArray[ioIndex][0] - 1, ioArray[ioIndex][1] - 1
            isConflict = False

            for core2, s2, f2 in scheduleLog:
                if startTime[coreId] < f2:
                    ioIndex2 = int(mapping[core2])
                    src2, sink2 = ioArray[ioIndex2][0] - 1, ioArray[ioIndex2][1] - 1
                    if check_path_conflict(dir_np, coreId, src1, sink1, core2, src2, sink2):
                        startTime[coreId] = max(startTime[coreId], f2)
                        isConflict = True

            pos = ioAssignments[ioIndex].index(coreId)
            subtasks = allSubtaskTimes[ioIndex][pos]
            subIdx = coreSubtaskIndex[coreId]

            if subIdx >= len(subtasks):
                isAllocated[coreId] = True
                continue

            duration = subtasks[subIdx]

            if isConflict:
                intervals = []
                for core2, s2, f2 in scheduleLog:
                    if core2 == coreId: continue
                    ioIndex2 = int(mapping[core2])
                    src2, sink2 = ioArray[ioIndex2][0] - 1, ioArray[ioIndex2][1] - 1
                    if check_path_conflict(dir_np, coreId, src1, sink1, core2, src2, sink2):
                        if s2 < startTime[coreId]:
                            intervals.append([s2, f2])

                if intervals:
                    intervals.sort(key=lambda x: x[0])
                    merged = _merge_sorted(intervals)
                else:
                    merged = []

                gaps = []
                if not merged:
                    if duration <= startTime[coreId]: gaps.append((0, startTime[coreId]))
                else:
                    if merged[0][0] > 0: gaps.append((0, merged[0][0]))
                    for i in range(len(merged) - 1):
                        if merged[i + 1][0] > merged[i][1]:
                            gaps.append((merged[i][1], merged[i + 1][0]))
                    if startTime[coreId] > merged[-1][1]:
                        gaps.append((merged[-1][1], startTime[coreId]))

                best_gap = None
                best_slack = None
                for gs, ge in gaps:
                    avail = ge - gs
                    if avail >= duration:
                        slack = avail - duration
                        if best_gap is None or slack < best_slack:
                            best_gap = (gs, gs + duration)
                            best_slack = slack

                if best_gap:
                    s, f = best_gap
                    _insert_by_finish(scheduleLog, (coreId, s, f))
                    globalTime[ioIndex] = max(globalTime[ioIndex], f)
                    coreSubtaskIndex[coreId] += 1
                    if coreSubtaskIndex[coreId] < len(subtasks):
                        startTime[coreId] = f
                        passiveQueue.put((f, coreId))
                    else:
                        isAllocated[coreId] = True
                    for other in ioAssignments[ioIndex]:
                        if not isAllocated[other]: startTime[other] = max(startTime[other], f)
                    made_progress = True
                else:
                    passiveQueue.put((startTime[coreId], coreId))
            else:
                f = startTime[coreId] + duration
                _insert_by_finish(scheduleLog, (coreId, startTime[coreId], f))
                globalTime[ioIndex] = max(globalTime[ioIndex], f)
                coreSubtaskIndex[coreId] += 1
                if coreSubtaskIndex[coreId] < len(subtasks):
                    startTime[coreId] = f
                    passiveQueue.put((f, coreId))
                else:
                    isAllocated[coreId] = True
                for other in ioAssignments[ioIndex]:
                    if not isAllocated[other]: startTime[other] = max(startTime[other], f)
                made_progress = True

        if not made_progress and passiveQueue.empty():
            break

    makespan = max(globalTime) if globalTime else 0
    return -makespan, scheduleLog, preemptions_map


# ==========================================
# SECTION 3: NEURAL NETWORK
# ==========================================
#
# CHANGES vs original IGSA PointerNet:
#   #1: self.io_embed (nn.Embedding) added.
#       preemption_head input dim: hidden_dim + IO_EMBED_DIM.
#       In forward and get_log_prob_components, the head sees
#       cat(decoder_output, io_embed(selected_io_t)) — IO-conditioned.
#   #6: output.detach() is fed into the preemption head.
#       Gradients from preemption loss cannot reach encoder/decoder/attn.
#       Backbone is protected from preemption noise.

device = torch.device('cpu')


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, p):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=p).to(device)

    def forward(self, input, mask):
        lengths = mask.sum(dim=1).cpu()
        packed_inputs = pack_padded_sequence(input, lengths, enforce_sorted=False).to(device)
        packed_outputs, (hidden, cell) = self.rnn(packed_inputs)
        output, _ = pad_packed_sequence(packed_outputs)
        return output, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, p):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=p).to(device)

    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0).to(device)
        output, (hidden, cell) = self.rnn(input, (hidden, cell))
        output = output.squeeze(0)
        return output, (hidden, cell)


class Attention(nn.Module):
    def __init__(self, enc_dim, dec_dim, logit_clipping=True, clip_value=10):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(enc_dim + dec_dim, dec_dim), nn.Tanh()).to(device)
        self.v = nn.Linear(dec_dim, 1, bias=False).to(device)
        self.logit_clipping = logit_clipping
        self.clip_value = clip_value

    def forward(self, decoder_output, encoder_outputs, num_io, mask):
        seq_len = encoder_outputs.shape[0]
        decoder_output = decoder_output.unsqueeze(1).repeat(1, seq_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = self.attn(torch.cat((decoder_output, encoder_outputs), dim=2))
        attention = self.v(energy).squeeze(2)
        if self.logit_clipping:
            attention = self.clip_value * torch.tanh(attention)
        attention = attention.masked_fill(mask == 0, float('-inf'))
        attentions, _ = torch.topk(attention, k=num_io, dim=1)
        return attentions


class PointerNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, p, device,
                 num_preemption_buckets=NUM_PREEMPTION_BUCKETS,
                 max_num_io=MAX_NUM_IO, io_embed_dim=IO_EMBED_DIM):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, n_layers, p)
        self.decoder = Decoder(input_dim, hidden_dim, n_layers, p)
        self.attn = Attention(hidden_dim, hidden_dim)
        self.initial_decoder_input = nn.Parameter(torch.zeros(1, input_dim))
        self.device = device
        self.num_preemption_buckets = num_preemption_buckets

        # #1 fix: IO embedding for conditioning the preemption head
        self.io_embed = nn.Embedding(max_num_io, io_embed_dim).to(device)

        # Preemption head input: cat(decoder_output, io_embedding)
        self.preemption_head = nn.Sequential(
            nn.Linear(hidden_dim + io_embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_preemption_buckets)
        ).to(device)

    def _preempt_logits(self, output, io_indices, detach_backbone=True):
        """Compute preemption logits conditional on the selected IO.
        When detach_backbone=True (training), prevents preemption
        gradient from flowing into encoder/decoder/attn (#6 fix)."""
        feat = output.detach() if detach_backbone else output
        io_emb = self.io_embed(io_indices)
        return self.preemption_head(torch.cat([feat, io_emb], dim=-1))

    def forward(self, input, num_io, mask, num_samples=1,
                use_preemption_head=True):
        """
        If use_preemption_head=False, sample preemption uniformly from
        buckets instead of from the head. Used by --disable_preemption_head.
        """
        batch_size = input.size(1); seq_len = input.size(0)
        predicted_mappings = torch.zeros(batch_size * num_samples, seq_len, dtype=torch.long).to(self.device)
        predicted_preemptions = torch.zeros(batch_size * num_samples, seq_len, dtype=torch.long).to(self.device)

        encoder_outputs, (hidden, cell) = self.encoder(input, mask)
        encoder_outputs = encoder_outputs.repeat_interleave(num_samples, dim=1)
        hidden = hidden.repeat_interleave(num_samples, dim=1)
        cell = cell.repeat_interleave(num_samples, dim=1)
        mask_decoding = mask.repeat_interleave(num_samples, dim=0).clone()
        decoder_input = self.initial_decoder_input.repeat(batch_size * num_samples, 1)
        log_probs_sum = 0

        for t in range(seq_len):
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)

            # Mapping head
            logits = self.attn(output, encoder_outputs, num_io, mask_decoding)
            log_probs = F.log_softmax(logits, dim=1)
            selected_indices = torch.multinomial(log_probs.exp(), 1).squeeze(1)
            predicted_mappings[:, t] = selected_indices
            selected_log_probs = log_probs.gather(1, selected_indices.unsqueeze(1)).squeeze(1)
            log_probs_sum = log_probs_sum + selected_log_probs

            # Preemption head (IO-conditioned)
            if use_preemption_head:
                # In forward pass (sampling), keep gradients live in case
                # any caller wants policy-gradient. Detach happens during
                # the actual REINFORCE update via get_log_prob_components.
                preemption_logits = self._preempt_logits(output, selected_indices, detach_backbone=False)
                preemption_log_probs = F.log_softmax(preemption_logits, dim=1)
                preemption_indices = torch.multinomial(preemption_log_probs.exp(), 1).squeeze(1)
                predicted_preemptions[:, t] = preemption_indices
                preempt_lp = preemption_log_probs.gather(1, preemption_indices.unsqueeze(1)).squeeze(1)
                log_probs_sum = log_probs_sum + preempt_lp
            else:
                # Uniform-random bucket — ablation mode (#disable_preemption_head)
                rand_indices = torch.randint(0, self.num_preemption_buckets,
                                              (predicted_preemptions.size(0),),
                                              device=self.device)
                predicted_preemptions[:, t] = rand_indices

            input_expanded = input.repeat_interleave(num_samples, dim=1)
            gather_indices = selected_indices.view(1, -1, 1).expand(1, -1, input.size(2))
            decoder_input = input_expanded.gather(0, gather_indices).squeeze(0)
            mask_decoding.scatter_(1, selected_indices.unsqueeze(1), 0)

        return predicted_mappings, predicted_preemptions, log_probs_sum

    def get_log_prob_components(self, input, mask, num_io, target_mappings,
                                 target_preemptions=None):
        """
        Teacher-forced log probabilities, returned as (mapping_lp,
        preemption_lp). The preemption_lp uses output.detach() internally
        so its gradient does not contaminate the encoder/decoder/attn
        backbone (#6 fix).
        Returns (mapping_lp, preemption_lp). preemption_lp is None when
        target_preemptions is None.
        """
        batch_size = input.size(1); seq_len = input.size(0)
        encoder_outputs, (hidden, cell) = self.encoder(input, mask)
        decoder_input = self.initial_decoder_input.repeat(batch_size, 1)
        mask_decoding = mask.clone()

        mapping_lp = torch.zeros(batch_size).to(self.device)
        preempt_lp = torch.zeros(batch_size).to(self.device) if target_preemptions is not None else None

        for t in range(seq_len):
            output, (hidden, cell) = self.decoder(decoder_input, hidden, cell)

            # Mapping log-prob (gradients flow into encoder/decoder/attn)
            logits = self.attn(output, encoder_outputs, num_io, mask_decoding)
            log_probs = F.log_softmax(logits, dim=1)
            actions = target_mappings[:, t]
            mapping_lp = mapping_lp + log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Preemption log-prob (gradients flow only into io_embed + head, NOT backbone)
            if target_preemptions is not None:
                preempt_logits = self._preempt_logits(output, actions, detach_backbone=True)
                preempt_log_probs = F.log_softmax(preempt_logits, dim=1)
                pa = target_preemptions[:, t]
                preempt_lp = preempt_lp + preempt_log_probs.gather(1, pa.unsqueeze(1)).squeeze(1)

            gather_indices = actions.view(1, -1, 1).expand(1, -1, input.size(2))
            decoder_input = input.gather(0, gather_indices).squeeze(0)
            mask_decoding.scatter_(1, actions.unsqueeze(1), 0)

        return mapping_lp, preempt_lp

    def get_log_prob(self, input, mask, num_io, target_mappings, target_preemptions=None):
        """Backwards-compatible wrapper. Returns SUM of mapping_lp and
        preemption_lp (or just mapping_lp if no preemption targets)."""
        m_lp, p_lp = self.get_log_prob_components(input, mask, num_io,
                                                   target_mappings, target_preemptions)
        return m_lp + p_lp if p_lp is not None else m_lp


# ==========================================
# SECTION 4: GA UTILITIES
# ==========================================

class Individual:
    def __init__(self, genes, fitness=None, preemptions=None):
        self.genes = genes
        self.fitness = fitness
        self.preemptions = preemptions


class Population:
    def __init__(self, individuals):
        self.individuals = individuals

    def sort(self):
        self.individuals.sort(key=lambda x: x.fitness if x.fitness is not None else float('-inf'),
                              reverse=True)

    def get_fittest(self):
        return self.individuals[0]


def crossover(parent1, parent2):
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
    """#2 fix: preemption draws snap to PREEMPTION_BUCKETS so the bucket
    index used in REINFORCE matches the value evaluated.
    continuous=True reverts #2: draws are continuous in [0.05, 0.99]."""
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
    candidates = random.sample(pop.individuals, k)
    return max(candidates, key=lambda x: x.fitness)


def objfunct_wrapper(mappings, dir_coords, cores, io_pairs, all_hops, pool=None, preemptions_list=None):
    if isinstance(mappings, torch.Tensor):
        mappings_list = mappings.cpu().tolist()
    else:
        mappings_list = mappings
    dir_np = dir_coords.cpu().numpy()
    tasks = []
    core_config = cores[0]
    for i, m in enumerate(mappings_list):
        p = preemptions_list[i] if preemptions_list is not None else None
        tasks.append((m, dir_np, core_config, io_pairs, all_hops, p))
    if not tasks:
        return torch.tensor([], dtype=torch.float32).to(device), None
    if pool is not None:
        results = pool.map(simulate_single_mapping, tasks)
    else:
        results = [simulate_single_mapping(t) for t in tasks]
    penalties = [r[0] for r in results]
    return torch.tensor(penalties, dtype=torch.float32).to(device), None


def evaluate_population_parallel(pop, dir_coords, cores, io_pairs, all_hops, pool=None):
    dir_np = dir_coords.cpu().numpy()
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


# ==========================================
# SECTION 5: TRAINING HELPERS
# ==========================================

def _do_model_update(model, optimizer, population, data_tensor, mask, num_io_local,
                     ema_baseline, EMA_DECAY, train_preempt_head=True):
    """
    Combined REINFORCE update on top 30% of population.
    Preemption gradients are isolated to io_embed + preemption_head
    via output.detach() inside get_log_prob_components (#6 fix).
    Bucket indices are computed from population's actual preemption
    values, which are guaranteed bucket-aligned by #2 fix.
    """
    population.sort()
    top_k = max(5, len(population.individuals) // 3)
    train_inds = population.individuals[:top_k]

    train_genes = torch.tensor([ind.genes for ind in train_inds]).to(device)
    train_preemption_indices = torch.tensor([
        preemptions_to_bucket_indices(ind.preemptions) for ind in train_inds
    ], dtype=torch.long).to(device)
    train_rewards = torch.tensor([ind.fitness for ind in train_inds], dtype=torch.float32).to(device)

    current_mean = train_rewards.mean().item()
    if ema_baseline is None:
        ema_baseline = current_mean
    else:
        ema_baseline = EMA_DECAY * ema_baseline + (1 - EMA_DECAY) * current_mean

    advantages = train_rewards - ema_baseline
    adv_std = advantages.std()
    if adv_std > 1e-6:
        advantages = advantages / (adv_std + 1e-6)

    model.train()
    optimizer.zero_grad()

    batch_size_train = len(train_inds)
    batch_input = data_tensor[:, 0:1, :].repeat(1, batch_size_train, 1)
    batch_mask = mask.repeat(batch_size_train, 1)

    target_preempts = train_preemption_indices if train_preempt_head else None
    mapping_lp, preempt_lp = model.get_log_prob_components(
        batch_input, batch_mask, num_io_local, train_genes, target_preempts
    )

    total_lp = mapping_lp + (preempt_lp if preempt_lp is not None else 0)
    pg_loss = -(total_lp * advantages).mean()
    pg_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return ema_baseline


def warmup_preemption_head(model, train_samples, data_tensor, mask, num_io_local,
                           epochs=HEAD_WARMUP_EPOCHS, lr=HEAD_WARMUP_LR):
    """
    #3 fix: supervised NLL pretraining of the preemption head on top-K
    from the warm-up GA pass. Backbone is implicitly frozen because the
    head's input uses output.detach() (#6 fix); we also restrict the
    optimizer to only the head + io_embed parameters as a belt-and-suspenders
    safeguard.
    """
    if not train_samples:
        return

    head_params = (list(model.io_embed.parameters())
                   + list(model.preemption_head.parameters()))
    head_opt = torch.optim.Adam(head_params, lr=lr)

    genes_list = [s[0] for s in train_samples]
    preempt_list = [s[1] for s in train_samples]
    bucket_targets = [preemptions_to_bucket_indices(p) for p in preempt_list]

    genes_t = torch.tensor(genes_list, dtype=torch.long).to(device)
    bucket_t = torch.tensor(bucket_targets, dtype=torch.long).to(device)

    bs = len(train_samples)
    batch_input = data_tensor[:, 0:1, :].repeat(1, bs, 1)
    batch_mask = mask.repeat(bs, 1)

    model.train()
    for e in range(epochs):
        head_opt.zero_grad()
        _, preempt_lp = model.get_log_prob_components(
            batch_input, batch_mask, num_io_local, genes_t, bucket_t
        )
        loss = -preempt_lp.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
        head_opt.step()
        if (e + 1) % 10 == 0:
            print(f"  [Head Warmup] epoch {e+1}/{epochs}, NLL: {-preempt_lp.mean().item():.4f}")


# ==========================================
# SECTION 6: PRETRAIN (mapping head only)
# ==========================================

def pretrain_model(num_cores, num_io, epoch, lr=1e-4):
    """
    Pretrains the mapping backbone only. The preemption head is left at
    random init; it gets warmed up in run_lmga's first outer iter (#3).
    To prevent the head from receiving any gradient here, we use random
    bucket preemptions and pass target_preemptions=None to the REINFORCE
    update.
    """
    print(f"--- Pretraining (IGSA-FIXED, mapping head only): {num_cores}c {num_io}io ---")

    data, cores, io, all_hops = prep_data(num_cores, num_io, test=False)
    data_tensor = data.permute(1, 0, 2).to(device)
    num_io_local = len(io)
    mask = torch.ones(data_tensor.size(1), num_cores).to(device)

    input_dim = 2; hidden_dim = 128
    model = PointerNet(input_dim, hidden_dim, 2, 0.1, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_workers = mp.cpu_count()
    pool = mp.Pool(processes=num_workers)

    try:
        for e in range(epoch):
            model.train()
            optimizer.zero_grad()

            # Use_preemption_head=False during pretraining: NN samples
            # mappings, preemptions drawn uniformly from buckets.
            mappings, _, log_probs_sum = model(
                data_tensor, num_io_local, mask, num_samples=1, use_preemption_head=False
            )

            # Provide random bucket preemptions matching the sample count
            preempt_values = [
                [random.choice(PREEMPTION_BUCKETS) for _ in range(num_cores)]
                for _ in range(mappings.shape[0])
            ]

            penalties, _ = objfunct_wrapper(
                mappings, data_tensor[:, 0, :], cores, io, all_hops,
                pool=pool, preemptions_list=preempt_values
            )

            baseline = penalties.mean()
            advantage = penalties - baseline
            loss = -(log_probs_sum * advantage).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            print(f"[Pretrain] Epoch {e+1}/{epoch} | Loss: {loss.item():.4f} | Avg Reward: {baseline.item():.2f}")
    finally:
        pool.close()
        pool.join()

    torch.save(model.state_dict(), f"lmga_model_{num_cores}cores_{num_io}io.pt")
    print("Pretraining complete (mapping backbone saved).")


# ==========================================
# SECTION 7: MAIN RUN
# ==========================================

def run_lmga(num_cores, num_io, ga_generations_per_iter=100, pop_size=100,
             stability_target=20, save_model=True, output_suffix="",
             # Ablation flags
             disable_preemption_head=False,
             disable_constructive=False,
             disable_inloop_updates=False,
             disable_warmup=False,
             continuous_preemptions=False,
             head_warmup_epochs=HEAD_WARMUP_EPOCHS,
             num_constructive=NUM_CONSTRUCTIVE,
             update_interval=UPDATE_INTERVAL):

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
    mask = torch.ones(1, num_cores).to(device)

    input_dim = 2; hidden_dim = 128
    model = PointerNet(input_dim, hidden_dim, 2, 0.1, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Load pretrained backbone. Head + io_embed start random — that's fine
    # because warmup_preemption_head trains them before they affect search.
    pretrained_path = f"lmga_model_{num_cores}cores_{num_io}io.pt"
    if os.path.exists(pretrained_path):
        try:
            state_dict = torch.load(pretrained_path, map_location=device)
            missing, _ = model.load_state_dict(state_dict, strict=False)
            new_keys = [m for m in missing if any(k in m for k in ['preemption', 'io_embed'])]
            print(f"Loaded backbone from {pretrained_path}; "
                  f"{len(new_keys)} head/io_embed keys at random init "
                  f"(will be warmed up).")
        except Exception as e:
            print(f"Error loading model ({e}), starting fresh.")
    else:
        print("No pretrained model found. Starting from scratch.")

    best_global_fitness = float('-inf')
    best_global_mapping = None
    best_global_preemptions = None
    best_population_snapshot = []
    stability_counter = 0
    outer_loop_iter = 0

    ema_baseline = None
    EMA_DECAY = 0.9
    IMPROVEMENT_THRESHOLD = 0.005

    num_workers = mp.cpu_count()
    pool = mp.Pool(processes=num_workers)

    head_warmed_up = disable_warmup or disable_preemption_head  # skip warmup if head disabled

    try:
        while stability_counter < stability_target:
            outer_loop_iter += 1
            # Warm-up flag: only the FIRST outer iter is treated as warm-up.
            # During warm-up: no constructive, no in-loop updates, random
            # bucket preemptions for NN samples. The head is then trained
            # on the warm-up's top-K before outer iter 2 begins.
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
                # During warm-up OR when head is disabled, use random bucket
                # preemptions for NN-sampled individuals. Otherwise sample
                # both from the model.
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

            # #4 (related): random initial individuals also get bucket preemptions
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

            if best_global_fitness == float('-inf'):
                is_real_improvement = True
                relative_improvement = 0
            else:
                relative_improvement = (current_fitness - best_global_fitness) / abs(best_global_fitness)
                is_real_improvement = relative_improvement > IMPROVEMENT_THRESHOLD

            if is_real_improvement and current_fitness > best_global_fitness:
                tag = "FIRST BEST" if best_global_fitness == float('-inf') else "NEW BEST"
                pct = f" ({relative_improvement*100:+.2f}%)" if best_global_fitness != float('-inf') else ""
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
                # #3 fix: supervised NLL on top-K of warm-up GA
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
                # Don't run the mapping REINFORCE update on this iter — the
                # warm-up GA pop is closer to random than to a learned signal.
                # Also reset stability so the warm-up iter doesn't count
                # against convergence budget.
                stability_counter = 0
            else:
                # Normal end-of-outer REINFORCE update
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


# ==========================================
# SECTION 8: CLI
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="run", choices=["pretrain", "run", "multirun"])
    parser.add_argument("--num_cores", type=int, default=32)
    parser.add_argument("--num_io", type=int, default=2)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--constructive", type=int, default=NUM_CONSTRUCTIVE)
    parser.add_argument("--update_interval", type=int, default=UPDATE_INTERVAL)
    parser.add_argument("--head_warmup_epochs", type=int, default=HEAD_WARMUP_EPOCHS)
    # Ablation flags
    parser.add_argument("--disable_preemption_head", action="store_true",
                        help="Use random bucket preemptions instead of NN head. "
                             "Isolates mapping head contribution.")
    parser.add_argument("--disable_constructive", action="store_true",
                        help="Set NUM_CONSTRUCTIVE=0. Isolates constructive operator.")
    parser.add_argument("--disable_inloop_updates", action="store_true",
                        help="Skip in-loop NN updates. Isolates in-loop training.")
    parser.add_argument("--disable_warmup", action="store_true",
                        help="Skip head warm-up. Reproduces original broken behavior.")
    parser.add_argument("--continuous_preemptions", action="store_true",
                        help="Revert fix #2: use continuous preemptions in [0.20,0.90] "
                             "instead of bucket-snapped. Finer GA resolution, but the "
                             "preemption head's REINFORCE actions no longer match rewards.")
    args = parser.parse_args()

    if args.mode == "pretrain":
        pretrain_model(args.num_cores, args.num_io, epoch=args.epoch)

    elif args.mode == "run":
        run_lmga(args.num_cores, args.num_io,
                 disable_preemption_head=args.disable_preemption_head,
                 disable_constructive=args.disable_constructive,
                 disable_inloop_updates=args.disable_inloop_updates,
                 disable_warmup=args.disable_warmup,
                 continuous_preemptions=args.continuous_preemptions,
                 head_warmup_epochs=args.head_warmup_epochs,
                 num_constructive=args.constructive,
                 update_interval=args.update_interval)

    elif args.mode == "multirun":
        ablation_tag = "_full"
        if args.disable_preemption_head: ablation_tag = "_noHead"
        elif args.disable_constructive:  ablation_tag = "_noConstruct"
        elif args.disable_inloop_updates: ablation_tag = "_noInloop"
        elif args.disable_warmup: ablation_tag = "_noWarmup"
        elif args.continuous_preemptions: ablation_tag = "_contPreempt"

        print(f"=== MULTI-RUN ({ablation_tag.strip('_').upper()}): {args.num_runs} runs for "
              f"{args.num_cores}c {args.num_io}io ===")

        all_results = []
        for run_num in range(1, args.num_runs + 1):
            print(f"\n{'='*60}\nRUN {run_num}/{args.num_runs}\n{'='*60}")
            run_start = time()
            fitness = run_lmga(
                args.num_cores, args.num_io,
                save_model=False, output_suffix=f"_run{run_num}{ablation_tag}",
                disable_preemption_head=args.disable_preemption_head,
                disable_constructive=args.disable_constructive,
                disable_inloop_updates=args.disable_inloop_updates,
                disable_warmup=args.disable_warmup,
                continuous_preemptions=args.continuous_preemptions,
                head_warmup_epochs=args.head_warmup_epochs,
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

        print(f"\n{'='*60}\nMULTI-RUN SUMMARY ({ablation_tag.strip('_').upper()}): "
              f"{args.num_cores}c {args.num_io}io, {args.num_runs} runs")
        print(f"{'='*60}")
        print(f"Best:  {best_fitness:.2f} (run {best_run})")
        print(f"Worst: {worst_fitness:.2f} (run {worst_run})")
        print(f"Mean:  {mean_fitness:.2f} ± {std_fitness:.2f}")
        print(f"Best hit count: {best_hit_count}/{args.num_runs} (within 0.1%)")
        print(f"Mean Runtime: {np.mean(times):.2f}s ± {np.std(times):.2f}s")
        for run_num, fitness, rt in all_results:
            marker = " <-- BEST" if fitness == best_fitness else (" <-- WORST" if fitness == worst_fitness else "")
            print(f"  Run {run_num:2d}: fitness={fitness:.2f}, time={rt:.2f}s{marker}")

        summary_filename = f"multirun_{args.num_cores}cores_{args.num_io}io_{args.num_runs}runs_igsafixed{ablation_tag}.txt"
        with open(summary_filename, "w") as f:
            f.write(f"# Algorithm: LMGA-IGSA-FIXED ({ablation_tag.strip('_')})\n")
            f.write(f"# Ablation: disable_preemption_head={args.disable_preemption_head}, "
                    f"disable_constructive={args.disable_constructive}, "
                    f"disable_inloop_updates={args.disable_inloop_updates}, "
                    f"disable_warmup={args.disable_warmup}\n")
            f.write(f"# Num Runs: {args.num_runs}\n")
            f.write(f"# Best: {best_fitness:.2f} (run {best_run})\n")
            f.write(f"# Worst: {worst_fitness:.2f} (run {worst_run})\n")
            f.write(f"# Mean ± Std: {mean_fitness:.2f} ± {std_fitness:.2f}\n")
            f.write(f"# Best Hit Count: {best_hit_count}/{args.num_runs}\n")
            f.write(f"# Mean Runtime: {np.mean(times):.2f}s ± {np.std(times):.2f}s\n\n")
            f.write("run,fitness,runtime\n")
            for run_num, fitness, rt in all_results:
                f.write(f"{run_num},{fitness:.2f},{rt:.4f}\n")
        print(f"\nSummary saved to {summary_filename}")

        for run_num in range(1, args.num_runs + 1):
            run_file = f"top_100_data_{args.num_cores}cores_{args.num_io}io_run{run_num}{ablation_tag}.txt"
            if os.path.exists(run_file):
                os.remove(run_file)
        print("Cleaned up per-run output files.")
