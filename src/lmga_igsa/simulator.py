"""
The fitness function / environment: given a core->IO mapping and a
preemption vector, simulate the resulting schedule and return the
(negative) makespan.

FIX #2 lives here: `simulate_single_mapping`'s fallback path (used only
when a caller evaluates an individual with no stored preemptions) draws
from `PREEMPTION_BUCKETS` instead of a continuous distribution. This
guarantees any preemption value that ever reaches REINFORCE training is
bucket-aligned, so the bucket-index *action* and the fitness *reward*
always refer to the same value. In normal operation (via
`ga.evaluate_population_parallel`), every `Individual` already carries
`preemptions`, so this fallback is defensive rather than load-bearing.
"""

import math
import random
from queue import PriorityQueue

import numpy as np

from .config import PREEMPTION_BUCKETS
from .data import check_path_conflict


def _insert_by_finish(lst, ev):
    """Insert (core, start, finish) into lst, kept sorted by finish time."""
    f = ev[2]
    lo, hi = 0, len(lst)
    while lo < hi:
        mid = (lo + hi) // 2
        if lst[mid][2] < f:
            lo = mid + 1
        else:
            hi = mid
    lst.insert(lo, ev)


def _merge_sorted(intervals):
    """Merge a list of [start, finish] intervals, already sorted by start."""
    if not intervals:
        return []
    merged = []
    ps, pf = intervals[0]
    for s, f in intervals[1:]:
        if s <= pf:
            pf = max(pf, f)
        else:
            merged.append([ps, pf])
            ps, pf = s, f
    merged.append([ps, pf])
    return merged


def simulate_single_mapping(args):
    """
    Simulate one (mapping, preemption) assignment and return
    (-makespan, schedule_log, preemptions_used).

    args is either:
      (mapping, dir_np, core_config, ioArray, all_hops, stored_preemptions)
      (mapping, dir_np, core_config, ioArray, all_hops)   # legacy 5-tuple

    stored_preemptions, when given, is used as-is (this is the normal
    path: the GA's Individual already carries bucket-aligned
    preemptions). When None or mismatched in length, a fresh
    bucket-snapped vector is drawn (fix #2's fallback).
    """
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
                if patternsToProcess == 0:
                    patternsToProcess = 1
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
            if not isAllocated[c]:
                processQueue.append((st, c))
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
                    if core2 == coreId:
                        continue
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
                    if duration <= startTime[coreId]:
                        gaps.append((0, startTime[coreId]))
                else:
                    if merged[0][0] > 0:
                        gaps.append((0, merged[0][0]))
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
                        if not isAllocated[other]:
                            startTime[other] = max(startTime[other], f)
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
                    if not isAllocated[other]:
                        startTime[other] = max(startTime[other], f)
                made_progress = True

        if not made_progress and passiveQueue.empty():
            break

    makespan = max(globalTime) if globalTime else 0
    return -makespan, scheduleLog, preemptions_map