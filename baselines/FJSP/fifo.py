#!/usr/bin/env python
import argparse
import itertools
from dataclasses import dataclass
from time import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from benchmark.utils import read_fjsp_file


@dataclass(frozen=True)
class Task:
    r"""Task dataclass.

    Attributes:
        job (int): Job identifier.
        step (int): Step identifier.
        machine_time (Dict): Machine time dictionary, machine id as keys, and time as values.
    """
    job: int
    step: int
    machine_time: Dict


@dataclass
class MachineState:
    r"""Machine state dataclass.

    Attributes:
        uid (int): Machine unique identifier.
        release_time (int): Time at which the machine will be available.
        current_task (Task | None): Current task being processed.
    """
    uid: int
    release_time: int = 0
    current_task: Task | None = None


class Queue:
    r"""Queue class. """

    def __init__(self, n_machines):
        self._q = [[] for _ in range(n_machines)]

    def enqueue(self, t: Task):
        for m in t.machine_time.keys():
            self._q[m].append(t)

    def get(self, m: int):
        if len(self._q[m]) > 0:
            op = self._q[m].pop(0)
            for mq in op.machine_time.keys():
                if op in self._q[mq]:
                    self._q[mq].remove(op)
            return op
        else:
            return None

    def empty(self):
        return all([len(x) == 0 for x in self._q])


class LWRQueue(Queue):
    r"""LWR queue class. """

    def enqueue(self, t: Task):
        super().enqueue(t)
        for m in t.machine_time.keys():
            self._q[m] = sorted(self._q[m], key=lambda x: x.machine_time[m])


class MWRQueue(Queue):
    r"""MWR queue class. """

    def enqueue(self, t: Task):
        super().enqueue(t)
        for m in t.machine_time.keys():
            self._q[m] = sorted(self._q[m], key=lambda x: x.machine_time[m], reverse=True)


def solve(times, n_jobs, n_machines, variant='fifo', **kwargs):
    r"""Solve the Flexible Job Shop Problem.

    Args:
        times (List[List[Tuple[int, int]]]): List of machines times, each element is a list of tuples
            representing the machine time for each job and step.
        n_jobs (int): Number of jobs.
        n_machines (int): Number of machines.
        variant (str): Queue variant, one of ['fifo', 'lwr', 'mwr'].

    Returns:
        int: Makespan.
    """
    _VERBOSE = kwargs.get('verbose', False)

    all_tasks = {}
    for j in range(len(times)):
        for t in range(len(times[j])):
            all_tasks[(j, t)] = Task(j, t, {m - 1: d for m, d in times[j][t]})

    states = [MachineState(i) for i in range(n_machines)]

    if variant == 'fifo':
        queue_cls = Queue
    elif variant == 'lwr':
        queue_cls = LWRQueue
    elif variant == 'mwr':
        queue_cls = MWRQueue
    queue = queue_cls(n_machines)
    for j in range(n_jobs):
        queue.enqueue(all_tasks[(j, 0)])

    t = 0
    while not queue.empty() or any([m.current_task is not None for m in states]):
        rt = [m.release_time for m in states if m.current_task is not None]
        if len(rt) > 0:
            # NOTE: ORB7 issue
            # assert min(rt) > t
            t = min(rt)
        else:
            assert t == 0

        for m in states:
            if m.release_time == t and m.current_task is not None:

                next_id = (m.current_task.job, m.current_task.step + 1)
                if next_id in all_tasks:
                    queue.enqueue(all_tasks[next_id])
                m.current_task = None

        for m in states:
            if m.current_task is not None:
                continue
            m.current_task = queue.get(m.uid)
            if m.current_task is not None:
                m.release_time = t + m.current_task.machine_time[m.uid]
                if _VERBOSE:
                    print(f'T_{m.current_task.job, m.current_task.step} '
                          f'@ m_id {m.uid}, '
                          f's_time {t}, f_time {m.release_time}')
    if _VERBOSE:
        print(f'All tasks completed at {t}')
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./benchmark/FJSP/Barnes/Text/mt10c1.fjs')
    parser.add_argument('--variant', choices=['fifo', 'lwr', 'mwr'], default='fifo')
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args()
    # args = vars(args)
    tic = time()
    n_jobs, n_machines, jobs = read_fjsp_file(args.path)
    makespan = solve(jobs, n_jobs, n_machines, variant=args.variant, verbose=args.verbose)
    toc = time()
    # print(f"Time: {toc - tic:.2f}")
    # print(f"{n_jobs}, {n_machines}, makespan: {makespan}, time: {toc - tic:.4f}")
    print({'solver': f'list_{args.variant}',
           'solution': makespan,
           'walltime': (toc - tic),
           'problem': f'{args.path.split("/")[-1]}',
           'n_jobs': n_jobs,
           'm_machines': n_machines,
           }
          )


if __name__ == '__main__':
    main()
