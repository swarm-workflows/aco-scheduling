#!/usr/bin/env python
import argparse
import itertools
from dataclasses import dataclass
from time import time
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from benchmark_fjsp.utils import read_fjsp_file
from utils import store


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


def solve(machines_times, n_jobs, n_machines, args=dict()):
    r"""Solve the Flexible Job Shop Problem.

    Args:
        machines_times (List[List[Tuple[int, int]]]): List of machines times, each element is a list of tuples
            representing the machine time for each job and step.

    Returns:
        int: Makespan.
    """

    # n_jobs, n_machines = len(machines_times), len(machines_times)

    all_tasks = {(j, t): Task(j, t, {m - 1: d for m, d in machines_times[j][t]})
                 for j, t in itertools.product(range(n_jobs), range(n_machines))}

    states = [MachineState(i) for i in range(n_machines)]

    queue = Queue(n_machines)
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
        if args.get("verbose", False):
            print(f't = {t}')

        for m in states:
            if m.release_time == t and m.current_task is not None:
                if args.get("verbose", False):
                    print(f'Task {m.current_task} complete')
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
                if args.get("verbose", False):
                    print(f'Task {m.current_task} start, release {m.release_time}')
    if args.get("verbose", False):
        print(f'All tasks completed at {t}')
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./benchmark_fjsp/Monaldo/Fjsp/Job_Data/Barnes/Text/mt10c1.fjs')
    parser.add_argument('--store', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    args = parser.parse_args()
    # args = vars(args)
    tic = time()
    n_jobs, n_machines, jobs = read_fjsp_file(args.path)
    makespan = solve(jobs, n_jobs, n_machines, args=dict())
    toc = time()
    # print(f"Time: {toc - tic:.2f}")
    print(f"{n_jobs}, {n_machines}, makespan: {makespan}, time: {toc - tic:.4f}")
    if args.store:
        path = "."
        store(path, {'solver': f'list_{args.variant}',
                     'solution': makespan,
                     'time': (toc - tic),
                     'problem': f'{args.path.split("/")[-1]}',
                     'jobs': len(n_jobs),
                     'machines': len(n_machines),
                     })


if __name__ == '__main__':
    main()
