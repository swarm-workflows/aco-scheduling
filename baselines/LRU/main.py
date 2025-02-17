import argparse
import itertools
from dataclasses import dataclass
from time import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from benchmark.utils import read_file
from utils import store


@dataclass(frozen=True)
class Task:
    job: int
    step: int
    machine: int
    time: int


@dataclass
class MachineState:
    uid: int
    release_time: int = 0
    current_task: Task | None = None


class Queue:
    r""" FIFO queue for tasks """

    def __init__(self, n_machines):
        self._q = [[] for _ in range(n_machines)]

    def enqueue(self, t: Task):
        self._q[t.machine].append(t)

    def get(self, m: int):
        if len(self._q[m]) > 0:
            return self._q[m].pop(0)
        else:
            return None

    def empty(self):
        return all([len(x) == 0 for x in self._q])


class LRUQueue(Queue):
    r""" LRU queue for tasks """

    def enqueue(self, t: Task):
        super().enqueue(t)
        self._q[t.machine] = sorted(self._q[t.machine], key=lambda t: t.time)


def solve(times, machines, queue_cls):
    jobs_n, machines_n = len(times), len(times[0])

    all_tasks = {(j, t): Task(j, t, machines[j][t], times[j][t])
                 for j, t in itertools.product(range(jobs_n), range(machines_n))}

    states = [MachineState(i) for i in range(machines_n)]

    queue = queue_cls(machines_n)
    for j in range(jobs_n):
        queue.enqueue(all_tasks[(j, 0)])

    t = 0
    while not queue.empty() or any([m.current_task is not None for m in states]):
        rt = [m.release_time for m in states if m.current_task is not None]
        if len(rt) > 0:
            assert min(rt) > t
            t = min(rt)
        else:
            assert t == 0
        print(f't = {t}')

        for m in states:
            if m.release_time == t and m.current_task is not None:
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
                m.release_time = t + m.current_task.time
                print(f'Task {m.current_task} start, release {m.release_time}')

    print(f'All tasks completed at {t}')
    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default="ft")
    parser.add_argument('--id', type=str, default="06")
    parser.add_argument('--format', type=str, default="taillard", choices=["standard", "taillard"])
    parser.add_argument('--variant', choices=['fifo', 'lru'], default='fifo')
    parser.add_argument('--store', type=str)
    args = parser.parse_args()
    if args.format == "taillard":
        times, machines = read_file(f"benchmark/{args.problem}/Taillard_specification/{args.problem}{args.id}.txt",
                                    args.problem,
                                    args.id,
                                    args.format)
    else:
        times, machines = read_file(f"benchmark/{args.problem}/{args.problem}{args.id}.txt",
                                    args.problem,
                                    args.id,
                                    args.format)

    queue_cls = Queue if args.variant == 'fifo' else LRUQueue
    tic = time()
    makespan = solve(times, machines, queue_cls)
    toc = time()
    print(f"Time: {toc - tic:.2f}")
    if args.store:
        store(args.store, {
            'solver': f'list_{args.variant}',
            'solution': makespan,
            'time': (toc - tic),
            'problem': f'{args.format}_{args.problem}_{args.id}',
            'times': len(times),
            'machines': len(times[0]),
        })


if __name__ == '__main__':
    main()
