#!/usr/bin/env python
import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass
from typing import Tuple, Dict
from benchmark_fjsp.utils import read_fjsp_file
from utils import store
from time import time

@dataclass(frozen=True)
class Task:
    job: int
    step: int
    machine_time: Dict

@dataclass
class MachineState:
    uid: int
    release_time: int = 0
    current_task: Task | None = None

class Queue:
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

def solve(machines_times):
    jobs_n, machines_n = len(machines_times), len(machines_times)

    all_tasks = {(j, t): Task(j, t, {m-1: d for m, d in machines_times[j][t]})
        for j,t in itertools.product(range(jobs_n), range(machines_n))}

    states = [MachineState(i) for i in range(machines_n)]

    queue = Queue(machines_n)
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
                m.release_time = t + m.current_task.machine_time[m.uid]
                print(f'Task {m.current_task} start, release {m.release_time}')

    print(f'All tasks completed at {t}')
    return t



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='./benchmark_fjsp/Monaldo/Fjsp/Job_Data/Brandimarte_Data/Text/Mk06.fjs')
    parser.add_argument('--store', type=str)
    args = parser.parse_args()
    data = read_fjsp_file(args.path)
    tic = time()
    makespan = solve(data[2])
    toc = time()
    print(f"Time: {toc - tic:.2f}")
    if args.store:
        store(args.store, {
            'solver': f'list_{args.variant}',
            'solution': makespan,
            'time': (toc - tic),
            'problem': f'{args.format}_{args.problem}_{args.id}',
            'times': len(data),
            'machines': len(data[0]),
        })

if __name__ == '__main__':
        main()
