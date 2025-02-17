import argparse
import itertools
from dataclasses import dataclass
from time import time
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from benchmark.utils import read_jssp_file


@dataclass(frozen=True)
class Task:
    job: int
    step: int
    machine: int
    time: int


@dataclass
class MachineState:
    m_id: int
    release_time: int = 0
    current_task: Task | None = None


class Queue:
    r""" FIFO queue for tasks """

    def __init__(self, n_machines):
        self._q = [[] for _ in range(n_machines)]

    def enqueue(self, t: Task):
        r""" Enqueue a task, first in first out (FIFO), append the task to the end of the queue for the machine.

        Args:
            t (Task): task to enqueue.
        """
        self._q[t.machine].append(t)

    def get(self, m: int):
        r""" Get the next task to process for the machine.

        Args:
            m (int): machine id

        Returns:
            (Task): task to process or None if there are no tasks to process
        """
        if len(self._q[m]) > 0:
            return self._q[m].pop(0)
        else:
            return None

    def empty(self):
        return all([len(x) == 0 for x in self._q])


class LWRQueue(Queue):
    r""" LWR queue for tasks """

    def enqueue(self, t: Task):
        r""" Enqueue a task, Least Work Remainining (LWR), sort the queue by time in ascending order.

        Args:
            t (Task): task to enqueue.
        """
        # append the task to the queue of the machine
        super().enqueue(t)
        # sort the queue by time in ascending order
        self._q[t.machine] = sorted(self._q[t.machine], key=lambda t: t.time)


class MWRQueue(Queue):
    r""" MWR queue for tasks """

    def enqueue(self, t: Task):
        r""" Enqueue a task, Most Work Remainining (MWR), sort the queue by time in descending order.

        Args:
            t (Task): task to enqueue.
        """
        # append the task to the queue of the machine
        super().enqueue(t)
        # sort the queue by time in descending order
        self._q[t.machine] = sorted(self._q[t.machine], key=lambda t: t.time, reverse=True)


def solve(times, machines, variant="fifo", **kwargs) -> int:
    r""" Solve the JSSP problem.

    Args:
        times: list of list of int, job processing times
        machines: list of list of int, machine processing times
        variant (str): queue class to use for task scheduling

    Returns:
        (int): makespan
    """

    _VERBOSE = kwargs.get('verbose', False)
    n_jobs, m_machines = len(times), len(times[0])

    all_tasks = {(j_id, t_id): Task(j_id, t_id, machines[j_id][t_id], times[j_id][t_id])
                 for j_id, t_id in itertools.product(range(n_jobs), range(m_machines))}

    m_states = [MachineState(i) for i in range(m_machines)]

    if variant == 'mwr':
        queue_cls = MWRQueue
    elif variant == 'lwr':
        queue_cls = LWRQueue
    else:
        queue_cls = Queue

    # initiate the queue
    queue = queue_cls(m_machines)
    # enqueue the queue with the first task of each job
    for j in range(n_jobs):
        queue.enqueue(all_tasks[(j, 0)])

    t = 0
    # if the queue is not empty or there are still tasks to be processed
    while not queue.empty() or any([m.current_task is not None for m in m_states]):
        # check remaining tasks
        rt = [m.release_time for m in m_states if m.current_task is not None]
        if len(rt) > 0:
            assert min(rt) > t
            t = min(rt)
        else:
            assert t == 0
        # if _VERBOSE:
        #     print(f't = {t}')

        for m in m_states:
            # check all tasks in the machine finished
            if m.release_time == t and m.current_task is not None:
                # if _VERBOSE:
                #     print(f'Task {m.current_task} complete')
                next_id = (m.current_task.job, m.current_task.step + 1)
                # insert next task to queue
                if next_id in all_tasks:
                    queue.enqueue(all_tasks[next_id])
                m.current_task = None

        for m in m_states:
            if m.current_task is not None:
                continue
            m.current_task = queue.get(m.m_id)
            if m.current_task is not None:
                m.release_time = t + m.current_task.time
                if _VERBOSE:
                    print(f'T_{m.current_task.job, m.current_task.step} '
                          f'@ m_id {m.current_task.machine}, '
                          f's_time {t}, f_time {m.release_time}')
    if _VERBOSE:
        print(f'All tasks completed at {t}')

    return t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default="ft")
    parser.add_argument('--id', type=str, default="06")
    parser.add_argument('--format', type=str, default="taillard", choices=["standard", "taillard"])
    parser.add_argument('--variant', choices=['fifo', 'lwr', 'mwr'], default='fifo')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.format == "taillard":
        times, machines = read_jssp_file(
            f"benchmark/JSSP/{args.problem}/Taillard_specification/{args.problem}{args.id}.txt",
            args.problem,
            args.id,
            args.format)
    else:
        times, machines = read_jssp_file(f"benchmark/JSSP/{args.problem}/{args.problem}{args.id}.txt",
                                         args.problem,
                                         args.id,
                                         args.format)

    tic = time()
    makespan = solve(times, machines, variant=args.variant, verbose=args.verbose)
    toc = time()
    print({
        'solver': f'list_{args.variant}',
        'solution': makespan,
        'time': (toc - tic),
        'problem': f'{args.format}_{args.problem}_{args.id}',
        'times': len(times),
        'machines': len(times[0]),
    })


if __name__ == '__main__':
    main()
