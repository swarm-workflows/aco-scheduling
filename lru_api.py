from dataclasses import dataclass
from time import time
import collections
import numpy as np


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
    """FIFO queue for tasks"""

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
    """LRU queue for tasks"""

    def enqueue(self, t: Task):
        super().enqueue(t)
        self._q[t.machine].append(t)
        self._q[t.machine] = sorted(self._q[t.machine], key=lambda t: t.time)


class LRUSolver:
    """Solver class to match OR-Tools interface"""

    def __init__(self):
        self.wall_time = 0
        self.objective_value = 0
        self.num_conflicts = 0
        self.num_branches = 0


def lru_api(jobs, machines, variant='lru', **kwargs):
    """Solve the problem using LRU/FIFO approach

    Args:
        jobs (numpy.ndarray or list): Array/list of job durations
        machines (numpy.ndarray or list): Array/list of machine assignments
        variant (str): Either 'lru' or 'fifo'

    Returns:
        LRUSolver: Solver object with solution information
    """
    # Convert inputs to numpy arrays if they aren't already
    jobs = np.array(jobs)
    machines = np.array(machines)

    jobs_n, machines_n = jobs.shape
    queue_cls = LRUQueue if variant == 'lru' else Queue

    # Convert to 0-based machine indices if needed
    machines = machines - 1 if machines.min() > 0 else machines

    # Create all tasks
    all_tasks = {
        (j, t): Task(j, t, machines[j][t], jobs[j][t])
        for j in range(jobs_n)
        for t in range(machines_n)
    }

    states = [MachineState(i) for i in range(machines_n)]
    queue = queue_cls(machines_n)

    # Initialize with first task of each job
    for j in range(jobs_n):
        queue.enqueue(all_tasks[(j, 0)])

    start_time = time()
    t = 0
    while not queue.empty() or any([m.current_task is not None for m in states]):
        rt = [m.release_time for m in states if m.current_task is not None]
        if len(rt) > 0:
            t = min(rt)

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
                m.release_time = t + m.current_task.time

    solver = LRUSolver()
    solver.wall_time = time() - start_time
    solver.objective_value = t
    return solver
