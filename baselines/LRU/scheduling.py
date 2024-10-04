#!/usr/bin/env python3

import sys
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
from mealpy import ACOR, PSO, WOA, PermutationVar, Problem

from baselines.LRU.LRUSchedule import LRUSchedule, OptimizedLRUSchedule


def create_random_job_scheduling_problem(n_jobs, n_machines, seed=None):
    r""" Create a random job scheduling problem with n_jobs and n_machines.

    Args:
        n_jobs (int): Number of jobs.
        n_machines (int): Number of machines.
        seed (int): Random seed.

    Returns:
        np.ndarray: Random job scheduling problem with n_jobs and n_machines.
    """
    if seed is not None:
        np.random.seed(seed)

    mat_data = np.random.triangular(10, 150, 300, size=n_jobs * n_machines)
    mat_data = mat_data.reshape(n_jobs, n_machines)
    # mat_data = np.random.normal((n_jobs*n_machines)) * 50 + 50  # Random processing times
    # mat_data = np.random.normal((n_jobs*n_machines)) * 50 + 50  # Random processing times
    # mat_data = np.random.random((n_jobs,n_machines)) * 100

    # sys.exit(0)
    return (mat_data)


def visualizeLRU(data, machine_schedule, path=None, label=None):
    """
    Visualization for job scheduling problem with LRU (we can merge it with the top one but needs some polishing).
    """
    with plt.style.context('ggplot'):
        n_machines, n_jobs = data['n_machines'], data['n_jobs'],

        fig, ax = plt.subplots()
        Y = np.arange(n_machines)
        # Create bars for the Gantt chart
        for machine_idx in range(n_machines):
            for start_time, job_idx in machine_schedule[machine_idx]["jobs"]:
                ax.barh(
                    machine_idx,
                    machine_schedule[machine_idx]["total_time"] -
                    start_time,
                    left=start_time,
                    height=0.5,
                    label=f"Job {job_idx}",
                    align='center')

        # Set labels and titles
        ax.set_xlabel("Time")
        ax.set_ylabel("Machine")
        ax.set_yticks(Y)
        ax.set_yticklabels([f"M{i}" for i in Y])
        ax.set_title("Job Scheduling: %s" % label)
        plt.tight_layout()
        if path is not None:
            plt.savefig(path)
        plt.show()


def perm_to_machine_schedule(data, x):
    n_machines, time_mat = data["n_machines"], data["job_times"]
    machine_schedule = [{"jobs": [], "total_time": 0} for _ in range(n_machines)]
    for machine_idx, job_idx in enumerate(x):
        machine_idx = machine_idx % n_machines
        ms = machine_schedule[machine_idx]
        ms["jobs"].append([ms["total_time"], job_idx])
        ms["total_time"] += time_mat[job_idx][machine_idx]
    return machine_schedule


def visualize(data, x, path=None, label=None):
    machine_schedule = perm_to_machine_schedule(data, x)
    visualizeLRU(data, machine_schedule, path, label)


def PSO_Scheduling(problem):
    model = PSO.AIW_PSO(epoch=100, pop_size=100, seed=10)
    model.solve(problem)

    print(f"Best agent: {model.g_best}")                    # Encoded solution
    print(f"Best solution: {model.g_best.solution}")        # Encoded solution
    print(f"Best fitness: {model.g_best.target.fitness}")
    # Decoded (Real) solution
    print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution)}")
    x_decoded = model.problem.decode_solution(model.g_best.solution)
    x = x_decoded["per_var"]
    visualize(data, x, path='schedule-pso.png', label=f"PSO makespan: {model.g_best.target.fitness}")
    model.history.save_global_objectives_chart(filename="goc-pso")
    model.history.save_local_objectives_chart(filename="loc-pso")


def ACOR_Scheduling(problem):
    model = ACOR.OriginalACOR(epoch=100, pop_size=100, seed=10)
    model.solve(problem)
    print(f"Best agent: {model.g_best}")                    # Encoded solution
    print(f"Best solution: {model.g_best.solution}")        # Encoded solution
    print(f"Best fitness: {model.g_best.target.fitness}")
    # Decoded (Real) solution
    print(f"Best real scheduling: {model.problem.decode_solution(model.g_best.solution)}")
    x_decoded = model.problem.decode_solution(model.g_best.solution)
    x = x_decoded["per_var"]
    visualize(data, x, path='schedule-aco.png', label=f"ACO makespan: {model.g_best.target.fitness}")
    model.history.save_global_objectives_chart(filename="goc-aco")
    model.history.save_local_objectives_chart(filename="loc-aco")


def LRU_Scheduling(data):
    lru_schedule = LRUSchedule(data)
    print("Solving scheduling with LRU policy")
    lru_schedule.solve()
    machine_schedule = lru_schedule.getSolution()
    schedule_makespan = lru_schedule.getMakespan()
    print(f"LRU makespan: {schedule_makespan}")
    visualizeLRU(data, machine_schedule, path='schedule-lru.png', label=f"LRU makespan: {schedule_makespan}")


def LRU_Scheduling_Avg_Longest_First(data):
    lru_schedule = LRUSchedule(data, "mean")
    print("Solving scheduling with LRU policy and placing jobs with longer mean runtime first")
    lru_schedule.solve()
    machine_schedule = lru_schedule.getSolution()
    schedule_makespan = lru_schedule.getMakespan()
    print(f"LRU mean longer runtime first makespan: {schedule_makespan}")
    visualizeLRU(data, machine_schedule, path='schedule-lru-avg.png',
                 label=f"LRU mean longer runtime first makespan: {schedule_makespan}")


def LRU_Scheduling_Median_Longest_First(data):
    lru_schedule = LRUSchedule(data, "median")
    print("Solving scheduling with LRU policy and placing jobs with longer median runtime first")
    lru_schedule.solve()
    machine_schedule = lru_schedule.getSolution()
    schedule_makespan = lru_schedule.getMakespan()
    print(f"LRU median longer runtime first makespan: {schedule_makespan}")
    visualizeLRU(data, machine_schedule, path='schedule-lru-median.png',
                 label=f"LRU median longer runtime first makespan: {schedule_makespan}")


def LRU_Scheduling_Optimized_Shortest_First(data):
    lru_schedule = OptimizedLRUSchedule(data, True)
    print("Solving scheduling with Optimized LRU policy, with shortest job first")
    lru_schedule.solve()
    machine_schedule = lru_schedule.getSolution()
    schedule_makespan = lru_schedule.getMakespan()
    print(f"LRU optimized makespan shortest job first: {schedule_makespan}")
    visualizeLRU(data, machine_schedule, path='schedule-opt-lru-short-first.png',
                 label=f"LRU optimized makespan shortest job first: {schedule_makespan}")


def LRU_Scheduling_Optimized_Longest_First(data):
    lru_schedule = OptimizedLRUSchedule(data, False)
    print("Solving scheduling with Optimized LRU policy, with longest job first")
    lru_schedule.solve()
    machine_schedule = lru_schedule.getSolution()
    schedule_makespan = lru_schedule.getMakespan()
    print(f"LRU optimized makespan longest job first: {schedule_makespan}")
    visualizeLRU(data, machine_schedule, path='schedule-opt-lru-long-first.png',
                 label=f"LRU optimized makespan longest job first: {schedule_makespan}")


class JobShopProblem(Problem):
    def __init__(self, bounds=None, minmax="min", data=None, **kwargs):
        self.data = data
        super().__init__(bounds, minmax, **kwargs)

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded["per_var"]
        machine_times = np.zeros(self.data["n_machines"])  # Total time for each machine
        # DEBUG: x is the permutation of jobs, calculation here is wrong.
        for machine_idx, job_idx in enumerate(x):
            machine_idx = int(machine_idx) % (self.data["n_machines"])  # Apply modulo operation
            machine_times[machine_idx] += job_times[job_idx][machine_idx]
        makespan = np.max(machine_times)
        return np.max(makespan)


if __name__ == "__main__":

    parser = ArgumentParser(description="Scheduling Experiments")
    parser.add_argument("-m", "--machines",
                        metavar="INT",
                        type=int,
                        default=10,
                        help="Number of machines. (Default: 10)")
    parser.add_argument("-j", "--jobs", metavar="INT", type=int, default=300, help="Number of jobs. (Default: 300)")
    parser.add_argument("-s", "--seed", metavar="INT", type=int, default=1, help="Numpy seed. (Default: 1)")

    args = parser.parse_args()
    job_times = create_random_job_scheduling_problem(n_jobs=args.jobs, n_machines=args.machines, seed=args.seed)
    print(job_times.shape)
    print(job_times)

    n_jobs = job_times.shape[0]
    n_machines = job_times.shape[1]

    data = {
        "job_times": job_times,
        "n_jobs": n_jobs,
        "n_machines": n_machines,
        "log_to": None
    }

    LRU_Scheduling(data)  # Naive LRU policy, first in first out on next available host
    # Short jobs based on their AVG runtime, then first in first out on next available host
    LRU_Scheduling_Avg_Longest_First(data)
    # Short jobs based on their Median runtime, then first in first out on next available host
    LRU_Scheduling_Median_Longest_First(data)
    # This is the optimal LRU policy, placing the fastest job at the available host
    LRU_Scheduling_Optimized_Shortest_First(data)
    # This is the worst of all since it peaks the slowest job at the available host - we can disregard
    LRU_Scheduling_Optimized_Longest_First(data)

    # exit()

    bounds = PermutationVar(valid_set=list(range(0, n_jobs)), name="per_var")
    problem = JobShopProblem(bounds=bounds, minmax="min", data=data)

    PSO_Scheduling(problem)
    ACOR_Scheduling(problem)
