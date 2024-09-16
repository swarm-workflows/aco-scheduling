"""This model implements a simple jobshop named ft06.

A jobshop is a standard scheduling problem when you must sequence a
series of task_types on a set of machines. Each job contains one task_type per
machine. The order of execution and the length of each job on each
machine is task_type dependent.

The objective is to minimize the maximum completion time of all
jobs. This is called the makespan.
"""

import argparse
import collections
from glob import glob
from time import time

from ortools.sat.colab import visualization
from ortools.sat.python import cp_model

from .utils import read_file


def jobshop_problem(durations, machines) -> None:
    """Solves the jobshop from benchmark."""
    # Creates the solver.
    model = cp_model.CpModel()
    jobs_count = len(durations)
    machines_count = len(durations[0])

    all_machines = range(0, machines_count)
    all_jobs = range(0, jobs_count)

    # Computes horizon dynamically.
    horizon = sum([sum(durations[i]) for i in all_jobs])

    task_type = collections.namedtuple("task_type", "start end interval")

    # Creates jobs.
    all_tasks = {}
    for i in all_jobs:
        for j in all_machines:
            start_var = model.new_int_var(0, horizon, "start_%i_%i" % (i, j))
            duration = durations[i][j]
            end_var = model.new_int_var(0, horizon, "end_%i_%i" % (i, j))
            interval_var = model.new_interval_var(
                start_var, duration, end_var, "interval_%i_%i" % (i, j)
            )
            all_tasks[(i, j)] = task_type(
                start=start_var, end=end_var, interval=interval_var
            )

    # Create disjuctive constraints.
    machine_to_jobs = {}
    for i in all_machines:
        machines_jobs = []
        for j in all_jobs:
            for k in all_machines:
                if machines[j][k] == i:
                    machines_jobs.append(all_tasks[(j, k)].interval)
        machine_to_jobs[i] = machines_jobs
        model.add_no_overlap(machines_jobs)

    # Precedences inside a job.
    for i in all_jobs:
        for j in range(0, machines_count - 1):
            model.add(all_tasks[(i, j + 1)].start >= all_tasks[(i, j)].end)

    # Makespan objective.
    obj_var = model.new_int_var(0, horizon, "makespan")
    model.add_max_equality(
        obj_var, [all_tasks[(i, machines_count - 1)].end for i in all_jobs]
    )
    model.minimize(obj_var)

    # Solve the model.
    solver = cp_model.CpSolver()
    # solver.parameters.log_search_progress = True
    solver.parameters.max_time_in_seconds = 600.0
    status = solver.solve(model)

    # Output the solution.
    if status == cp_model.OPTIMAL:
        if visualization.RunFromIPython():
            starts = [
                [solver.value(all_tasks[(i, j)][0]) for j in all_machines]
                for i in all_jobs
            ]
            visualization.DisplayJobshop(starts, durations, machines, "FT06")
        else:
            print("Optimal makespan: %i" % solver.objective_value)
    else:
        # If not optimal, print the best solution found.
        print("Time limit, best known makespan: %i" % solver.ObjectiveValue())


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--problem", type=str, default="abz")
    args.add_argument("--id", type=str, default="5")
    args.add_argument("--format", type=str, default="standard", choices=["standard", "taillard"])
    args.add_argument("--all", action="store_true")
    args = args.parse_args()

    if args.all:
        if args.format == "taillard":
            files = glob("*/Taillard_specification/*.txt")
        else:
            files = glob("*/*.txt")

        for fn in sorted(files[:]):
            durations, machines = read_file(fn, problem=args.problem, id=args.id, format=args.format)
            n_jobs = len(durations)
            n_machines = len(durations[0])
            print(f"Solving {fn.split('/')[-1].split('.')[0]} {n_jobs} {n_machines}", end="\t")
            tic = time()
            jobshop_problem(durations, machines)
            toc = time()
            print(f"Time: {toc - tic:.2f}")
    else:
        if args.format == "standard":
            fn = f"./{args.problem}/{args.problem}{args.id}.txt"
        else:
            fn = f"./{args.problem}/Taillard_specification/{args.problem}{args.id}.txt"
        durations, machines = read_file(fn)
        n_jobs = len(durations)
        n_machines = len(durations[0])
        print(f"Solving {fn.split('/')[-1].split('.')[0]} {n_jobs} {n_machines}", end="\t")
        tic = time()
        jobshop_problem(durations, machines)
        toc = time()
        print(f"Time: {toc - tic:.2f}")
