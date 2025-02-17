import argparse
import os
from glob import glob
from time import time

import numpy as np

from benchmark.utils import read_file
from lru_api import lru_api
from utils import store


def solve_jobshop(durations, machines, variant='lru'):
    """Solve jobshop problem using LRU/FIFO approach"""
    # Convert inputs to numpy arrays
    durations = np.array(durations)
    machines = np.array(machines)

    solver = lru_api(durations, machines, variant=variant)
    return True, solver.objective_value, solver


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", type=str, default="abz")
    parser.add_argument("--id", type=str, default="5")
    parser.add_argument("--format", type=str, default="taillard", choices=["standard", "taillard"])
    parser.add_argument("--variant", type=str, default="lru", choices=["lru", "fifo"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--store", type=str)
    args = parser.parse_args()
    base_dir = os.path.join(os.path.dirname(__file__), "../benchmark")

    if args.all:
        if args.format == "taillard":
            files = glob(f"{base_dir}/*/Taillard_specification/*.txt")
        else:
            files = glob(f"{base_dir}/*/*.txt")

        for fn in sorted(files[:]):
            durations, machines = read_file(fn, problem=args.problem, id=args.id, format=args.format)
            n_jobs = len(durations)
            n_machines = len(durations[0])
            print(f"Solving {fn.split('/')[-1].split('.')[0]} {n_jobs} {n_machines}", end="\t")

            tic = time()
            optimal, solution, solver = solve_jobshop(durations, machines, variant=args.variant)
            toc = time()

            print(f"Time: {toc - tic:.2f}")
            if args.store:
                store(
                    os.path.join(args.store, f'lru_test_{args.problem}_{args.id}_{args.format}.json'),
                    {
                        'solver': f'lru_{args.variant}',
                        'solution': solution,
                        'optimal': optimal,
                        'time': solver.wall_time,
                        'problem': f'{args.format}_{args.problem}_{args.id}',
                        'times': n_jobs,
                        'machines': n_machines,
                    })
    else:
        if args.format == "standard":
            fn = f"{base_dir}/{args.problem}/{args.problem}{args.id}.txt"
        else:
            fn = f"{base_dir}/{args.problem}/Taillard_specification/{args.problem}{args.id}.txt"

        durations, machines = read_file(fn, problem=args.problem, id=args.id, format=args.format)
        n_jobs = len(durations)
        n_machines = len(durations[0])
        print(f"Solving {fn.split('/')[-1].split('.')[0]} {n_jobs} {n_machines}", end="\t")

        tic = time()
        optimal, solution, solver = solve_jobshop(durations, machines, variant=args.variant)
        toc = time()

        print(f"Time: {toc - tic:.2f}")
        print({
            'solver': f'lru_{args.variant}',
            'solution': solution,
            'optimal': optimal,
            'time': solver.wall_time,
            'problem': f'{args.format}_{args.problem}_{args.id}',
            'times': n_jobs,
            'machines': n_machines,
        })
        if args.store:
            store(args.store, {
                'solver': f'lru_{args.variant}',
                'solution': solution,
                'optimal': optimal,
                'time': solver.wall_time,
                'problem': f'{args.format}_{args.problem}_{args.id}',
                'times': n_jobs,
                'machines': n_machines,
            })
