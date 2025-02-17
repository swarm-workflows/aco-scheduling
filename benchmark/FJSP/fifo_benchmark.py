import argparse
import csv
import os
from glob import glob
from time import time

import numpy as np
from tqdm import tqdm

from baselines.FJSP.fifo import LWRQueue, MWRQueue, Queue, solve
from benchmark.utils import read_fjsp_file


def run_benchmark(problem_file):
    """Run all variants on a single problem instance"""
    n_jobs, n_machines, jobs = read_fjsp_file(problem_file)
    instance = os.path.splitext(os.path.basename(problem_file))[0]

    results = {
        'instance': instance,
        'n_jobs': n_jobs,
        'n_machines': n_machines
    }

    # Run each variant
    for variant, queue_cls in [
        ('fifo', Queue),
        ('lwr', LWRQueue),
        ('mwr', MWRQueue)
    ]:
        tic = time()
        makespan = solve(jobs, n_jobs, n_machines, variant=variant)
        toc = time()

        results[f'{variant}_makespan'] = makespan
        results[f'{variant}_walltime'] = toc - tic

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="benchmark/FJSP/fifo_results.csv")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    results = []

    files = glob(f"{base_dir}/**/*.fjs", recursive=True)

    # Run benchmark on each file
    for fn in tqdm(sorted(files)):
        try:
            result = run_benchmark(fn)
            results.append(result)
            print(f"Completed {result['instance']}: "
                  f"FIFO={result['fifo_makespan']}, "
                  f"LWR={result['lwr_makespan']}, "
                  f"MWR={result['mwr_makespan']}")
        except Exception as e:
            print(f"Error processing {fn}: {e}")

    # Save results to CSV
    fieldnames = ['instance', 'n_jobs', 'n_machines',
                  'fifo_makespan', 'fifo_walltime',
                  'lwr_makespan', 'lwr_walltime',
                  'mwr_makespan', 'mwr_walltime']

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
