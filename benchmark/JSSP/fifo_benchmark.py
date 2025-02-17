import argparse
import csv
import os
from glob import glob
from time import time

import numpy as np
from tqdm import tqdm

from baselines.JSSP.fifo import LWRQueue, MWRQueue, Queue, solve
from benchmark.utils import read_jssp_file


def run_benchmark(problem_file, format="taillard", problem="", id=""):
    """Run all variants on a single problem instance"""
    times, machines = read_jssp_file(problem_file, problem, id, format)
    n_jobs = len(times)
    n_machines = len(times[0])
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
        makespan = solve(times, machines, variant=variant)
        toc = time()

        results[f'{variant}_makespan'] = makespan
        results[f'{variant}_walltime'] = toc - tic

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="fifo_results.csv")
    parser.add_argument("--format", type=str, default="taillard",
                        choices=["standard", "taillard"])
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    results = []

    # Get all benchmark files
    if args.format == "taillard":
        files = glob(f"{base_dir}/*/Taillard_specification/*.txt")
    else:
        files = glob(f"{base_dir}/*/*.txt")

    # Run benchmark on each file
    for fn in tqdm(sorted(files)):
        try:
            result = run_benchmark(fn, format=args.format)
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
