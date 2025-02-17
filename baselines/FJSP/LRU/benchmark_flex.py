#!/usr/bin/env python
import argparse
import os
import time
from glob import glob
from pathlib import Path
from typing import Dict, List

from main_flex import solve

from benchmark_fjsp.utils import read_fjsp_file
from utils import store


def find_fjs_files(root_dir: str) -> List[str]:
    """Find all .fjs files recursively in directory"""
    return [str(p) for p in Path(root_dir).rglob('*.fjs')]


def run_benchmark(file_path: str) -> Dict:
    """Run benchmark on a single file"""
    data = read_fjsp_file(file_path)
    tic = time.time()
    makespan = solve(data[2], args=None)
    toc = time.time()

    return {
        'file': file_path,
        'makespan': makespan,
        'time': toc - tic,
        'jobs': len(data[2]),
        'machines': len(data[2][0])
    }


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data-dir', default='./benchmark_fjsp/Monaldo/Fjsp/Job_Data',
    #                     help='Root directory containing benchmark files')
    # parser.add_argument('--output', required=True,
    #                     help='Output file to store results')
    # args = parser.parse_args()

    # Find all benchmark files
    # files = find_fjs_files(args.data_dir)
    # print(f'Found {len(files)} benchmark files')

    # Run benchmarks
    results = []
    # files = glob("./benchmark_fjsp/**/*.fjs", recursive=True)
    files = glob("./benchmark_fjsp/Monaldo/Fjsp/Job_Data/Dauzere_Data/Text/*.fjs", recursive=True)

    for fn in files:
        print(fn)
        n_jobs, n_machines, jobs = read_fjsp_file(fn)
        tic = time.time()
        makespan = solve(jobs, n_jobs, n_machines, args=dict())
        toc = time.time()
        print(f"{n_jobs}, {n_machines}, makespan: {makespan}, time: {toc - tic:.4f}")
        # print(f'Processing {file_path}...')
        # try:
        #     result = run_benchmark(file_path)
        #     results.append(result)
        #     print(f'  Makespan: {result["makespan"]}, Time: {result["time"]:.2f}s')
        # except Exception as e:
        #     print(f'  Error processing {file_path}: {str(e)}')
        #     continue

    # Store results
    # store(args.output, {
    #     'solver': 'flex',
    #     'results': results,
    #     'timestamp': time.time()
    # })

    # print(f'Benchmark complete. Results saved to {args.output}')


if __name__ == '__main__':
    main()
