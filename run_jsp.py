import argparse
import random

from benchmark.utils import read_file
from utils import store
from time import time
from jsp.disjunctive_graph import DisjunctiveGraph
from jsp.aco import ACO, ACO_LS
import numpy as np
#

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default="ft")
    parser.add_argument('--id', type=str, default="06")
    parser.add_argument('--format', type=str, default="taillard", choices=["standard", "taillard"])
    # hyperparameters
    parser.add_argument('--epoch', type=int, default=300, help="Maximum number of iterations")
    parser.add_argument('--store', type=str)
    parser.add_argument('--enable-ls', action='store_true')

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
    n, m = len(times), len(times[0])

    times = np.asarray(times, dtype=np.int64)
    machines = np.asarray(machines, dtype=np.int64)
    print(times)
    print(machines)

    jobs = np.zeros(list(times.shape) + [2], dtype=np.int64)
    jobs[:, :, 0] = machines
    jobs[:, :, 1] = times
    dg = DisjunctiveGraph(jobs=jobs, c_map="gist_rainbow")
    if args.enable_ls:
        aco = ACO_LS(dg, 100, num_iterations=args.epoch, ant_random_init=False)
    else:
        aco = ACO(dg, 100, num_iterations=args.epoch, ant_random_init=False)

    tic = time()
    aco_path, aco_cost = aco.find_minimum_makespan("S", "T", num_ants=4)
    toc = time()
    print(aco_path, aco_cost)
    if args.store:
        store(args.store, {
            'solver': 'aco_ls' if args.enable_ls else 'aco',
            'solution': int(aco_cost),
            'time': (toc - tic),
            'problem': f'{args.format}_{args.problem}_{args.id}',
            'times': n,
            'machines': m,
        })

if __name__ == '__main__':
    main()
