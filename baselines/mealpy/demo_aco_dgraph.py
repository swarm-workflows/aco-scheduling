""" Mealpy example with precedence constraints:

* [x] add precedence constraints
* [x] build graph with conjunctive graph (nx.DiGraph) and disjunctive graph (nx.Graph)
* [x] add a set of permutation variables for disjunctive graph
* [x] check DAG
* [x] compute makespan
* [x] provide OR-Tools as baseline (A constraint programming solver, `pip install -U ortools`)
"""
import argparse
from collections import defaultdict
from copy import deepcopy

import networkx as nx
import numpy as np
from matplotlib.pylab import f
from mealpy import ACOR, PermutationVar, Problem
from jsp.aco import ACO
from jsp.disjunctive_graph import DisjunctiveGraph

from benchmark.utils import read_file
from ortools_api import ortools_api
from utils import convert_to_nx, plot_aco_vs_ortools


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default="ft")
    parser.add_argument('--id', type=str, default="06")
    parser.add_argument('--format', type=str, default="taillard", choices=["standard", "taillard"])
    parser.add_argument('--log_to', type=str, default="console", choices=["console", "file"])

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

    print(f'times: {times}\nmachines:{machines}')
    jobs = []
    for idx in range(0, len(machines[0])):
        tasks = [(machines[idx][i], times[idx][i]) for i in range(0, len(times[0]))]
        print(tasks)
        jobs.append(tasks)

    # g1, g2 = load_network_fn(args.data, args.pos)

    # case 1 example:
    # https://developers.google.com/optimization/scheduling/job_shop
    # n, m = 3, 3
    # times = np.array([[3, 2, 2], [2, 1, 4], [4, 3, 0]])
    # machines = np.array([[1, 2, 3], [1, 3, 2], [2, 3, 1]]) - 1

    # case2: random jobs (n=10, m=8)
    # n = 20
    # m = 10
    # times = np.random.randint(1, 10, (n, m))
    # machines = generate_random_machines(n, m)

    # case3: load from benchmark
    g1, g2 = convert_to_nx(times, machines, n, m)
    p = DisjunctiveGraph(jobs=jobs)
    model = ACO(graph=p)
    best_path, best_makespan = model.find_minimum_makespan(source='S', target='T', num_ants=100)
    print(f"ACO - best solution\npath: {best_path}, makespan: {best_makespan}")

    # solve the problem in ortools
    if isinstance(times, list):
        times = np.array(times)
    if isinstance(machines, list):
        machines = np.array(machines)
    solver = ortools_api(times, machines)

    # aco_best = []
    # for i in range(len(model.history.list_global_best)):
    #     aco_best.append(model.history.list_global_best[i].target.fitness)

    ortools_best = solver.objective_value
    print(ortools_best)
    # fig, axs = plt.subplots(figsize=(4, 4), tight_layout=True)
    # axs.plot(np.arange(len(aco_best)), aco_best, label='ACO')
    # axs.axhline(y=ortools_best, xmin=0, xmax=len(aco_best), color='r', linestyle='--', label="OR-Tools")
    # axs.set_xlabel('Epoch')
    # axs.set_ylabel('Makespan')
    # axs.legend()
    # plt.savefig(f"ACO_vs_ORTools_{n}_{m}.png")
    # plt.show()


if __name__ == '__main__':
    np.random.seed(42)
    main()
