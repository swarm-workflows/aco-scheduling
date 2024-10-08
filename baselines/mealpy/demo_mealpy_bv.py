""" Mealpy example with precedence constraints:

* [x] add precedence constraints
* [x] build graph with conjunctive graph (nx.DiGraph) and disjunctive graph (nx.Graph)
* [x] add binary variables for disjunctive graph
* [x] check DAG
* [x] compute makespan
* [x] provide OR-Tools as baseline (A constraint programming solver, `pip install -U ortools`)
"""
import argparse
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from mealpy import ACOR, BinaryVar, Problem

from benchmark.utils import read_file
from ortools_api import ortools_api
from utils import convert_to_nx, draw_networks, plot_aco_vs_ortools, store
from time import time

class DGProblem(Problem):
    def __init__(self, C: nx.DiGraph, D: nx.Graph, **kwargs):
        # conjunctive graph
        self.C = C
        # disjunctive graph
        self.D = D
        n_ants = kwargs.get("n_ants", 10)
        for node in D.nodes:
            assert C.has_node(node)
        self.enumerated_nodes = list(D.nodes)
        # NOTE: graph bounds as binary variables
        self.graph_bounds = BinaryVar(n_vars=len(D.edges), name='dir_var')

        # super().__init__(bounds=self.graph_bounds, minmax='min', log_to='console')
        super().__init__(bounds=self.graph_bounds, minmax='min', log_to='file', log_file=f"history_{n_ants}.txt")

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded['dir_var']
        return self.compute_makespan(x)

    def build_dag(self, x):
        r""" build dag from permutation.

        Args:
            x (list): permutation

        Returns:
            dag (nx.DiGraph): directed acyclic graph
        """
        dag = self.C.copy()
        assert isinstance(dag, nx.DiGraph)
        # for each edge in D, assign direction u->v if x[edge_idx] == 1, else v->u
        # check graph is acyclic per each operation, otherwise remove the edge
        for edge_idx, (u, v) in enumerate(self.D.edges()):
            if x[edge_idx] == 1:
                dag.add_edge(u, v)
                if nx.is_directed_acyclic_graph(dag):
                    continue
                else:
                    dag.remove_edge(u, v)
            else:
                dag.add_edge(v, u)
                if nx.is_directed_acyclic_graph(dag):
                    continue
                else:
                    dag.remove_edge(v, u)
        return dag

    def compute_makespan(self, x):
        r""" Compute the makespan of the given permutation.
        """
        dag = self.build_dag(x)
        # DEBUG: large graphs end with `inf` makespan
        if not nx.is_directed_acyclic_graph(dag):
            return np.inf

        node_dist = {}
        for node in nx.topological_sort(dag):
            node_w = dag.nodes[node]['duration']
            dists = [node_dist[n] for n in dag.predecessors(node)]
            if len(dists) > 0:
                node_w += max(dists)

            node_dist[node] = node_w
        return max(node_dist.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default="ft")
    parser.add_argument('--id', type=str, default="06")
    parser.add_argument('--format', type=str, default="taillard", choices=["standard", "taillard"])
    parser.add_argument('--n_ants', type=int, default=10)
    parser.add_argument('--store', type=str)
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
    if isinstance(times, list):
        times = np.array(times)
    if isinstance(machines, list):
        machines = np.array(machines)
    g1, g2 = convert_to_nx(times, machines, n, m)
    p = DGProblem(g1, g2, n_ants=args.n_ants)
    model = ACOR.OriginalACOR(epoch=3, pop_size=args.n_ants, )

    # model = PSO.OriginalPSO(epoch=100, pop_size=100, seed=10)
    tic = time()
    model.solve(p, mode="swarm", n_workers=24)
    toc = time()
    # print(model.g_best.solution)
    print("ACO - best solution", model.g_best.target.fitness)

    # model2 = PSO.OriginalPSO(epoch=100, pop_size=10, seed=10)
    # model2.solve(p, mode="swarm", n_workers=24, starting_solutions=np.array(
    #     [model.g_best.solution.tolist()] * 10))
    # # starting_solutions in (N*pop_size)
    # print("ACO+PSO - best solution", model2.g_best.target.fitness)

    res_dag = p.build_dag([int(x) for x in model.g_best.solution])
    draw_networks(res_dag)

    # solve the problem in ortools
    solver = ortools_api(times, machines)

    aco_best = []
    for i in range(len(model.history.list_global_best)):
        aco_best.append(model.history.list_global_best[i].target.fitness)

    ortools_best = solver.objective_value

    # plot_aco_vs_ortools(aco_best, ortools_best, fn=f"ACO_vs_ORTools_{args.problem}_{args.id}")

    if args.store:
        store(args.store, {
            'solver': 'demo_mealpy_bv',
            'solution': model.g_best.target.fitness,
            'time': (toc - tic),
            'problem': f'{args.format}_{args.problem}_{args.id}',
            'times': n,
            'machines': m,
            'ortools_best': ortools_best,
            'epoch': aco_best,
        })

if __name__ == '__main__':
    np.random.seed(42)
    main()
