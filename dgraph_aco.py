#!/usr/bin/env python
import argparse

import networkx as nx
import numpy as np
from mealpy import ACOR, PSO, PermutationVar, Problem

from ortools_api import ortools_api
from utils import convert_to_nx, draw_networks, generate_random_machines


class DGProblem(Problem):
    def __init__(self, C: nx.DiGraph, D: nx.Graph):
        # conjunctive graph
        self.C = C
        # disjunctive graph
        self.D = D

        for node in D.nodes:
            assert C.has_node(node)
        self.enumerated_nodes = list(D.nodes)
        self.graph_bounds = PermutationVar(valid_set=list(range(len(self.enumerated_nodes))), name='per_var')

        super().__init__(bounds=self.graph_bounds, minmax='min')

    def obj_func(self, x):
        x_decoded = self.decode_solution(x)
        x = x_decoded['per_var']
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
        # enumerate nodes in conjunctive graph based on x,
        # if edge exist in DisjGraph, add edge to dag
        for i in range(len(x)):
            start = self.enumerated_nodes[x[i]]
            for j in range(i + 1, len(x)):
                end = self.enumerated_nodes[x[j]]
                # DEBUG: dag could be (start -> end) or (end -> start)
                if self.D.has_edge(start, end):
                    dag.add_edge(start, end)

                for u, v in [(start, end), (end, start)]:
                    dag.add_edge(u, v)
                    if nx.is_directed_acyclic_graph(dag):
                        break
                    else:
                        dag.remove_edge(u, v)
                        dag.add_edge(v, u)
                        if nx.is_directed_acyclic_graph(dag):
                            break
                        else:
                            dag.remove_edge(v, u)

                # if self.D.has_edge(start, end):
                #     for edge in [(start, end), (end, start)]:
                #         dag.add_edge(*edge)
                #         if nx.is_directed_acyclic_graph(dag):
                #             break
                #         dag.remove_edge(*edge)
                #     raise Exception("Invalid DAG")
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
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--data', required=True)
    # parser.add_argument('--pos', type=int, default=0)
    # args = parser.parse_args()

    # g1, g2 = load_network_fn(args.data, args.pos)

    # case 1 example:
    # https://developers.google.com/optimization/scheduling/job_shop
    # n, m = 3, 3
    # times = np.array([[3, 2, 2], [2, 1, 4], [4, 3, 0]])
    # machines = np.array([[1, 2, 3], [1, 3, 2], [2, 3, 1]])

    # case2: random jobs
    n = 10
    m = 8
    times = np.random.randint(1, 10, (n, m))
    machines = generate_random_machines(n, m)

    g1, g2 = convert_to_nx(times, machines, n, m)
    p = DGProblem(g1, g2)
    model = ACOR.OriginalACOR(epoch=100, pop_size=100, seed=10)
    # model = PSO.OriginalPSO(epoch=100, pop_size=100, seed=10)
    model.solve(p)
    print(model.g_best.solution)

    res_dag = p.build_dag([int(x) for x in model.g_best.solution])
    draw_networks(res_dag, fn="dgraph_aco")

    # solve the problem in ortools
    ortools_api(times, machines - 1)


if __name__ == '__main__':
    np.random.seed(42)
    main()
