#!/usr/bin/env python
import argparse
import networkx as nx
import numpy as np
import itertools
import matplotlib.pyplot as plt
from mealpy import PermutationVar, ACOR, PSO, Problem

class DGProblem(Problem):
    def __init__(self, C: nx.DiGraph, D: nx.Graph):
        self.C = C
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
        dag = self.C.copy()
        assert isinstance(dag, nx.DiGraph)
        for i in range(len(x)):
            start = self.enumerated_nodes[x[i]]
            for j in range(i+1, len(x)):
                end = self.enumerated_nodes[x[j]]
                if self.D.has_edge(start, end):
                    dag.add_edge(start, end)

        return dag

    def compute_makespan(self, x):
        dag = self.build_dag(x)
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

def convert_to_nx(times, machines, n_jobs, n_machines):
    print(f'jobs={n_jobs}, machines={n_machines}')
    dep_graph = nx.DiGraph()
    dep_graph.add_node('s', duration=0)
    dep_graph.add_node('t', duration=0)
    for job, step in itertools.product(range(n_jobs), range(n_machines)):
        machine = machines[job][step] - 1
        prev_machine = machines[job][step - 1] - 1 if step > 0 else None
        duration = times[job][step]
        dep_graph.add_node((job, machine), duration=duration)
        if prev_machine is not None:
            dep_graph.add_edge((job, prev_machine), (job, machine))
        else:
            dep_graph.add_edge('s', (job, machine))
        if step == n_machines - 1:
            dep_graph.add_edge((job, machine), 't')

    res_graph = nx.Graph()
    for job, machine in itertools.product(range(n_jobs), range(n_machines)):
        res_graph.add_node((job, machine))
    for job1, job2, machine in itertools.product(range(n_jobs), range(n_jobs), range(n_machines)):
        if job1 < job2:
            res_graph.add_edge((job1, machine), (job2, machine))

    return dep_graph, res_graph

def load_network_fn(fn: str, sid: int):
    data = np.load(fn)
    print(data.shape)
    times, machines = data[sid]
    print(times.shape)
    n_jobs, n_machines = data.shape[2:]
    return convert_to_nx(times, machines, n_jobs, n_machines)

def draw_networks(g1, g2=None):
    with plt.style.context('ggplot'):
        pos = nx.spring_layout(g1, seed=7)
        nx.draw_networkx_nodes(g1, pos, node_size=7)
        nx.draw_networkx_labels(g1, pos)
        nx.draw_networkx_edges(g1, pos, arrows=True)
        if g2:
            nx.draw_networkx_edges(g2, pos, edge_color='r')
        plt.draw()
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--pos', type=int, default=0)
    args = parser.parse_args()

    g1, g2 = load_network_fn(args.data, args.pos)
#    draw_networks(g1, g2)
    p = DGProblem(g1, g2)
    model = ACOR.OriginalACOR(epoch=100, pop_size=100, seed=10)
    model.solve(p)
    print(model.g_best.solution)

    res_dag = p.build_dag([int(x) for x in model.g_best.solution])
    draw_networks(res_dag)

if __name__ == '__main__':
    main()

