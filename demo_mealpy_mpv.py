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

from benchmark.utils import read_file
from ortools_api import ortools_api
from utils import convert_to_nx


class DGProblem(Problem):
    r""" Disjunctive Graph Problem for Job Shop Scheduling. """

    def __init__(self, C: nx.DiGraph, D: nx.Graph, n_jobs: int, n_machines: int, **kwargs):
        r""" Initialize the problem.

        Args:
            C (nx.DiGraph): conjunctive graph
            D (nx.Graph): disjunctive graph
            n (int): number of jobs
            m (int): number of machines
        """
        # conjunctive graph - directed graph, disjunctive graph - undirected graph
        self.C, self.D = C, D
        self.n_jobs, self.n_machines = n_jobs, n_machines

        # group nodes by machines
        self.group_nodes = defaultdict(list)
        for node in D.nodes:
            self.group_nodes[node[1]].append(node)
        # check nodes in D is subset of nodes in C (except S, T)
        for node in D.nodes:
            assert C.has_node(node)

        # NOTE: define a set of mixed permutation variables
        bounds = [PermutationVar(valid_set=list(range(n_jobs)), name=f"m_{i}") for i in range(n_machines)]

        # logging
        log_to = kwargs.get('log_to', 'console')
        if log_to == 'file':
            log_file = kwargs.get('log_file', f"history_{n_jobs}_{n_machines}.txt")
            super().__init__(bounds=bounds, minmax='min', log_to='file', log_file=log_file)
        elif log_to == 'console':
            super().__init__(bounds=bounds, minmax='min', log_to='console')

        verbose = kwargs.get('verbose', False)

    def obj_func(self, x):
        r""" Objective function

        Args:
            x (list): variables
        """
        x_decoded = self.decode_solution(x)
        return self.compute_makespan(x_decoded)

    def build_dag(self, x):
        r""" Build DAG from variables.

        Args:
            x (dict): mix of permutation variables

        Returns:
            dag (nx.DiGraph): directed acyclic graph
        """
        dag = self.C.copy()
        assert isinstance(dag, nx.DiGraph)

        # enumerate machines
        for i in range(self.n_machines):
            # build directed edges in the disjunctive graph
            for idx in range(len(x[f"m_{i}"]) - 1):
                s_node, t_node = self.group_nodes[i][x[f"m_{i}"][idx]], self.group_nodes[i][x[f"m_{i}"][idx + 1]]
                dag.add_edge(s_node, t_node)
                if nx.is_directed_acyclic_graph(dag):
                    # print(f"add edge: {s_node} -> {t_node}")
                    continue
                else:
                    # print(f"edge not add {s_node} -> {t_node}")
                    dag.remove_edge(s_node, t_node)
        return dag

    def compute_makespan(self, x):
        r""" Compute the makespan of the given permutation.

        Args:
            x (list): variables
        """
        dag = self.build_dag(x)
        # nx.draw_networkx(dag)
        # plt.savefig("dag.png")
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
        # return node_dist["t"]


class ACORLocalSearch(ACOR.OriginalACOR):
    r""" ACO with Local Search.

    Local search strategy:
        * [x] swap two adjacent jobs in the permutation if it improves the makespan

    """

    def __init__(self, negate_var=None, *args, **kwargs):
        r""" Initialize the ACO with Local Search.

        Args:
            negate_var (str): variable to negate
        """
        super().__init__(*args, **kwargs)
        self.negate_var = negate_var
        self.verbose = kwargs.get("verbose", False)

    def evolve(self, epoch):
        r""" Evolve the population.

        Args:
            epoch (int): number of epochs
        """
        super().evolve(epoch)

        pop_new = []
        for i in range(len(self.pop)):
            pop_new.append(self.improve(self.pop[i]))
        pop_new = self.update_target_for_population(pop_new)
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_new, self.pop_size, self.problem.minmax)
        if self.verbose:
            print('iter', epoch)

    def improve(self, agent):
        r""" Improve the solution by examing the variables

        Args:
            agent (Agent): agent to improve
        """
        best_makespan = self.problem.obj_func(agent.solution)
        best = agent.solution
        changed = False

        decoded = self.problem.decode_solution(agent.solution)
        for m_id in decoded:
            for idx in range(len(decoded[m_id]) - 1):
                candidate = deepcopy(decoded)
                # change its variable
                candidate[m_id][idx], candidate[m_id][idx + 1] = candidate[m_id][idx + 1], candidate[m_id][idx]
                encoded = self.problem.encode_solution([candidate[v.name] for v in self.problem.bounds])
                new_makespan = self.problem.obj_func(encoded)
                if new_makespan < best_makespan:
                    best_makespan = new_makespan
                    best = encoded
                    changed = True

        if changed:
            if self.verbose:
                print('imp')
            return self.generate_empty_agent(best)

        return agent


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
    p = DGProblem(g1, g2, n, m, log_to=args.log_to)
    model = ACOR.OriginalACOR(epoch=300, pop_size=10, )
    model.solve(p, mode="swarm", n_workers=48)
    print("ACO - best solution", model.g_best.target.fitness)

    model_ls = ACORLocalSearch(epoch=300, pop_size=10)
    model_ls.solve(p, mode="swarm", n_workers=48)
    print("ACO + LS - best solution", model_ls.g_best.target.fitness)
    # res_dag = p.build_dag([int(x) for x in model.g_best.solution])
    # draw_networks(res_dag)

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
