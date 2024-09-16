""" Mealpy example with precedence constraints:

* [x] define the variable as floatvar, S_{ij} starting time of operation ij
* [x] precedence constraint: S_{ij} + P_{ij} <= S_{i, j+1}
* [x] no overlap constraint: S_{ij} + P_{ij} <= S_{kl} or S_{kl} + P_{kl} <= S_{ij}
    which is equivalent to (S_{ij} + P_{ij} - S_{kl}) (S_{kl} + P_{kl} - S_{ij}) < 0
"""

import argparse
from collections import defaultdict
from copy import deepcopy

from matplotlib.pylab import f
import networkx as nx
import numpy as np
from mealpy import ACOR, PermutationVar, Problem, FloatVar

from benchmark.utils import read_file
from ortools_api import ortools_api
from utils import convert_to_nx
import random
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from joblib import Parallel, delayed


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

        ub = kwargs['ub']
        # NOTE: define a set of float variables. Each variable represents starting time of a node in disjunctive graph.
        bounds = [FloatVar(lb=0, ub=ub, name=f"s_{i}_{j}") for i in range(n_jobs) for j in range(n_machines)] + \
            [FloatVar(lb=0, ub=0, name='s'), FloatVar(lb=0, ub=ub, name='t')]

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

    def compute_makespan(self, x):
        r""" Compute the makespan of the given permutation.

        Args:
            x (list): variables
        """
        print(x['t'], end=" ")
        return x['t'] + sum(self.precedence_const(x)) + sum(self.no_overlap_const(x))

    def violate(self, value):
        r""" Violation function.
        ..math::
            \max(0, value^2)

        Args:
            value (float): value

        Returns:
            float: violation value
        """
        return 0 if value < 0 else value**2

    def precedence_const(self, x):
        r""" Precedence constraints on conjunctive edges.

        Args:
            x (list): variables

        Returns:
            list: list of violated constraints values
        """
        def _const1(u, v):
            r""" Precedence constraint on conjunctive graph. Edge from u to v.

            Args:
                u (tuple): node u
                v (tuple): node v
            """
            if u == "s":
                j_v, m_v = v
                return x['s'] + self.C.nodes['s'].get('duration') - x[f"s_{j_v}_{m_v}"]
            elif v == "t":
                j_u, m_u = u
                return x[f"s_{j_u}_{m_u}"] + self.C.nodes[u].get("duration") - x['t']
            else:
                j_u, m_u = u
                j_v, m_v = v
                return (x[f"s_{j_u}_{m_u}"] + self.C.nodes[u].get("duration") - x[f"s_{j_v}_{m_v}"])

        # precedence constraint on conjunctive graph
        const_res = []
        for u, v in self.C.edges:
            self.C[u].get("duration")
            const_res.append(_const1(u, v))
        return [self.violate(val) for val in const_res]

    def no_overlap_const(self, x):
        r""" No overlap constraint on disjunctive edges.

        Args:
            x (list): variables
        """
        def _const2(u, v):
            j_u, m_u = u
            j_v, m_v = v
            return (x[f"s_{j_u}_{m_u}"] + self.C.nodes[u].get("duration") - x[f"s_{j_v}_{m_v}"]) * \
                (x[f"s_{j_v}_{m_v}"] + self.C.nodes[v].get("duration") - x[f"s_{j_u}_{m_u}"])

        # no overlap constraint on disjunctive graph
        const_res = []
        for u, v in self.D.edges:
            const_res.append(_const2(u, v))
        return [self.violate(val) for val in const_res]


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
        # ACOR iteration
        super().evolve(epoch)

        # update the population with local search
        pop_new = []
        if epoch % 10 == 0:
            for i in range(len(self.pop)):
                # pop_new.append(self.improve(self.pop[i]))
                # pop_new.append(self.improve_v2(self.pop[i]))
                # pop_new.append(self.improve_v3(self.pop[i]))
                pop_new.append(self.improve_v4(self.pop[i]))
        pop_new = self.update_target_for_population(pop_new)
        # maintain the same population size
        self.pop = self.get_sorted_and_trimmed_population(self.pop + pop_new, self.pop_size, self.problem.minmax)
        if self.verbose:
            print('iter', epoch)

    def improve(self, agent):
        r""" Improve the solution by examing the variables

        Notes:
          - It will go through all the machines and swap two adjacent jobs, and improve consistantly.
          - O(m * n) time complexity
        Args:
            agent (Agent): agent to improve
        """
        best_makespan = self.problem.obj_func(agent.solution)
        best = agent.solution
        changed = False

        decoded = self.problem.decode_solution(agent.solution)
        # NOTE: O(m * n) time complexity
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

    def improve_v2(self, agent, k=20):
        r""" Improve the solution by examing the variables

        Notes:
          - It will random select a set of tuple to swap
          - O(k) time complexity
        Args:
            agent (Agent): agent to improve
        """
        best_makespan = self.problem.obj_func(agent.solution)
        best = agent.solution
        changed = False

        decoded = self.problem.decode_solution(agent.solution)
        # NOTE: O(m * n) time complexity

        def generate_random_tuples(data, k):
            r""" Generate random tuples from the decoded solution."""
            result = []
            keys = list(data.keys())

            for _ in range(k):
                key = random.choice(keys)
                values = data[key]
                if len(values) < 2:
                    continue  # Skip if there are not enough elements to form a pair
                idx = random.randint(0, len(values) - 2)
                pair = (values[idx], values[idx + 1])
                result.append((key, pair))

            return result

        random_tuples = generate_random_tuples(decoded, k)
        for m_id, (idx1, idx2) in random_tuples:
            candidate = deepcopy(decoded)
            candidate[m_id][idx1], candidate[m_id][idx2] = candidate[m_id][idx2], candidate[m_id][idx1]
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

    def improve_v3(self, agent):
        r""" Improve the solution by examing the variables

        Notes:
          - Parallize the search with ThreadPoolExecutor
          - O(k) time complexity
        Args:
            agent (Agent): agent to improve
        """
        best_makespan = self.problem.obj_func(agent.solution)
        best = agent.solution
        changed = False

        decoded = self.problem.decode_solution(agent.solution)
        futures = []

        def evaluate_swap(m_id, idx, decoded, problem):
            r""" Evaluate the swap of two adjacent jobs. """
            candidate = deepcopy(decoded)
            # change its variable
            candidate[m_id][idx], candidate[m_id][idx + 1] = candidate[m_id][idx + 1], candidate[m_id][idx]
            encoded = problem.encode_solution([candidate[v.name] for v in problem.bounds])
            new_makespan = problem.obj_func(encoded)
            return new_makespan, encoded

        # With ThreadPoolExecutor: 1.1 s per epoch
        with ThreadPoolExecutor() as executor:
            for m_id in decoded:
                for idx in range(len(decoded[m_id]) - 1):
                    futures.append(executor.submit(evaluate_swap, m_id, idx, decoded, self.problem))

            for future in as_completed(futures):
                new_makespan, encoded = future.result()
                if new_makespan < best_makespan:
                    best_makespan = new_makespan
                    best = encoded
                    changed = True

        if changed:
            if self.verbose:
                print('imp')
            return self.generate_empty_agent(best)

        return agent

    def improve_v4(self, agent):
        r""" Improve the solution by examing the variables

        Notes:
          - Parallize the search process by using joblib
          - O(k) time complexity
        Args:
            agent (Agent): agent to improve
        """
        best_makespan = self.problem.obj_func(agent.solution)
        best = agent.solution
        changed = False

        decoded = self.problem.decode_solution(agent.solution)
        futures = []

        def evaluate_swap(m_id, idx, decoded, problem):
            r""" Evaluate the swap of two adjacent jobs. """
            candidate = deepcopy(decoded)
            # change its variable
            candidate[m_id][idx], candidate[m_id][idx + 1] = candidate[m_id][idx + 1], candidate[m_id][idx]
            encoded = problem.encode_solution([candidate[v.name] for v in problem.bounds])
            new_makespan = problem.obj_func(encoded)
            return new_makespan, encoded

        # Use joblib to parallelize the evaluation of swaps
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_swap)(m_id, idx, decoded, self.problem)
            for m_id in decoded
            for idx in range(len(decoded[m_id]) - 1)
        )

        for new_makespan, encoded in results:
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
    # hyperparameters
    parser.add_argument('--epoch', type=int, default=300, help="Maximum number of iterations")
    parser.add_argument('--pop_size', type=int, default=10, help="Number of population size")
    parser.add_argument('--sample_count', type=int, default=25, help="Number of newly generated samples")
    parser.add_argument('--intent_factor', type=float, default=0.5, help="Intensification factor (selection pressure)")
    parser.add_argument('--zeta', type=float, default=1.0, help="Deivation-distance ratio")

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
    ub = np.array(times).sum()
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
    p = DGProblem(g1, g2, n, m, log_to=args.log_to, ub=ub)
    model = ACOR.OriginalACOR(epoch=args.epoch, pop_size=args.pop_size)
    model.solve(p, mode="swarm", n_workers=48)
    print("ACO - best solution", model.g_best.target.fitness)

    # model_ls = ACORLocalSearch(epoch=args.epoch, pop_size=args.pop_size)
    # model_ls.solve(p, mode="swarm", n_workers=48)
    # print("ACO + LS - best solution", model_ls.g_best.target.fitness)

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


if __name__ == '__main__':
    np.random.seed(42)
    main()
